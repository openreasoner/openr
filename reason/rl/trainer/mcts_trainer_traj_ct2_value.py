import gc
from functools import partial
import math
import os
import re
from typing import List, Optional
import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader, DistributedSampler
import time
from tsllm.distributed.utils import print_rank_0, print_with_rank
from tsllm.envs import (
    get_default_critic_data_builder,
    get_env_datasets,
)
from tsllm.model import ValueHeadedLLM, AutoModelForCausalLMWithValueHead
from tsllm.rl.config import TrainConfig
from tsllm.rl.data.node_types_new import TrajBatch, TrajInstance
from tsllm.rl.data.traj_buffer import MultiTrajBuffer
from tsllm.rl.trainer.base_trainer import BaseMCTSTrainer
from tsllm.rl.trainer.opt_utils import get_scheduler_class
import tree as dm_tree
from tqdm import tqdm
import json


def loop_iter(loader):
    while True:
        for x in loader:
            yield x


def load_jsonl(file_path):
    data_list = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())
            data_list.append(data)
    return data_list


class AccelerateMCTSTrainer(BaseMCTSTrainer):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        if config.model.value_model_type_name == "ValueHeadLLM":
            self.model = self.setup_model(ValueHeadedLLM)
        elif config.model.value_model_type_name == "AutoModelForCausalLMWithValueHead":
            self.model = self.setup_model(AutoModelForCausalLMWithValueHead)
        else:
            raise ValueError(
                "Unknown value model type name {}.".format(
                    config.model.value_model_type_name
                )
            )
        # run in pure_bf16
        self.model = self.model.to(torch.bfloat16)
        self.opt = self.setup_optimizer()

        (
            self.model,
            self.opt,
        ) = self.accelerator.prepare(
            self.model,
            self.opt,
        )

        self.train_q_ds, self.test_q_ds = get_env_datasets(
            self.config.train.env_name, **self.config.train.task_dataset_kwargs
        )

        sampler = DistributedSampler(self.train_q_ds, shuffle=False)
        self.task_train_loader = DataLoader(
            self.train_q_ds, batch_size=1, sampler=sampler, shuffle=False
        )
        test_sampler = DistributedSampler(self.test_q_ds, shuffle=True)
        self.task_test_loader = DataLoader(
            self.test_q_ds, batch_size=1, sampler=test_sampler, shuffle=False
        )
        self.problem_train_iter = loop_iter(self.task_train_loader)
        self.problem_test_iter = loop_iter(self.task_test_loader)

        # store on_policy data
        self.onpolicy_buffer = MultiTrajBuffer(
            num=len(self.task_train_loader),
            per_problem_max_size=self.config.train.onpolicy_per_problem_max_size,
            pad_token_id=self.tokenizer.pad_token_id,
            buffer_name="onpolicy_value",
        )
        self.onpolicy_buffer.clear_all()

        self.onpolicy_train_test_buffer = MultiTrajBuffer(
            num=len(self.task_train_loader),
            per_problem_max_size=self.config.train.onpolicy_per_problem_max_size,
            pad_token_id=self.tokenizer.pad_token_id,
            buffer_name="onpolicy_value",
        )
        self.onpolicy_train_test_buffer.clear_all()

        self.onpolicy_test_buffer = MultiTrajBuffer(
            num=len(self.task_test_loader),
            per_problem_max_size=self.config.train.onpolicy_per_problem_max_size,
            pad_token_id=self.tokenizer.pad_token_id,
            buffer_name="onpolicy_value",
        )
        self.onpolicy_test_buffer.clear_all()

        # question2idx
        self.q2idx_dict = {}
        for idx, problem_inst in enumerate(self.task_train_loader):
            question = problem_inst["question"][0]
            self.q2idx_dict[question] = idx

        # question2idx
        self.q2idx_dict_test = {}
        for idx, problem_inst in enumerate(self.task_test_loader):
            question = problem_inst["question"][0]
            self.q2idx_dict_test[question] = idx

        build_env_offline_data_component_fn = get_default_critic_data_builder(
            self.config.train.env_name
        )
        # init onpolicy buffer with pre-collect examples
        if self.config.train.pre_onpolicy_datapath is not None:
            traj_dict_list = build_env_offline_data_component_fn(
                jsonl_path=self.config.train.pre_onpolicy_datapath,
                q2idx_dict=self.q2idx_dict,
                tokenizer=self.tokenizer,
                is_few_shot=self.config.env.is_few_shot,
            )
            for traj_dict in traj_dict_list:
                self.add_traj(buffer_to_add=self.onpolicy_buffer, **traj_dict)

            print_with_rank("finish onpolicy buffer initialization")

        if self.config.train.pre_onpolicy_datapath_train_test is not None:
            traj_dict_list = build_env_offline_data_component_fn(
                jsonl_path=self.config.train.pre_onpolicy_datapath_train_test,
                q2idx_dict=self.q2idx_dict,
                tokenizer=self.tokenizer,
                is_few_shot=self.config.env.is_few_shot,
            )
            for traj_dict in traj_dict_list:
                self.add_traj(
                    buffer_to_add=self.onpolicy_train_test_buffer, **traj_dict
                )
            print_with_rank("finish onpolicy train test buffer initialization")

        if self.config.train.pre_onpolicy_datapath_test is not None:
            traj_dict_list = build_env_offline_data_component_fn(
                jsonl_path=self.config.train.pre_onpolicy_datapath_test,
                q2idx_dict=self.q2idx_dict_test,
                tokenizer=self.tokenizer,
                is_few_shot=self.config.env.is_few_shot,
            )
            for traj_dict in traj_dict_list:
                self.add_traj(buffer_to_add=self.onpolicy_test_buffer, **traj_dict)

            print_with_rank("finish onpolicy test buffer initialization")

        self.scheduler = self.setup_scheduler()
        self.scheduler = self.accelerator.prepare(self.scheduler)
        self.setup_tracker()

        self.ct2_generator = None

    def setup_scheduler(self):
        """
        Returns a learning rate scheduler derived from an instance's TRLConfig
        """
        train_config: TrainConfig = self.config.train
        assert train_config.train_epoch == 1
        len_sft = torch.tensor(
            len(self.onpolicy_buffer), device=self.accelerator.device
        )
        avg_sft = self.accelerator.gather(len_sft).float()
        buffer_max_length = avg_sft.max().item()
        total_training_steps = train_config.epochs * int(
            buffer_max_length
            / train_config.micro_batch_size
            / train_config.gradient_accumulation_steps
            + 1
        )
        print_rank_0("Total Training Step: {}.".format(total_training_steps))
        scheduler_class = get_scheduler_class(self.config.scheduler.name)
        if "warmup" in self.config.scheduler.name:
            self.config.scheduler.kwargs["num_training_steps"] = int(
                total_training_steps
            )
        print_rank_0(self.config.scheduler.kwargs)
        scheduler = scheduler_class(self.opt, **self.config.scheduler.kwargs)

        return scheduler

    def prepare_training(self):
        del self.ct2_generator
        gc.collect()
        torch.cuda.empty_cache()
        self.ct2_generator = None

    def loss_fn(self, minibatch: Optional[TrajBatch] = None, setting="train"):
        stats = {}
        minibatch.to(self.accelerator.device)

        onpolicy_output = self.model(
            input_ids=minibatch.input_ids,
            #  labels=minibatch.label,
            attention_mask=minibatch.attn_mask,
        )
        value = onpolicy_output.value
        mask = minibatch.mask
        returns = minibatch.returns

        # value loss includes value loss + final step reward loss
        value_delta = (returns[mask != 0] - value[mask != 0]) ** 2
        value_loss = value_delta.mean()
        stats[f"{setting}/value_loss"] = value_loss.detach().item()

        # final step reward statistics
        reward_loss = (returns[mask == 2] - value[mask == 2]) ** 2
        reward_loss = reward_loss.mean()
        stats[f"{setting}/reward_loss"] = reward_loss.detach().item()

        loss = self.config.train.value_loss_coef * value_loss

        stats[f"{setting}/total_loss"] = loss.detach().item()
        return loss, stats

    def add_traj(
        self,
        idx: int,
        query_str: str,
        answer: str,
        value_index: np.array,
        reward_list: np.array,
        buffer_to_add: MultiTrajBuffer,
    ):
        # result 1 for right, -1 for wrong
        @torch.inference_mode()
        def policy_forward_seq_value(input_str):
            input_ids = self.tokenizer(input_str, return_tensors="pt").input_ids.to(
                self.accelerator.device
            )
            value = self.unwrap_critic_model(input_ids=input_ids).value
            return value.cpu().float().numpy()

        buffer_to_add.add(
            idx,
            TrajInstance.from_string(
                query_str,
                answer,
                value_index,
                reward_list,
                self.tokenizer,
                policy_forward_seq_value,
                self.config.train.gamma,
                self.config.train.gae_lambda,
                use_gae=False,
                cal_value=True,
            ),
        )

    def learn(self, iter_count=0):
        train_config: TrainConfig = self.config.train
        self.iter_count = iter_count
        self.train_step = 0

        for i_epoch in range(train_config.epochs):
            stats = {}

            print_with_rank("TRAINING")

            self.model.train()
            # if train_config.pure_sft:
            #    assert train_config.sft_loss_coef is not None
            train_dataloader = self.onpolicy_buffer.create_loader(
                train_config.micro_batch_size, shuffle=True
            )
            # train_sft_dataloader = self.sft_buffer.create_loader(
            #    train_config.sft_micro_batch_size, shuffle=True)
            train_data_iter = loop_iter(train_dataloader)
            # train_sft_data_iter = loop_iter(train_sft_dataloader)

            gas = self.config.train.gradient_accumulation_steps

            len_onpolicy = torch.tensor(
                len(train_dataloader), device=self.accelerator.device
            )
            avg_onpolicy = self.accelerator.gather(len_onpolicy).float()

            # all sft buffer size
            all_onpolicy_num = torch.sum(
                avg_onpolicy.sum() * train_config.micro_batch_size
            ).item()

            assert self.config.train.train_epoch == 1
            nga = int(
                int(math.ceil(avg_onpolicy.max().item() / gas))
                * self.config.train.train_epoch
            )

            train_stats_list = []

            t0 = time.time()
            for i_nga in tqdm(range(nga), disable=not self.local_rank == 0):
                for i_gas in range(gas):
                    loss, cur_step_stats = self.train_iter(train_data_iter)
                    train_stats_list.append(cur_step_stats)

                self.opt.step()
                self.scheduler.step()
                self.opt.zero_grad()
                stats_gas = dm_tree.map_structure(
                    lambda *xs: np.mean(xs).item(), *train_stats_list[-gas:]
                )

                stats_gas["train/learning_rate"] = self.scheduler.get_last_lr()[0]
                self.accelerator.log(stats_gas, step=self.train_step)
                self.train_step += 1

            t1 = time.time()

            train_stats = dm_tree.map_structure(
                lambda *xs: np.mean(xs).item(), *train_stats_list
            )

            stats.update(train_stats)

            stats["time/training_time"] = t1 - t0
            stats["train/sft_buffer_size"] = all_onpolicy_num

            if self.local_rank == 0:
                print_rank_0("LOSS: {:.4f}, {}".format(loss, stats))

            if self.iter_count % train_config.checkpoint_interval == 0:
                subfolder = f"checkpoint_{self.iter_count}_ep{i_epoch}"
                directory = os.path.join(train_config.checkpoint_dir, subfolder)
                print_with_rank(f"Saving intermediate checkpoint into {directory}")
                if train_config.save_optimizer:
                    self.save(directory)
                else:
                    self.save_pretrained(directory)
                self.save_pretrained(
                    os.path.join(train_config.checkpoint_dir, "last_model_hf")
                )
                self.save_config()

            if self.iter_count % train_config.eval_interval == 0:
                print_with_rank("EVALUATING iteration:{}".format(self.iter_count))
                eval_stats = self.evaluate()
                stats.update(eval_stats)

            # replace all key
            for key in list(stats.keys()):
                if "loss" in key:
                    stats[key.replace("loss", "average_loss")] = stats.pop(key)
            self.accelerator.log(stats, step=self.iter_count)

            self.iter_count += 1

    def train_iter(self, data_iter):
        forward_time = -time.time()
        # onpolicy data loss
        minibatch = next(data_iter)
        loss, stats = self.loss_fn(minibatch)
        forward_time += time.time()
        backward_time = -time.time()
        self.accelerator.backward(loss)
        backward_time += time.time()

        stats["time/forward_time"] = forward_time
        stats["time/backward_time"] = backward_time

        return loss.item(), stats

    @torch.inference_mode()
    def evaluate(self):
        self.model.eval()
        train_test_dataloader = self.onpolicy_train_test_buffer.create_loader(
            self.config.train.micro_batch_size, shuffle=False
        )
        test_dataloader = self.onpolicy_test_buffer.create_loader(
            self.config.train.micro_batch_size, shuffle=False
        )

        def evaluate_on_dataloader(dataloader, setting):
            print_with_rank(setting)
            stats_list = []
            for minibatch in tqdm(dataloader, disable=not self.local_rank == 0):
                _, stats = self.loss_fn(minibatch, setting=setting)
                stats_list.append(stats)
            if len(stats_list) == 0:
                stats_list.append(
                    {
                        f"{setting}/value_loss": 0.0,
                        f"{setting}/reward_loss": 0.0,
                        f"{setting}/total_loss": 0.0,
                    }
                )
            return stats_list

        train_test_stats = evaluate_on_dataloader(
            train_test_dataloader, setting="train_test"
        )
        train_test_stats = dm_tree.map_structure(
            lambda *xs: np.mean(xs), *train_test_stats
        )

        test_stats = evaluate_on_dataloader(test_dataloader, setting="test")
        test_stats = dm_tree.map_structure(lambda *xs: np.mean(xs), *test_stats)

        device = self.accelerator.device

        eval_stats = dict()
        for k, v in train_test_stats.items():
            tmp_tensor = torch.tensor(v, device=device)
            gather_tensor = self.accelerator.gather(tmp_tensor)
            eval_stats[k] = gather_tensor.mean().item()

        for k, v in test_stats.items():
            tmp_tensor = torch.tensor(v, device=device)
            gather_tensor = self.accelerator.gather(tmp_tensor)
            eval_stats[k] = gather_tensor.mean().item()

        print_with_rank(eval_stats)
        return eval_stats
