import gc
import math
import os
from typing import List, Optional
import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader, DistributedSampler
import time
from tsllm.distributed.utils import print_rank_0, print_with_rank
from tsllm.envs import get_env_datasets, get_default_sft_data_builder
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

        self.model = self.setup_model()
        # run in pure_bf16
        self.model = self.model.to(self.accelerator.device, torch.bfloat16)
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
        print_with_rank(
            "#train tasks: {}, test_tasks: {}".format(
                len(self.train_q_ds), len(self.test_q_ds)
            )
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

        # store imitation data
        self.sft_buffer = MultiTrajBuffer(
            num=len(self.task_train_loader),
            per_problem_max_size=self.config.train.sft_per_problem_max_size,
            pad_token_id=self.tokenizer.pad_token_id,
            buffer_name="sft",
        )
        self.sft_buffer.clear_all()
        # question2idx
        self.q2idx_dict = {}
        for idx, problem_inst in enumerate(self.task_train_loader):
            question = problem_inst["question"][0]
            self.q2idx_dict[question] = idx
        # init sft buffer with pre-collect examples
        sft_data_builder_fn = get_default_sft_data_builder(self.config.train.env_name)
        if self.config.train.pre_sft_datapath is not None:
            # predata = load_jsonl(self.config.train.pre_sft_datapath)
            # for d in predata:
            #     question = d["question"]
            #     if question in self.q2idx_dict.keys():
            #         # because question here is split by distributed training,
            #         # so ignore those questions not stored in the dictionary
            #         task_idx = self.q2idx_dict[question]
            #         for a in d["answer"]:
            #             result = 1.0 if a["correct"] else 0.0
            #             if a["correct"]:
            #                 self.add_traj(
            #                     task_idx,
            #                     question,
            #                     a["text"],
            #                     result,
            #                     add_sft_buffer=True,
            #                 )
            sft_data_list = sft_data_builder_fn(
                jsonl_path=self.config.train.pre_sft_datapath,
                q2idx_dict=self.q2idx_dict,
                tokenizer=self.tokenizer,
                is_few_shot=self.config.env.is_few_shot,
                add_eos_token=True,
            )
            for sft_data in sft_data_list:
                self.add_traj(
                    sft_data["idx"],
                    query_str=sft_data["query_str"],
                    response_str=sft_data["response_str"],
                    result=1,
                )
            print_with_rank(
                "finish sft buffer initialization. size: {}".format(
                    len(self.sft_buffer)
                )
            )

        self.scheduler = self.setup_scheduler()
        self.scheduler = self.accelerator.prepare(self.scheduler)
        self.setup_tracker()

        self.ct2_generator = None

    @property
    def unwrap_model(self):
        # unwrap_model = self.model
        unwrap_model = self.accelerator.unwrap_model(self.model)  # .module
        return unwrap_model

    def setup_scheduler(self):
        """
        Returns a learning rate scheduler derived from an instance's TRLConfig
        """
        train_config: TrainConfig = self.config.train
        assert train_config.train_epoch == 1
        len_sft = torch.tensor(len(self.sft_buffer), device=self.accelerator.device)
        avg_sft = self.accelerator.gather(len_sft).float()
        sft_buffer_max_length = avg_sft.max().item()
        total_training_steps = train_config.epochs * int(
            sft_buffer_max_length
            / train_config.sft_micro_batch_size
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

    def loss_fn(self, sftbatch: Optional[TrajBatch] = None):
        stats = {}
        sftbatch.to(self.accelerator.device)
        # ============
        # sft policy loss
        # ============
        sft_output = self.model(
            input_ids=sftbatch.input_ids,
            labels=sftbatch.label,
            attention_mask=sftbatch.attn_mask,
            return_dict=True,
            use_cache=False,
        )

        policy_loss = sft_output.loss
        loss = self.config.train.sft_loss_coef * policy_loss
        stats["train/sft_policy_loss"] = policy_loss.detach().item()

        stats["train/total_loss"] = loss.detach().item()
        return loss, stats

    def add_traj(
        self,
        idx: int,
        query_str: str,
        response_str: str,
        result: int,
    ):
        # result 1 for right, 0 for wrong
        def policy_forward_seq_value(x):
            raise RuntimeError("IN SFT TRAINER YOU SHOULD NOT USE CRITIC TO COMPUTE V.")
        dummy_value_index = np.zeros(1)
        dummy_reward_list = np.ones(1)
        self.sft_buffer.add(
            idx,
            TrajInstance.from_string(
                query_str,
                response_str,
                dummy_value_index,
                dummy_reward_list,
                self.tokenizer,
                policy_forward_seq_value,
                self.config.train.gamma,
                self.config.train.gae_lambda,
                cal_value=False,
            ),
        )

    def learn(self, iter_count=0):
        train_config: TrainConfig = self.config.train
        self.iter_count = iter_count
        self.train_step = 0

        for i_epoch in range(train_config.epochs):
            stats = {}

            print_with_rank("TRAINING")
            self.prepare_training()

            self.model.train()
            train_sft_dataloader = self.sft_buffer.create_loader(
                train_config.sft_micro_batch_size, shuffle=True
            )
            train_sft_data_iter = loop_iter(train_sft_dataloader)

            gas = self.config.train.gradient_accumulation_steps

            # currently use the length of sft buffer as the number of training step
            len_sft = torch.tensor(
                len(train_sft_dataloader), device=self.accelerator.device
            )
            avg_sft = self.accelerator.gather(len_sft).float()

            # all sft buffer size
            all_sft_num = torch.sum(
                avg_sft.sum() * train_config.sft_micro_batch_size
            ).item()
            assert self.config.train.train_epoch == 1
            nga = int(
                int(math.ceil(avg_sft.max().item() / gas))
                * self.config.train.train_epoch
            )

            train_stats_list = []

            t0 = time.time()
            for i_nga in tqdm(range(nga), disable=not self.local_rank == 0):
                for i_gas in range(gas):
                    loss, stats = self.train_iter(train_sft_data_iter)
                    train_stats_list.append(stats)

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
            stats["train/sft_buffer_size"] = all_sft_num

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

            # replace all key
            for key in list(stats.keys()):
                if "loss" in key:
                    stats[key.replace("loss", "average_loss")] = stats.pop(key)
            self.accelerator.log(stats, step=self.iter_count)

            self.iter_count += 1

    def train_iter(self, train_sft_data_iter):
        forward_time = -time.time()
        # onpolicy data loss
        sftbatch = next(train_sft_data_iter)
        loss, stats = self.loss_fn(sftbatch)
        forward_time += time.time()
        backward_time = -time.time()
        self.accelerator.backward(loss)
        backward_time += time.time()

        stats["time/forward_time"] = forward_time
        stats["time/backward_time"] = backward_time

        return loss.item(), stats

    def save_pretrained(self, directory: Optional[str] = None, **kwargs):
        """Save the underlying Hugging Face model, tokenizer, and configuration files to a directory for
        later use.

        Args:
            directory (str, *optional*): The directory to save the trainer files to.
                NOTE: If not specified, the model will be saved to a directory named `hf_model` in the
                checkpoint directory as specified by the Trainer's config.
            **kwargs: Additional keyword arguments passed to the underlying Hugging Face model's
                `save_pretrained` method.
        """
        if directory is None:
            directory = os.path.join(self.config.train.checkpoint_dir, "hf_model")

        self.accelerator.wait_for_everyone()
        self.accelerator.unwrap_model(self.model).save_pretrained(
            directory,
            save_function=self.accelerator.save,
            is_main_process=self.accelerator.is_main_process,
            state_dict=self.accelerator.get_state_dict(self.model),
            **kwargs,
        )

        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(directory)

    def save(self, directory: Optional[str] = None, **kwargs):
        """Creates a checkpoint of the optimizer, scheduler and model"""
        dst_dir = directory or self.config.train.checkpoint_dir
        self.accelerator.save_state(dst_dir, **kwargs)

        if (
            self.config.model.peft_config is not None
            and self.accelerator.is_main_process
        ):
            # Remove "pytorch_model.bin" because it contains more than necessary,
            # let save_pretrained recreate it with just the value heads.
            model_file = os.path.join(dst_dir, "pytorch_model.bin")
            if os.path.exists(model_file):
                os.remove(model_file)
            self.accelerator.unwrap_model(self.model).save_pretrained(dst_dir)

    def load(self, directory: Optional[str] = None, **kwargs):
        """Load checkpoint of optimizer, scheduler and a model"""
        if self.config.model.peft_config is not None:

            def load_state_hook(models: List[torch.nn.Module], input_dir: str):
                with self.accelerator.main_process_first():
                    for model in models:
                        model.from_pretrained(input_dir)

            self.accelerator.register_load_state_pre_hook(load_state_hook)

            strict = False
        else:
            strict = True

        self.accelerator.load_state(
            directory or self.config.train.checkpoint_dir,
            load_module_strict=strict,
            **kwargs,
        )
