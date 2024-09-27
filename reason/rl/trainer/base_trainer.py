from abc import abstractmethod, ABC
from ast import List
from functools import partial
import json
import os
import sys
from typing import Callable, Iterable, Optional
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from distributed.utils import print_with_rank
from rl.trainer.opt_utils import get_optimizer_class
from rl.trainer.utils import flatten_dict, get_distributed_config, get_git_tag


class BaseMCTSTrainer(ABC):
    def __init__(self, config):
        self.config = config

        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.train.gradient_accumulation_steps,
            log_with=config.train.tracker,
            project_dir=config.train.logging_dir,
            split_batches=True,
        )

        self.accelerator.state.deepspeed_plugin.deepspeed_config[
            "train_micro_batch_size_per_gpu"
        ] = config.train.micro_batch_size

        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            torch.distributed.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", 0))])

        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer.tokenizer_path, cache_dir=self.config.model.cache_dir
        )
        self.tokenizer.padding_side = config.tokenizer.padding_side
        self.tokenizer.truncation_side = config.tokenizer.truncation_side

        # FIXME: this may not be right and not general for all tokenizer.
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = "<|padding|>"
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.sep_token = "<sep>"

    @abstractmethod
    def learn(self):
        raise NotImplementedError

    @property
    def unwrap_model(self):
        # unwrap_model = self.model
        unwrap_model = self.accelerator.unwrap_model(self.actor_critic_model)  # .module
        return unwrap_model

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

    def save_config(self, directory: Optional[str] = None):
        dst_dir = directory or self.config.train.checkpoint_dir
        config_path = os.path.join(dst_dir, "trainer_config.json")
        if self.accelerator.is_main_process:
            json.dump(self.config.to_dict(), open(config_path, "w"), indent=2)
            print_with_rank("Saving config to {}".format(config_path))

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

    def setup_tracker(self):
        script_name = os.path.basename(sys.argv[0]).rsplit(".", 1)[0]
        if not isinstance(self.config.model.model_path, str):
            model_name = str(self.config.model.model_path).split()[0]
        else:
            model_name = self.config.model.model_path.split("/")[-1]

        if self.accelerator.num_processes == 1:
            num_gpus = "1gpu"
        else:
            num_gpus = f"{self.accelerator.num_processes}gpus"
        branch = get_git_tag()[0]

        run_name = "/".join([script_name, model_name, num_gpus]) + f":{branch}"

        if self.accelerator.is_main_process:
            config_dict = self.config.to_dict()
            dist_config = get_distributed_config(self.accelerator)
            config_dict["distributed"] = dist_config
            init_trackers_kwargs = {}

            if self.config.train.tracker == "wandb":
                init_trackers_kwargs["wandb"] = {
                    "name": run_name,
                    "entity": self.config.train.entity_name,
                    "group": self.config.train.group_name,
                    "tags": self.config.train.tags + ["/".join(get_git_tag())],
                    "mode": "disabled" if os.environ.get("debug", False) else "online",
                }

                self.accelerator.init_trackers(
                    project_name=self.config.train.project_name,
                    config=config_dict,
                    init_kwargs=init_trackers_kwargs,
                )
            elif self.config.train.tracker == "tensorboard":
                # flatten config for tensorboard, split list in hparams into flatten config
                if config_dict["model"].get(
                    "peft_config", None
                ):  # tensorboard does not support peft config type
                    config_dict["model"]["peft_config"] = str(
                        config_dict["model"]["peft_config"]
                    )
                config_dict_flat = flatten_dict(config_dict)
                config_dict_flat["optimizer/kwargs/beta_1"] = config_dict_flat[
                    "optimizer/kwargs/betas"
                ][0]
                config_dict_flat["optimizer/kwargs/beta_2"] = config_dict_flat[
                    "optimizer/kwargs/betas"
                ][1]
                config_dict_flat.pop("optimizer/kwargs/betas", None)
                for ix, tag in enumerate(config_dict_flat.pop("train/tags")):
                    config_dict_flat[f"train/tag_{ix}"] = tag

                self.accelerator.init_trackers(
                    project_name=self.config.train.project_name,
                    config=config_dict_flat,
                )
            elif self.config.train.tracker is None:
                self.accelerator.init_trackers(
                    project_name=self.config.train.project_name
                )
            else:
                raise ValueError(
                    f"Only supported trackers are `wandb` and `tensorboard`. Got: `{self.config.train.tracker}`. "
                    "Set `tracker` to `None` to disable tracking."
                )

    def setup_model(self, model_class=AutoModelForCausalLM):
        """
        Returns a model derived from an instance's TRLConfig
        """
        model = self.get_arch(self.config, model_class)

        if self.config.model.model_arch_type == "seq2seq":
            raise NotImplementedError

        return model

    def get_arch(self, config, model_class=AutoModelForCausalLM):
        from_fn = partial(
            model_class.from_pretrained, cache_dir=self.config.model.cache_dir
        )
        # backward-compat: Try to create a randomly initialized architecture from a config
        if issubclass(type(config.model.model_path), transformers.PretrainedConfig):
            from_fn = model_class.from_config

        return from_fn(config.model.model_path)

    def setup_optimizer(self):
        """
        Returns an optimizer derived from an instance's TRLConfig
        """
        optimizer_class = get_optimizer_class(self.config.optimizer.name)
        optimizer = optimizer_class(
            self.model.parameters(),
            **self.config.optimizer.kwargs,
        )
        return optimizer
