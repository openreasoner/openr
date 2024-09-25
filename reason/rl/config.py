from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from peft import PeftConfig


@dataclass
class BaseConfig:
    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)

    def __getitem__(self, k):
        if hasattr(self, k):
            return getattr(self, k)
        else:
            raise KeyError("Have no attribute named {}.".format(k))

    def get(self, k, default):
        if hasattr(self, k):
            return getattr(self, k)
        else:
            return default


@dataclass
class ModelConfig(BaseConfig):
    model_path: str
    critic_model_path: Optional[str] = None
    cache_dir: Optional[str] = None
    model_arch_type: str = "causal"
    peft_config: Optional[Dict] = None
    value_state_dict_path: Optional[Dict] = None

    # config for critic model, 
    #  since we have "ValueHeadLLM" and "AutoModelForCausalLMWithValueHead"
    #  this may be removed in future versions.
    value_model_type_name: str="ValueHeadLLM"

    def __post_init__(self):
        from peft import get_peft_config

        if isinstance(self.peft_config, dict):
            self.peft_config = get_peft_config(self.peft_config)


@dataclass
class TokenizerConfig(BaseConfig):
    tokenizer_path: str
    padding_side: str = "left"
    truncation_side: str = "right"


@dataclass
class TrainConfig(BaseConfig):
    seq_length: int
    epochs: int
    micro_batch_size: Optional[int] = 4
    sft_micro_batch_size: Optional[int] = 4
    gradient_accumulation_steps: Optional[int] = 4

    n_rollout: int = 10
    n_problem_per_gpu_rollout: Optional[int] = 100
    n_step_per_rollout: Optional[int] = 20

    eval_interval: Optional[int] = 1
    eval_n_problem: Optional[int] = 1

    checkpoint_interval: int = 1

    gamma: float = 0.99
    gae_lambda: float = 0.95

    pure_sft: bool = False
    sft_loss_coef: Optional[float] = 1.0
    value_loss_coef: Optional[float] = 0.5
    train_epoch: Optional[int] = 1

    project_name: str = "MCTS_train"
    entity_name: Optional[str] = None
    group_name: Optional[str] = None

    checkpoint_dir: Optional[str] = "ckpts"
    save_optimizer: bool = True

    rollout_logging_dir: Optional[str] = None
    tracker: Optional[str] = "wandb"
    logging_dir: Optional[str] = None
    tags: Optional[List[str]] = field(default_factory=list)

    seed: int = 42

    minibatch_size: Optional[int] = None

    pre_sft_datapath: Optional[str] = None
    pre_onpolicy_datapath: Optional[str] = None
    pre_onpolicy_datapath_train_test: Optional[str] = None
    pre_onpolicy_datapath_test: Optional[str] = None

    onpolicy_per_problem_max_size: Optional[int] = 3
    sft_per_problem_max_size: Optional[int] = 5

    env_name: str = ""
    task_dataset_kwargs: dict = field(default_factory=dict)
    # task_dataset_kwargs is a dict that should store task-specific
    # kwargs for task_module.get_train_test_dataset
    # e.g. num_train_data: Optional[int] = 1000 is for envs.gsm8k


@dataclass
class MCTSConfig(BaseConfig):
    num_simulations: int = 20
    pb_c_base: float = 19652
    pb_c_init: float = 10
    root_dirichlet_alpha: float = 0.3
    root_noise_weight: float = 0.25


@dataclass
class OptimizerConfig(BaseConfig):
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulerConfig(BaseConfig):
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    warmup_ratio: Optional[float] = None
    num_warmup_steps: Optional[int] = None


# CAUTION: keep an eye on extra comma


@dataclass
class EnvConfig(BaseConfig):
    stop_str: str = "The answer is "
    max_actions: int = 2
    max_length: int = 6
    is_few_shot: bool = False
    generation_config: dict = field(default_factory=dict)


@dataclass
class FSDPConfig(BaseConfig):
    mixed_precision: bool = True
    use_fp16: bool = False
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    checkpoint_type: StateDictType = StateDictType.SHARDED_STATE_DICT
    # alternatively can use SHARDED_STATE_DICT save one file per rank, and can resize the world-size.
    fsdp_activation_checkpointing: bool = True
    pure_bf16: bool = True
    optimizer: str = "AdamW"


@dataclass
class RLConfig:
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    tokenizer: TokenizerConfig
    train: TrainConfig
    mcts: MCTSConfig
    env: EnvConfig
    fsdp: Optional[FSDPConfig] = None

    @classmethod
    def from_dict(cls, config: Dict):
        """
        Convert dictionary to TRLConfig.
        """
        return cls(
            model=ModelConfig.from_dict(config["model"]),
            tokenizer=TokenizerConfig.from_dict(config["tokenizer"]),
            optimizer=OptimizerConfig.from_dict(config["optimizer"]),
            scheduler=SchedulerConfig.from_dict(config["scheduler"]),
            train=TrainConfig.from_dict(config["train"]),
            mcts=MCTSConfig.from_dict(config["mcts"]),
            env=EnvConfig.from_dict(config["env"]),
            fsdp=FSDPConfig.from_dict(config["fsdp"]) if "fsdp" in config else None,
        )

    def to_dict(self):
        data = {
            "model": self.model.__dict__,
            "tokenizer": self.tokenizer.__dict__,
            "optimizer": self.optimizer.__dict__,
            "scheduler": self.scheduler.__dict__,
            "train": self.train.__dict__,
            "mcts": self.mcts.__dict__,
            "env": self.env.__dict__,
        }
        if self.fsdp is not None:
            data["fsdp"] = self.fsdp.__dict__

        return data
