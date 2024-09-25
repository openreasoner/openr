from enum import Enum
from functools import partial
import math
from typing import Optional
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LambdaLR


class OptimizerName(str, Enum):
    """Supported optimizer names"""

    ADAM: str = "adam"
    ADAMW: str = "adamw"
    # ADAM_8BIT_BNB: str = "adam_8bit_bnb"
    # ADAMW_8BIT_BNB: str = "adamw_8bit_bnb"
    SGD: str = "sgd"


def get_optimizer_class(name: OptimizerName):
    """
    Returns the optimizer class with the given name

    Args:
        name (str): Name of the optimizer as found in `OptimizerNames`
    """
    if name == OptimizerName.ADAM:
        return torch.optim.Adam
    if name == OptimizerName.ADAMW:
        return torch.optim.AdamW

    if name == OptimizerName.SGD.value:
        return torch.optim.SGD
    supported_optimizers = [o.value for o in OptimizerName]
    raise ValueError(
        f"`{name}` is not a supported optimizer. "
        f"Supported optimizers are: {supported_optimizers}"
    )


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return max(
        0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    )


def get_cosine_schedule_with_warmup(
    optimizer,
    num_training_steps: int,
    num_warmup_steps: Optional[int] = None,
    warmup_ratio: Optional[float] = None,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    assert (num_warmup_steps is not None) ^ (warmup_ratio is not None), (
        warmup_ratio,
        num_warmup_steps,
    )
    if warmup_ratio is not None:
        assert (
            warmup_ratio > 0 and warmup_ratio < 1.0
        ), "Invalid warmup ratio: {}".format(warmup_ratio)
        num_warmup_steps = int(num_training_steps * warmup_ratio)
    assert num_warmup_steps >= 0
    assert num_training_steps >= 0
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class SchedulerName(str, Enum):
    """Supported scheduler names"""

    COSINE_ANNEALING = "cosine_annealing"
    LINEAR = "linear"
    COSINE_WARMUP = "cosine_warmup"


def get_scheduler_class(name: SchedulerName):
    """
    Returns the scheduler class with the given name
    """
    if name == SchedulerName.COSINE_ANNEALING:
        return CosineAnnealingLR
    elif name == SchedulerName.LINEAR:
        return LinearLR
    elif name == SchedulerName.COSINE_WARMUP:
        return get_cosine_schedule_with_warmup
    supported_schedulers = [s.value for s in SchedulerName]
    raise ValueError(
        f"`{name}` is not a supported scheduler. "
        f"Supported schedulers are: {supported_schedulers}"
    )
