import torch.distributed as dist
import torch
from datetime import timedelta


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def print_with_rank(message):
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        print("[{}/{}]: {}".format(rank, world_size, message), flush=True)
    else:
        print(message, flush=True)


def init_distributed(timeout=timedelta(seconds=3 * 60 * 60)):
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timeout)
    world_size = dist.get_world_size()
    local_rank = dist.get_rank()
    return local_rank, world_size


def gather_scalar(x, rank, world_size):
    x_tensor = torch.tensor(x).to(f"cuda:{rank}")
    if rank == 0:
        x_list = [torch.zeros_like(x_tensor) for _ in range(world_size)]
        dist.gather(x_tensor, x_list, 0)
        return [x.item() for x in x_list]
    else:
        dist.gather(x_tensor)
