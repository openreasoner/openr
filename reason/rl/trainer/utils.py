import subprocess
from typing import Dict, MutableMapping, Tuple, Union

import accelerate


def flatten_dict(
    d: Union[dict, MutableMapping],
    parent_key: str = "",
    sep: str = "/",
) -> dict:
    # From: https://stackoverflow.com/a/6027615
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_distributed_config(accelerator: accelerate.Accelerator):
    """
    Return accelerator distributed config
    """

    dist_config = {
        "mixed_precision": accelerator.mixed_precision,
        "num_gpus": accelerator.num_processes,
    }

    if accelerator.state.deepspeed_plugin is not None:
        ds_plugin = accelerator.state.deepspeed_plugin
        dist_config.update(
            {
                "gradient_accumulation_steps": ds_plugin.gradient_accumulation_steps,
                "gradient_clipping": ds_plugin.gradient_clipping,
                "zero_stage": ds_plugin.zero_stage,
                "offload_optimizer_device": ds_plugin.offload_optimizer_device,
                "offload_param_device": ds_plugin.offload_param_device,
            }
        )

    return dist_config


def get_git_tag() -> Tuple[str, str]:
    """
    Returns commit's short hash and date
    """
    try:
        output = subprocess.check_output("git log --format='%h/%as' -n1".split())
        branch = subprocess.check_output("git rev-parse --abbrev-ref HEAD".split())
        return branch.decode()[:-1], output.decode()[1:-2]
    except subprocess.CalledProcessError:
        return "unknown", "unknown"
