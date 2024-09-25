import os
from typing import Optional, Union
import torch
from tsllm.distributed.utils import print_with_rank
from tsllm.model.modeling_prm import ValueHeadedLLM
from tsllm.model.modeling_actor_critic import AutoModelForCausalLMWithValueHead


def load_critic_model(
    critic_model_path: str,
    state_dict_path: Optional[str],
    device,
    value_model_type_name: str = "ValueHeadLLM",
) -> Union[ValueHeadedLLM, AutoModelForCausalLMWithValueHead]:
    ############ LOAD V MODEL ###################
    if value_model_type_name == "ValueHeadLLM":
        critic = ValueHeadedLLM.from_pretrained(critic_model_path).to(
            device=device, dtype=torch.bfloat16
        )
        # for some checkpoints stored by deepspeed with optimizer state
        #  need to load the state dict manually.
        if state_dict_path is not None:
            print_with_rank("Loading state dict from {}".format(state_dict_path))
            state_dict = torch.load(
                os.path.join(state_dict_path, "pytorch_model/mp_rank_00_model_states.pt"),
                map_location="cpu",
            )
            critic.base_model.load_state_dict(state_dict["module"], strict=False)
            critic.cat_head.load_state_dict(
                {
                    k.replace("cat_head.", ""): v
                    for k, v in state_dict["module"].items()
                    if k.startswith("cat_head.")
                }
            )
    elif value_model_type_name == "AutoModelForCausalLMWithValueHead":
        critic = AutoModelForCausalLMWithValueHead.from_pretrained(
            critic_model_path
        ).to(device=device, dtype=torch.bfloat16)

        if state_dict is not None:
            raise NotImplementedError
    else:
        raise ValueError(
            "Unknown value model type name {}.".format(value_model_type_name)
        )

    critic.eval()
    return critic
