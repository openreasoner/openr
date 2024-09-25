import torch
from typing import Union, List
from tsllm.model import ValueHeadedLLM
from tsllm.model.modeling_actor_critic import AutoModelForCausalLMWithValueHead
from tsllm.llm.text_generation import llm_gen_ct2
from transformers import AutoTokenizer
import re
import numpy as np
import requests


@torch.inference_mode()
def value_fn(
    critic: ValueHeadedLLM, tokenizer: AutoTokenizer, input_str: Union[List[str], str]
):
    if isinstance(input_str, list):
        indices2pick = torch.LongTensor(
            [len(tokenizer.encode(txt)) - 1 for txt in input_str]
        )
    else:
        indices2pick = torch.LongTensor([len(tokenizer.encode(input_str)) - 1])

    # print(input_str)
    inputs = tokenizer(input_str, return_tensors="pt", padding=True).to(critic.device)
    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")
    value = critic(**inputs).value.cpu()
    value = value.gather(1, indices2pick.unsqueeze_(1)).squeeze_(1).float().numpy()
    return value


@torch.inference_mode()
def value_fn_rlhf(
    critic: AutoModelForCausalLMWithValueHead,
    tokenizer: AutoTokenizer,
    input_str: Union[List[str], str],
):
    if isinstance(input_str, list):
        indices2pick = torch.LongTensor(
            [len(tokenizer.encode(txt)) - 1 for txt in input_str]
        )
    else:
        indices2pick = torch.LongTensor([len(tokenizer.encode(input_str)) - 1])
    inputs = tokenizer(input_str, return_tensors="pt", padding=True).to(critic.device)
    value = critic(**inputs, return_dict=True).value.cpu()
    value = value.gather(1, indices2pick.unsqueeze_(1)).squeeze_(1).float().numpy()
    return value


@torch.inference_mode()
def seq_value_fn(critic_model, tokenizer, input_str):
    input_ids = tokenizer(input_str, return_tensors="pt").input_ids.to(
        critic_model.device
    )
    value = critic_model(input_ids, return_dict=True).value
    return value.cpu().float().numpy()


def _value_inference_fastchat(
    model_name: str,
    unused_tokenizer: AutoTokenizer,
    input_str: Union[List[str], str],
    controller_addr="http://localhost:21101"
):
    ret = requests.post(
        controller_addr + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]

    headers = {"User-Agent": "FastChat Client"}
    gen_params = {
        "input_str": input_str
    }
    response = requests.post(
        worker_addr + "/worker_value_inference",
        headers=headers,
        json=gen_params,
        stream=True,
    )
    results = response.json()
    value = results["value"]
    return value