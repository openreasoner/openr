import torch
from typing import Union, List
from transformers import AutoTokenizer
import re
import numpy as np
import requests


def _value_inference_fastchat(
    model_name: str,
    # input_str: Union[List[str], str],
    question: Union[str, List[str]],
    answer: Union[str, List[str]],
    format: str,
    prm_step_tag: str = None,
    controller_addr="http://0.0.0.0:28777",
):
    ret = requests.post(
        controller_addr + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]
    if not worker_addr:
        raise ValueError("Value Model name {} does not exist.".format(model_name))

    headers = {"User-Agent": "FastChat Client"}
    gen_params = {
        "Q": question,
        "A": answer,
        "format": format,
        "prm_step_tag": prm_step_tag,
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
