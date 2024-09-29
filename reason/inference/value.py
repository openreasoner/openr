import torch
from typing import Union, List
from transformers import AutoTokenizer
import re
import numpy as np
import requests


def _value_inference_fastchat(
    model_name: str,
    input_str: Union[List[str], str],
    controller_addr="http://0.0.0.0:28777",
):
    ret = requests.post(
        controller_addr + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]

    headers = {"User-Agent": "FastChat Client"}
    gen_params = {"input_str": input_str}
    response = requests.post(
        worker_addr + "/worker_value_inference",
        headers=headers,
        json=gen_params,
        stream=True,
    )
    results = response.json()
    value = results["value"]
    return value
