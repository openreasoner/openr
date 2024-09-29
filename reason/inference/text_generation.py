from typing import List, Optional
import requests
from dataclasses import dataclass


@dataclass
class ConcatedLMGenResult:
    text: List[str]
    prompt_tokens: List[int]
    completion_tokens: List[int]
    cumulative_logprob: List[float]
    logp_avg_by_len: List[float]


def _generate_fastchat(
    query_str,
    model_name,
    n,
    temperature,
    top_p,
    top_k,
    max_new_tokens,
    stop_token_ids,
    stop_str,
    controller_addr,
) -> ConcatedLMGenResult:

    ret = requests.post(
        controller_addr + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]

    headers = {"User-Agent": "FastChat Client"}
    gen_params = {
        "model": model_name,
        "prompt": query_str,
        "temperature": temperature,
        "n": n,
        "top_p": top_p,
        "top_k": top_k,
        "stop_token_ids": stop_token_ids,
        "max_new_tokens": max_new_tokens,
        "stop": stop_str,
        "echo": False,
    }
    response = requests.post(
        worker_addr + "/worker_generate",
        headers=headers,
        json=gen_params,
        stream=True,
    )
    results = response.json()
    output_token_lens = results["output_token_len"]
    cum_logps = results["cumulative_logprob"]
    avg_len_logps = [clp / otl for clp, otl in zip(cum_logps, output_token_lens)]
    # return results["text"], avg_len_logps
    return ConcatedLMGenResult(
        text=results["text"],
        prompt_tokens=results["usage"]["prompt_tokens"],
        completion_tokens=results["usage"]["completion_tokens"],
        cumulative_logprob=cum_logps,
        logp_avg_by_len=avg_len_logps,
    )
