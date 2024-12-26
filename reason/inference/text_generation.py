from typing import List, Optional
import requests
from dataclasses import dataclass


@dataclass
class ConcatedLMGenResult:
    text: List[str]
    prompt_tokens: int  # List[int]
    num_tokens: List[int]
    cumulative_logprob: List[float]
    logp_avg_by_len: List[float]
    finish_reason: List[str]

    # post init compute number of completion_tokens
    def __post_init__(self):
        self.completion_tokens = sum(self.num_tokens)

    @classmethod
    def merge(cls, results: List["ConcatedLMGenResult"]) -> "ConcatedLMGenResult":
        merged_text = []
        merged_prompt_tokens = 0
        merged_num_tokens = []
        merged_cumulative_logprob = []
        merged_logp_avg_by_len = []
        merged_finish_reason = []
        for result in results:
            merged_text.extend(result.text)
            merged_prompt_tokens += result.prompt_tokens
            merged_num_tokens.extend(result.num_tokens)
            merged_cumulative_logprob.extend(result.cumulative_logprob)
            merged_logp_avg_by_len.extend(result.logp_avg_by_len)
            merged_finish_reason.extend(result.finish_reason)
        return cls(
            text=merged_text,
            prompt_tokens=merged_prompt_tokens,
            num_tokens=merged_num_tokens,
            cumulative_logprob=merged_cumulative_logprob,
            logp_avg_by_len=merged_logp_avg_by_len,
            finish_reason=merged_finish_reason,
        )


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
    include_stop_str_in_output,
    controller_addr,
    use_lora,
) -> ConcatedLMGenResult:

    ret = requests.post(
        controller_addr + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]
    if not worker_addr:
        raise ValueError("Language Model name {} does not exist.".format(model_name))

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
        "include_stop_str_in_output": include_stop_str_in_output,
        "use_lora": use_lora,
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
    avg_len_logps = [
        clp / max(1, otl) for clp, otl in zip(cum_logps, output_token_lens)
    ]
    # return results["text"], avg_len_logps
    return ConcatedLMGenResult(
        text=results["text"],
        prompt_tokens=results["usage"]["prompt_tokens"],
        num_tokens=results["output_token_len"],
        cumulative_logprob=cum_logps,
        logp_avg_by_len=avg_len_logps,
        finish_reason=results["finish_reason"],
    )
