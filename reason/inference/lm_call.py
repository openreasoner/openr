from dataclasses import dataclass
from typing import List, Optional
from reason.inference.text_generation import ConcatedLMGenResult, _generate_fastchat


@dataclass
class LMCallingConfig:
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1 # -1 for vllm by default
    max_new_tokens: int = 512
    stop_token_ids: Optional[List[int]] = None
    stop_str: Optional[str] = None
    include_stop_str_in_output: bool = False


class LanguageModelCallingFunction:
    def __init__(self, lm_step_tag: str=None):
        self.lm_step_tag = lm_step_tag

    def __call__(self, input_str: str, config: LMCallingConfig) -> ConcatedLMGenResult:
        raise NotImplementedError

class VLLMRemoteCaller(LanguageModelCallingFunction):
    def __init__(self, model_name, controller_addr="http://0.0.0.0:28777", lm_step_tag: str=None):
        self.model_name = model_name
        self.controller_addr = controller_addr
        super().__init__(lm_step_tag)

    def __call__(self, input_str: str, config: LMCallingConfig) -> ConcatedLMGenResult:
        return _generate_fastchat(
            query_str=input_str,
            model_name=self.model_name,
            n=config.n,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            max_new_tokens=config.max_new_tokens,
            stop_token_ids=config.stop_token_ids,
            stop_str=config.stop_str,
            controller_addr=self.controller_addr,
            include_stop_str_in_output=config.include_stop_str_in_output,
        )

class FastChatRemoteCaller(LanguageModelCallingFunction):
    def __init__(self, model_name, controller_addr="http://0.0.0.0:28777", lm_step_tag: str=None):
        self.model_name = model_name
        self.controller_addr = controller_addr
        super().__init__(lm_step_tag)

    def __call__(self, input_str: str, config: LMCallingConfig) -> ConcatedLMGenResult:
        # XXX(ziyu): Low-efficiency implementation, can not accept to much calls
        text = [] 
        prompt_token = []
        num_tokens = []
        cumulative_logprob = []
        logp_avg_by_len = []
        finish_reason = []

        for i in range(config.n):
            res =   _generate_fastchat(
                    query_str=input_str,
                    model_name=self.model_name,
                    n=1, # this is not used
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    max_new_tokens=config.max_new_tokens,
                    stop_token_ids=config.stop_token_ids,
                    stop_str=config.stop_str,
                    controller_addr=self.controller_addr,
                    include_stop_str_in_output=config.include_stop_str_in_output,
                )
            text.append(res.text[0])
            cumulative_logprob.append(res.cumulative_logprob[0])
            logp_avg_by_len.append(res.logp_avg_by_len[0])
            prompt_token.append(res.prompt_tokens[0])
            num_tokens.append(res.num_tokens[0])
            finish_reason.append(res.finish_reason[0])
        return ConcatedLMGenResult(
            text=text,
            prompt_tokens=prompt_token,
            num_tokens=num_tokens,
            cumulative_logprob=cumulative_logprob,
            logp_avg_by_len=logp_avg_by_len,
            finish_reason=finish_reason
        )