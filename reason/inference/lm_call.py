from dataclasses import dataclass
from typing import List, Optional
from reason.inference.text_generation import ConcatedLMGenResult, _generate_fastchat


@dataclass
class LMCallingConfig:
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    max_new_tokens: int = 512
    stop_token_ids: Optional[List[int]] = None
    stop_str: Optional[str] = None


class LanguageModelCallingFunction:
    def __call__(self, input_str: str, config: LMCallingConfig) -> ConcatedLMGenResult:
        raise NotImplementedError


class VLLMRemoteCaller(LanguageModelCallingFunction):
    def __init__(self, model_name, controller_addr="http://0.0.0.0:28777"):
        self.model_name = model_name
        self.controller_addr = controller_addr

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
        )
