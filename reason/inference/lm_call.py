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
            include_stop_str_in_output=config.include_stop_str_in_output,
        )

class FastChatRemoteCaller(LanguageModelCallingFunction):
    def __init__(self, model_name, controller_addr="http://0.0.0.0:28777"):
        self.model_name = model_name
        self.controller_addr = controller_addr

    def __call__(self, input_str: str, config: LMCallingConfig) -> ConcatedLMGenResult:
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

if __name__ == "__main__":
    format_str = '<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n'
    q = "Evaluate $(1+2i)6-3i$."
    # text = format_str.format(question=q)
    text='<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\nSimplify $\\tan 100^\\circ + 4 \\sin 100^\\circ.$<|im_end|>\n<|im_start|>assistant\nTo simplify the expression \\(\\tan 100^\\circ + 4 \\sin 100^\\circ\\), we start by using the identity \\(\\tan 100^\\circ = \\tan (180^\\circ - 80^\\circ) = -\\tan 80^\\circ\\). Therefore, the expression becomes:\n\n\\[\n\\tan 100^\\circ + 4 \\sin 100^\\circ = -\\tan 80^\\circ + 4 \\sin 100^\\circ\n\\]\n\nNext, we use the identity \\(\\sin 100^\\circ = \\sin (180^\\circ - 80^\\circ) = \\sin 80^\\circ\\). So the expression further simplifies to:\n\n\\[\n-\\tan 80^\\circ + 4 \\sin 80^\\circ\n\\]\n\nWe can express \\(\\tan 80^\\circ\\) as \\(\\frac{\\sin 80^\\circ}{\\cos 80^\\circ}\\). Substituting this into the expression, we get:\n\n\\[\n-\\frac{\\sin 80^\\circ}{\\cos 80^\\circ} + 4 \\sin 80^\\circ\n\\]\n\nTo combine these terms, we need a common denominator. The common denominator is \\(\\cos 80^\\circ\\), so we rewrite the expression as:\n\n\\[\n-\\frac{\\sin 80^\\circ}{\\cos 80^\\circ} + \\frac{4 \\sin 80^\\circ \\cos 80^\\circ}{\\cos 80^\\circ} = \\frac{-\\sin 80^\\circ + 4 \\sin 80^\\circ \\cos 80^\\circ}{\\cos 80^\\circ}\n\\]\n\nWe can factor out \\(\\sin 80^\\circ\\) from the numerator:\n\n\\[\n\\frac{\\sin 80^\\circ (-1 + 4 \\cos 80^\\circ)}{\\cos 80^\\circ}\n\\]\n\nThis simplifies to:\n\n\\[\n\\sin 80^\\circ \\cdot \\frac{-1 + 4 \\cos 80^\\circ}{\\cos 80^\\circ} = \\sin 80^\\circ \\left( \\frac{-1}{\\cos 80^\\circ} + 4 \\right) = \\sin 80^\\circ \\left( -\\sec 80^\\circ + 4 \\right)\n\\]\n\nWe know from trigonometric identities that \\(\\sec 80^\\circ = \\frac{1}{\\cos 80^\\circ} = \\csc 10^\\circ\\) (since \\(\\sec (90^\\circ - \\theta) = \\csc \\theta\\)). Therefore, \\(-\\sec 80^\\circ = -\\csc 10^\\circ\\). So the expression becomes:\n\n\\[\n\\sin 80^\\circ \\left( -\\csc 10^\\circ + 4 \\right) = \\sin 80^\\circ \\left( -\\frac{1}{\\sin 10^\\circ} + 4 \\right)\n\\]\n\nWe can rewrite \\(\\sin 80^\\circ\\) as \\(\\cos 10^\\circ\\) (since \\(\\sin (90^\\circ - \\theta) = \\cos \\theta\\)). So the expression is:\n\n\\[\n\\cos 10^\\circ \\left( -\\frac{1}{\\sin 10^\\circ} + 4 \\right) = \\cos 10^\\circ \\left( -\\frac{1}{\\sin 10^\\circ} \\right) + \\cos 10^\\circ \\cdot 4 = -\\frac{\\cos 10^\\circ}{\\sin 10^\\circ} + 4 \\cos 10^\\circ = -\\cot 10^\\circ + 4 \\cos 10^\\circ\n\\]\n\nWe know from trigonometric identities that \\(\\cot 10^\\circ = \\frac{\\cos 10^\\circ}{\\sin 10^\\circ}\\). So we have:\n\n\\[\n-\\cot 10^\\circ + 4 \\cos 10^\\circ = -\\frac{\\cos 10^\\circ}{\\sin 10^\\circ} + 4 \\cos 10^\\circ = -\\frac{\\cos 10^\\circ}{\\sin 10^\\circ} + \\frac{4 \\cos 10^\\circ \\sin 10^\\circ}{\\sin 10^\\circ} = \\frac{-\\cos 10^\\circ + 4 \\cos 10^\\circ \\sin 10^\\circ}{\\sin 10^\\circ}\n\\]\n\nWe can factor out \\(\\cos 10^\\circ\\) from the numerator:\n\n'
    lm_call = FastChatRemoteCaller("Qwen2.5-Math-7B-Instruct")
    ret = lm_call(text, config=LMCallingConfig(
        n=4, 
        stop_str="\n\n", 
        temperature=0.7,
        include_stop_str_in_output=True))

    print(ret)