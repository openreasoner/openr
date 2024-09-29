from dataclasses import dataclass
from reason.inference.lm_call import LMCallingConfig, VLLMRemoteCaller
from reason.inference.rm_call import RMRemoteCaller
from reason.evaluation.evaluator import SolutionOutput

@dataclass
class CoTConfig:
    pass
def cot(config: CoTConfig, gen_config: LMCallingConfig, prompt, llm_call: VLLMRemoteCaller, rm_call: RMRemoteCaller) -> SolutionOutput:
    gen_config = LMCallingConfig(
        n=1,
        temperature=0,
        top_k=None,
        top_p=1.,
        max_new_tokens=gen_config.max_new_tokens,
    )
    return best_of_n(prompt, llm_call, rm_call)

@dataclass
class BestOfNConfig:
    num_sequence: int = 32
def best_of_n(config: BestOfNConfig, gen_config: LMCallingConfig, prompt, lm_call: VLLMRemoteCaller, rm_call: RMRemoteCaller) -> SolutionOutput:
    if gen_config.max_new_tokens < 256:
        print("Warning: max_new_tokens is less than 256")

    gen_config.n = config.num_sequence
    
    output = lm_call(prompt, gen_config)
    completion_tokens = [0] * len(output.text)
    completion_tokens[-1] = output.completion_tokens
    return SolutionOutput(
        solutions=output.text,
        completion_tokens=completion_tokens,
    )
