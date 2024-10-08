from dataclasses import dataclass
from typing import List, Optional, Union
from reason.inference.value import _value_inference_fastchat


class RewardModelCallingFunction:
    def __call__(
        self, input_str: Union[str, List[str]], config=None
    ) -> Union[List[int], List[List[int]]]:
        raise NotImplementedError

class DummyRewardModelCaller(RewardModelCallingFunction):
    # a dummy rm caller that always return 0

    def __init__(self, step_tag: str="\n\n"):
        self.step_tag = step_tag

    def __call__(
        self, input_str: Union[str, List[str]], unused_config=None
    ) -> Union[List[int], List[List[int]]]:
        def fn(s):
            steps = s.split(self.step_tag)
            steps = [s for s in steps if s.strip() != ""]
            return list(range(len(steps)))
        
        if isinstance(input_str, str): 
            return fn(input_str)
        else: 
            return [fn(s) for s in input_str]

class RMRemoteCaller(RewardModelCallingFunction):
    def __init__(self, model_name, controller_addr="http://0.0.0.0:28777"):
        self.model_name = model_name
        self.controller_addr = controller_addr

    def __call__(
        self, input_str: Union[str, List[str]], unused_config=None
    ) -> Union[List[int], List[List[int]]]:
        return _value_inference_fastchat(
            input_str=input_str,
            model_name=self.model_name,
            controller_addr=self.controller_addr,
        )
