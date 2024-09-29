from dataclasses import dataclass
from typing import List, Optional, Union
from reason.inference.value import _value_inference_fastchat


class RewardModelCallingFunction:
    def __call__(
        self, input_str: str, config=None
    ) -> Union[List[int], List[List[int]]]:
        raise NotImplementedError


class RMRemoteCaller(RewardModelCallingFunction):
    def __init__(self, model_name, controller_addr="http://0.0.0.0:28777"):
        self.model_name = model_name
        self.controller_addr = controller_addr

    def __call__(
        self, input_str: str, unused_config=None
    ) -> Union[List[int], List[List[int]]]:
        return _value_inference_fastchat(
            input_str=input_str,
            model_name=self.model_name,
            controller_addr=self.controller_addr,
        )
