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
        def split_query_response_hack(input_str):
            assert self.step_tag != "\n\n"
            # XXX: this is only a solution for current prm
            # works only for qwen-2.5-math
            query, response = input_str.split('<|im_start|>assistant\n')
            assert self.step_tag not in query
            # change \n\n to self.step_tag
            splits = response.split("\n\n")
            splits = [s.strip() for s in splits]
            response = f" {self.step_tag}".join([s for s in splits if s != ""])
            response += f" {self.step_tag}"
            # response = response.replace("\n\n", self.step_tag)
            return query + '<|im_start|>assistant\n' + response
        
        def fn(s):
            s = split_query_response_hack(s)
            steps = s.split(self.step_tag)
            steps = [s for s in steps if s.strip() != ""]
            return list(range(len(steps)))
        
        if isinstance(input_str, str): 
            return fn(input_str)
        else: 
            return [fn(s) for s in input_str]

class RMRemoteCaller(RewardModelCallingFunction):
    def __init__(self, model_name, controller_addr="http://0.0.0.0:28777", step_tag="\n\n"):
        self.model_name = model_name
        self.controller_addr = controller_addr
        self.step_tag = step_tag

    def __call__(
        self, input_str: Union[str, List[str]], unused_config=None
    ) -> Union[List[int], List[List[int]]]:
        def split_query_response_hack(input_str):
            assert self.step_tag != "\n\n"
            # XXX: this is only a solution for current prm
            # works only for qwen-2.5-math
            # split question from '<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n'
            query, response = input_str.split('<|im_end|>\n<|im_start|>assistant\n')
            _, query = query.split('<|im_start|>user\n')
            assert self.step_tag not in query
            # change \n\n to self.step_tag
            splits = response.split("\n\n")
            splits = [s.strip() for s in splits]
            sep_str = f" {self.step_tag}"
            response = sep_str.join([s for s in splits if s != ""])
            response += sep_str
            # response = response.replace("\n\n", self.step_tag)
            return f"{query} {response}"
        
        if isinstance(input_str, str):
            input_str = split_query_response_hack(input_str)
        else:
            input_str = [split_query_response_hack(s) for s in input_str]
        
        return _value_inference_fastchat(
            input_str=input_str,
            model_name=self.model_name,
            controller_addr=self.controller_addr,
        )
