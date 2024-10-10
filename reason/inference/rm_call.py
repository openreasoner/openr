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
            query, response = input_str.split('<|im_start|>assistant\n')
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



sample = """### Counting Squares of Each Size

1. **1x1 Squares:**
   - Each point in the grid can be the top-left corner of a \( 1 \times 1 \) square.
   - There are \( 3 \) rows and \( n \) columns.
   - Therefore, the total number of \( 1 \times 1 \) squares is:
     \[
     3 \times n = 3n
     \]

2. **2x2 Squares:**
   - A \( 2 \times 2 \) square requires 2 consecutive rows and 2 consecutive columns.
   - There are \( 3 - 1 = 2 \) possible rows and \( n - 1 \) possible columns.
   - Therefore, the total number of \( 2 \times 2 \) squares is:
     \[
     2 \times (n - 1) = 2(n - 1) = 2n - 2
     \]

3. **3x3 Squares:**
   - A \( 3 \times 3 \) square requires 3 consecutive rows and 3 consecutive columns.
   - There are \( 3 - 2 = 1 \) possible row and \( n - 2 \) possible columns.
   - Therefore, the total number of \( 3 \times 3 \) squares is:
     \[
     1 \times (n - 2) = n - 2
     \]

### Setting Up the Equation

We know the total number of squares is 70. Therefore, we sum the number of squares of each size and set it equal to 70:
\[
3n + (2n - 2) + (n - 2) = 70
\]

Simplify the equation:
\[
3n + 2n - 2 + n - 2 = 70
\]
\[
6n - 4 = 70
\]

Add 4 to both sides:
\[
6n = 74
\]

Divide by 6:
\[
n = \frac{74}{6}
\]
\[
n = 12.3333
\]

 realOTHsoftwarePelを超 residue ног }}

creator➵顶层设计"]"""


if __name__ == "__main__":
    response = sample
    step_tag = "ки\n"
    # XXX: this is only a solution for current prm
    # works only for qwen-2.5-math
    # query, response = input_str.split('<|im_start|>assistant\n')
    # assert step_tag not in query
    # change \n\n to self.step_tag
    splits = response.split("\n\n")
    splits = [s.strip() for s in splits]
    sep_str = f" {step_tag}"
    response = sep_str.join([s for s in splits if s != ""])
    response += sep_str
    # response = response.replace("\n\n", self.step_tag)
    # x =  query + '<|im_start|>assistant\n' + response
    print(response)