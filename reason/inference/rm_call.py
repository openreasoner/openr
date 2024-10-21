from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from reason.inference.value import _value_inference_fastchat


@dataclass
class RewardModelBaseConfig:
    step_tag: str
    # a format string that takes in question and answer
    #  need to have {question} and {answer} in the string
    format_str: str


class RewardModelCallingFunction:
    def __init__(self, config: RewardModelBaseConfig):
        self.config = config
        self.step_tag = config.step_tag
        self.format_str = config.format_str

    def __call__(
        self,
        question_answer_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
        lm_step_tag: str,
    ) -> Union[List[int], List[List[int]]]:
        raise NotImplementedError

    def replace_step_tag(self, answer: str, lm_step_tag: str):
        splits = answer.split(lm_step_tag)
        splits = [s.strip() for s in splits]
        # add a whitespace to avoid tokenization issue
        response = f" {self.step_tag}".join([s for s in splits if s != ""])
        response += f" {self.step_tag}"
        return response


class DummyRewardModelCaller(RewardModelCallingFunction):
    # a dummy rm caller that always return 0

    def __init__(self, config: RewardModelBaseConfig):
        super().__init__(config)

    def __call__(
        self,
        question_answer_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
        lm_step_tag: str,
    ) -> Union[List[int], List[List[int]]]:

        def fn(s):
            steps = s.split(self.step_tag)
            steps = [s for s in steps if s.strip() != ""]
            return list(range(len(steps)))

        if isinstance(question_answer_pairs[0], str):
            return fn(
                self.format_str.format(
                    question=question_answer_pairs[0],
                    answer=self.replace_step_tag(question_answer_pairs[1], lm_step_tag),
                )
            )
        else:
            return [
                fn(
                    self.format_str.format(
                        question=s[0],
                        answer=self.replace_step_tag(s[1], lm_step_tag),
                    )
                )
                for s in question_answer_pairs
            ]


@dataclass
class RemoteRewardModelConfig(RewardModelBaseConfig):
    model_name: str
    controller_addr: str


class RMRemoteCaller(RewardModelCallingFunction):
    def __init__(self, config: RemoteRewardModelConfig):
        self.model_name = config.model_name
        self.controller_addr = config.controller_addr
        super().__init__(config)

    def __call__(
        self,
        question_answer_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
        lm_step_tag: str,
    ) -> Union[List[int], List[List[int]]]:

        if isinstance(question_answer_pairs[0], str):
            response = self.replace_step_tag(question_answer_pairs[1], lm_step_tag)
            input_str = self.format_str.format(
                question=question_answer_pairs[0], answer=response
            )
        else:
            input_str = [
                self.format_str.format(
                    question=s[0],
                    answer=self.replace_step_tag(s[1], lm_step_tag),
                )
                for s in question_answer_pairs
            ]
        return _value_inference_fastchat(
            input_str=input_str,
            model_name=self.model_name,
            controller_addr=self.controller_addr,
        )
