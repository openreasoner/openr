import copy
import re
from typing import List, Optional
import numpy as np
from envs.base_env import CoTEnv, NoLegalActionException, INVALID_ANS
from .prompt import COT_EXAMPLES, COT_TASK_DESC, PROBLEM_FORMAT_STR, SEP
from .verify_utils import extract_answer as extract_fn, grade_answer

ANS_RE = re.compile(r"The answer is: (.*) ки")
STOP_STR = "The answer is: "


def extract_answer(answer_str: str) -> str:
    return extract_fn(answer_str, pattern=ANS_RE)


def extract_groundtruth(groundtruth_str: str) -> str:
    return groundtruth_str


def judge_correct(
    problem_str: str, extracted_groundtruth: Optional[str], answer: str
) -> bool:
    return grade_answer(given_answer=answer, ground_truth=extracted_groundtruth)


class Env(CoTEnv):
    sep = SEP

    def __init__(
        self,
        config,
        math_problems,
        llm_gen_fn,
        task_desc_str: str = COT_TASK_DESC,
        cot_example_str: str = COT_EXAMPLES,
        problem_format_str: str = PROBLEM_FORMAT_STR,
        reset=True,
    ):
        super().__init__(
            config,
            math_problems,
            llm_gen_fn,
            task_desc_str,
            cot_example_str,
            problem_format_str,
            reset,
        )

    @property
    def stop_str(self):
        return STOP_STR

    def post_process_act(self, action: str):
        if not action.endswith(self.sep):
            if "ки" not in action:
                action = action.strip() + " ки"
        return action

    def _is_correct(self, completion):
        extracted_answer = extract_answer(completion)
        # print("Compare: {} -- {}".format(extrated_answer,
        #  self.math_problem['answer']))
        # return extrated_answer == self.math_problem['answer']
        return judge_correct(
            self.math_problem["question"], self.math_problem["answer"], extracted_answer
        )

    def get_reward(self):
        """To implement based on learned reward model"""
        return 0
