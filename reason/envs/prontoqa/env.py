import re
from typing import Optional
import sympy
from distributed.utils import print_with_rank
from envs.base_env import CoTEnv, NoLegalActionException, INVALID_ANS
import numpy as np
from .prompt import COT_EXAMPLES, COT_TASK_DESC, PROBLEM_FORMAT_STR, SEP
import jsonlines

STOP_STR = "The answer is "


def extract_answer(answer_str):
    try:
        answer = answer_str.strip().split("\n")[-1].lower()
        if "true" in answer and "false" not in answer:
            return True
        elif "true" not in answer and "false" in answer:
            return False
        else:
            # print("Invalid answer: {}".format(answer))
            return INVALID_ANS
    except:
        return INVALID_ANS


def extract_groundtruth(answer):
    # answer = (
    #     answer.strip()
    #     .split("\n")[-1]
    #     .lower()
    # )
    # if "true" in answer and "false" not in answer:
    #     return True
    # elif "true" not in answer and "false" in answer:
    #     return False
    # else:
    #     raise ValueError("Invalid answer: {}".format(answer))

    ans = extract_answer(answer)
    if ans == INVALID_ANS:
        raise ValueError("Invalid answer: {}".format(answer))
    else:
        return ans


def judge_correct(problem_str: str, groundtruth: bool, answer: bool):
    return answer == groundtruth


class PrOntoQAEnv(CoTEnv):
    sep = SEP

    def __init__(
        self,
        config,
        math_problems,
        llm_gen_fn,
        tokenizer,
        task_desc_str: str = COT_TASK_DESC,
        cot_example_str: str = COT_EXAMPLES,
        problem_format_str: str = PROBLEM_FORMAT_STR,
        reset=True,
    ):
        super().__init__(
            config,
            math_problems,
            llm_gen_fn,
            tokenizer,
            task_desc_str,
            cot_example_str,
            problem_format_str,
            reset,
        )

        # although the groundtruth answer has provided in math_problems['answer'], it might be still needed
        # self.qa_map = {}
        # with jsonlines.open("all_data_path", 'r') as reader:
        #     for obj in reader:
        #         answer_str = obj["answer"][0]["text"]
        #         answer = extract_answer(answer_str)
        #         question = obj["question"]
        #         self.qa_map[question] = answer

    @property
    def stop_str(self):
        return STOP_STR

    def _is_correct(self, completion):
        answer = extract_answer(completion)
        return judge_correct(
            self.math_problem["question"], self.math_problem["answer"], answer
        )

    def get_reward(self):
        return 0.0
