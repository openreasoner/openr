import re
from typing import Optional
import sympy
from tsllm.distributed.utils import print_with_rank
from tsllm.envs.base_env import CoTEnv, NoLegalActionException, INVALID_ANS
import numpy as np
from .prompt import COT_EXAMPLES, COT_TASK_DESC, PROBLEM_FORMAT_STR, SEP

STOP_STR = "The answer is "


def extract_answer(answer_str):
    try:
        expressions = (
            answer_str.strip()
            .split("\n")[-1]
            .lower()
            .replace(STOP_STR.lower(), "")
            .split("=")
        )
        lhs = expressions[0].strip()
        rhs = expressions[1].strip()
        # if int(rhs) != 24:
        #     return INVALID_ANS
    except:
        return INVALID_ANS
    return lhs


def extract_groundtruth(x: str):
    return None


def judge_correct(
    problem_str: str, extracted_groundtruth: Optional[str], extracted_answer
):
    numbers = re.findall(r"\d+", extracted_answer)
    problem_numbers = re.findall(r"\d+", problem_str)
    if sorted(numbers) != sorted(problem_numbers):
        return False
    try:
        return sympy.simplify(extracted_answer) == 24
    except Exception as e:
        return False


class Game24Env(CoTEnv):
    """
    Pure Question input is: 4 numbers seperated by whitespace
    e.g. 2 3 5 12
    """

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
        if "max_length" in config and config["max_length"] != 4:
            print_with_rank("In game24 max_length should be 4, force setting it to 4.")
        config["max_length"] = 4
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

    @property
    def stop_str(self):
        return STOP_STR

    def _is_correct(self, completion):
        extracted_answer = extract_answer(completion)
        return judge_correct(
            self.math_problem["question"], self.math_problem["answer"], extracted_answer
        )

    def get_reward(self):
        return 0.0
