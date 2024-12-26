import copy
import re
from typing import List, Optional
import numpy as np
from envs.base_env import CoTEnv, NoLegalActionException, INVALID_ANS
from .prompt import COT_EXAMPLES, COT_TASK_DESC, PROBLEM_FORMAT_STR, SEP
from .apps_executor import AppsExecutor
# from .verify_utils import extract_answer as extract_fn, grade_answer

ANS_RE = None
STOP_STR = None


EXECUTOR = AppsExecutor()

def extract_answer(answer_str: str) -> str:
    code = answer_str
    if 'Code:' in code:
        code = code.split('Code:')[1].strip()
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    return code


def extract_groundtruth(problem_instance: dict) -> dict:
    return problem_instance


def judge_correct(
        problem_str: str, problem_instance: dict, answer: str
) -> bool:
    with_verbal = False  # not added yet
    try:
        curr_res = EXECUTOR.check_correctness(problem_instance, answer, mode='test',
                                                   with_verbal=with_verbal)
        fixed = []
        verbal_feedbacks = []
        for e in curr_res:
            if isinstance(e, np.ndarray):
                e = e.item(0)
            if isinstance(e, np.bool_):
                e = bool(e)
            if with_verbal:
                verbal_feedbacks.append(e[1])
                e = e[0]
            fixed.append(e)

        curr_res = fixed
    except Exception as e:
        curr_res = [-1]

    # How to read results [-2] = compile error, [-1] = runtime error [False] = failed test case [True] = passed test case")
    assert isinstance(curr_res, list)
    pass_rate = np.mean(np.asarray(curr_res) > 0) if len(curr_res) > 0 else 0
    reward = pass_rate

    if reward == 1.0:
        return True
    else:
        return False


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

        self.problem = math_problems

    @property
    def stop_str(self):
        return STOP_STR

    def post_process_act(self, action: str):
        if not action.endswith(self.sep):
            action = action.strip() + self.sep

        return action

    def _is_correct(self, completion):
        extracted_answer = extract_answer(completion)
        return judge_correct(
            self.problem["question"], self.problem["answer"], extracted_answer
        )

    def get_reward(self):
        """To implement based on learned reward model"""
        return 0
