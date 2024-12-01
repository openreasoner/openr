from dataclasses import dataclass
from datetime import datetime
import importlib
from multiprocessing import Pool
from typing import Any, Callable, Dict, Optional, List, Union

import numpy as np
import ray
import re
from envs import get_default_query_str_builder, get_env_datasets
from reason.inference.lm_call import LanguageModelCallingFunction, LMCallingConfig
from reason.inference.rm_call import RewardModelCallingFunction
from reason.reranking.vote_utils import (
    MAJORITY_VOTE,
    PRM_MIN_MAX,
    PRM_MIN_VOTE,
    PRM_LAST_VOTE,
    PRM_LAST_MAX,
    AGG_FN_MAP,
)
from envs.base_env import INVALID_ANS


class Task:
    def __init__(
        self,
        task_name: str,
        is_few_shot: bool = False,
        cot_task_desc: str = None,
        cot_examples: str = None,
        problem_format_str: str = None,
    ):
        self.task_name = task_name
        task_module = importlib.import_module(f"envs.{task_name}")
        if task_name == "MATH" or "rstar":
            self.extract_answer = task_module.extract_answer
            self.extract_groundtruth = task_module.extract_groundtruth
            self.judge_correct = task_module.judge_correct
        else:
            raise NotImplementedError(f"Task {task_name} is not supported")

        self._is_few_shot = is_few_shot
        self.env_fn = task_module.Env

        self.cot_task_desc = cot_task_desc
        self.cot_examples = cot_examples
        self.problem_format_str = problem_format_str

    def prompt_fn(self, problem_input: str):
        return get_default_query_str_builder(
            self.task_name,
            cot_task_desc=self.cot_task_desc,
            cot_examples=self.cot_examples,
            problem_format_str=self.problem_format_str,
        )(problem_input, is_few_shot=self._is_few_shot)

    @property
    def test_ds(self):
        return get_env_datasets(self.task_name)[1]


CHOSEN_AGGR_METHODS = [
    MAJORITY_VOTE,
    PRM_MIN_MAX,
    PRM_MIN_VOTE,
    PRM_LAST_MAX,
    PRM_LAST_VOTE,
]


def judge_ans(
    problem_str: str,
    extracted_groundtruth: str,
    output_list: List[str],
    v_list: List[float],
    aggration_mode: str,
    extract_answer_fn,
    judge_correct_fn,
    normalize=False,
):
    ans_list = [extract_answer_fn(txt) for txt in output_list]
    valid_ans_list, valid_v_list = [], []
    for i, ans in enumerate(ans_list):
        if ans != INVALID_ANS:
            valid_ans_list.append(ans)
            valid_v_list.append(v_list[i])
    if len(valid_ans_list) == 0:
        return 0

    if "orm" in aggration_mode and normalize:
        # score_normalization: this is only necessary for [-1, 1] values
        valid_v_list = np.array(valid_v_list)
        valid_v_list -= valid_v_list.min()
        valid_v_list /= valid_v_list.max() + 1e-3
        valid_v_list = valid_v_list.tolist()
    aggregated_ans = AGG_FN_MAP[aggration_mode](valid_ans_list, valid_v_list)

    return (
        1 if judge_correct_fn(problem_str, extracted_groundtruth, aggregated_ans) else 0
    )


@dataclass
class SolutionOutput:
    solutions: List[str]
    # Define the completion tokens for each solution
    #  For best_of_n, it's a list of int, indicate how many tokens in each
    #      generation
    #  for beam search, it's a list of zeros, except the last element indicates total tokens
    #  for mcts, it's a list of int, indicate how many tokens comsumed between two paths
    completion_tokens: List[int]


@dataclass
class TreeSearchSolutionOutput(SolutionOutput):
    tree_completion_tokens: List[int]


class MathEvaluator:

    def __init__(
        self,
        task: Union[str, Task],
        lm_call: LanguageModelCallingFunction,
        rm_call: RewardModelCallingFunction,
    ):
        if isinstance(task, str):
            self._task = Task(task_name=task)
        else:
            assert isinstance(task, Task)
            self._task = task
        self.lm_call = lm_call
        self.rm_call = rm_call

    def evaluate_problem(
        self, problem_inst: Dict[str, str], solver_fn: Callable
    ) -> List[str]:
        solution: SolutionOutput = solver_fn(problem_inst, self.lm_call, self.rm_call)
        result, output = self.analyze_output(problem_inst, solution.solutions)
        total_completion_token = 0
        for i, o in enumerate(output):
            o["completion_tokens"] = solution.completion_tokens[i]
            if isinstance(solution, TreeSearchSolutionOutput):
                o["tree_completion_tokens"] = solution.tree_completion_tokens[i]
            # We define the completion_tokens as the tokens comsumed between two generated
            #  answers, therefore we need to take sum here.
            total_completion_token += solution.completion_tokens[i]
        result["total_completion_tokens"] = total_completion_token
        return problem_inst, result, output

    def analyze_output(self, problem_inst: Dict[str, str], gen_answers: List[str]):
        extracted_groundtruth = self._task.extract_groundtruth(problem_inst["answer"])

        if len(gen_answers) > 1:
            input_list = [(problem_inst["question"], txt) for txt in gen_answers]
            # XXX(ziyu): for tree search methods with value_fn, should not call rm
            #  to compute it again
            value_list = self.rm_call(input_list, lm_step_tag=self.lm_call.lm_step_tag)
        else:
            value_list = [[0]]
        output_list = [
            {"path_idx": i, "text": txt, "value": v}
            for i, (txt, v) in enumerate(zip(gen_answers, value_list))
        ]
        res = {
            agg_method: judge_ans(
                problem_inst["question"],
                extracted_groundtruth,
                gen_answers,
                value_list,
                agg_method,
                self._task.extract_answer,
                self._task.judge_correct,
            )
            for agg_method in (
                CHOSEN_AGGR_METHODS if len(gen_answers) > 1 else [MAJORITY_VOTE]
            )
        }
        return res, output_list


class LLMEvaluator:
    """
    LLM-as-Judge
    """

    def __init__(
        self, lm_call: LanguageModelCallingFunction, gen_config: LMCallingConfig
    ):
        self.lm_call = lm_call
        self.judge_sys_prompt = (
            "<|im_start|>system\nYour are given a math problem, a ground truth answer and a solution generated by an AI assistant. "
            "Please first extract the final answer from the solution, and then compare this solution with the ground truth answer "
            "for consistency. Please ignore any formatting differences. Requirement: If the assistant solution is correct, "
            "your output should be of format *<TAG>1</TAG>*, otherwise the output should be *<TAG>0</TAG>*.<|im_end|>"
        )  # Qwen style, feel free to change it

        self.judge_format_str = """
            <|im_start|>user\nHerer is a math problem: {question}\nThe ground truth answer is: [{ground_truth}]\n
            The solution of an AI assistant is [{gen_answer}]\n<im_end>\n|<im_start>|assistant\n"""

        self.gen_config = gen_config

    def evaluate_problem(self, problem_inst: Dict[str, str]):
        gen_config = LMCallingConfig(
            n=1,
            temperature=0,
            top_k=1,
            top_p=1.0,
            max_new_tokens=self.gen_config.max_new_tokens,
            stop_str=self.gen_config.stop_str,
        )
        if gen_config.max_new_tokens < 256:
            print("Warning: max new tokens is less than 256!")

        prompt = (
            self.judge_sys_prompt
            + "\n"
            + self.judge_format_str.format(
                question=problem_inst["question"],
                ground_truth=problem_inst["groundtruth"],
                gen_answer=problem_inst["gen_answer"],
            )
        )  # feel free to change the key here

        output = self.lm_call(prompt, gen_config)
        try:
            score_tag = self.process_output(output.text[0])
        except Exception as e:
            print(f"Error {e}\n Output = {output.text}")
            score_tag = 0

        return problem_inst, score_tag, output.text

    def process_output(self, review):
        try:
            tag = float(review)
            if tag in [0, 1]:
                return tag
        except:
            pass

        try:
            tag = review.lower().split("tag")[-2]
            pattern = r"[01]"
            matches = re.findall(pattern, tag)[0]
            tag = 1 if int(matches) == 1 else 0
        except:
            pattern = r"[01]"
            matches = re.findall(pattern, tag)[0]
            tag = 1 if int(matches) == 1 else 0

        return tag


@ray.remote
class RemoteMathEvaluator(MathEvaluator):
    def __init__(
        self,
        task: str,
        lm_call: LanguageModelCallingFunction,
        rm_call: RewardModelCallingFunction,
    ):
        super().__init__(task, lm_call, rm_call)


@ray.remote
class RemoteLLMEvaluator(LLMEvaluator):
    def __init__(
        self,
        lm_call: LanguageModelCallingFunction,
        gen_config: LMCallingConfig,
    ):
        super().__init__(lm_call, gen_config)
