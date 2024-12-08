from typing import List, Dict, Tuple
import os
import numpy as np
import json

from envs.MATH.env import CoTEnv
from envs.base_env import NoLegalActionException, ResetException
from tqdm import tqdm
from envs.MATH.env import extract_answer, extract_groundtruth, judge_correct
from reason.inference.lm_call import LMCallingConfig
from .prompt import (
    COT_TASK_DESC,
    CRITIQUE_TASK_DESC,
    REWRITE_TASK_DESC,
    COT_FORMAT_STR,
    CRITIQUE_FORMAT_STR,
    REWRITE_FORMAT_STR,
    SEP,
)
from distributed.utils import print_with_rank
from loguru import logger

from pathlib import Path

# Get the file path of the current script
CURRENT_DIR = Path(__file__).parent


ANS_RE = None
STOP_STR = None


def read_txt(file_path):
    assert str(file_path).endswith(".txt")
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    return data


def read_json(file_path):
    assert str(file_path).endswith(".json")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


class Env(CoTEnv):
    def __init__(
        self,
        config,
        math_problems,
        llm_gen_fn,
        task_desc_str: str = None,
        cot_example_str: str = None,
        problem_format_str: str = None,
        cot_task_desc_str: str = COT_TASK_DESC,
        critique_task_desc_str: str = CRITIQUE_TASK_DESC,
        rewrite_task_desc_str: str = REWRITE_TASK_DESC,
        cot_format_str: str = COT_FORMAT_STR,
        critique_format_str: str = CRITIQUE_FORMAT_STR,
        rewrite_format_str: str = REWRITE_FORMAT_STR,
        reset=False,
    ):
        """
        three types of thinking in reasoning:
        1. i.i.d sampling: generative independent response
        2. conditional sampling (counterfactual): what if the response is wrong?
        3. reflective sampling
        Args:
            config:
            math_problems:
            llm_gen_fn:
            task_desc_str:
            cot_example_str:
            problem_format_str:
            reset:
        """
        super().__init__(
            config,
            math_problems,
            llm_gen_fn,
            task_desc_str,
            cot_example_str,
            problem_format_str,
            reset=False,
        )

        self.current_node_type = None
        # LLM generation config
        self.gen_cfg = config["generation_config"]

        # default parameter setting
        self.num_first_answer = 3
        self.num_reviews = 3
        self.num_rewrite = 3
        self.max_new_tokens_answer = 1024
        self.max_new_tokens_review = 1024
        self.max_depth = 5

        self.print_log = True
        self.total_api_call_completion = 0
        self.total_tree_completion = 0

        # self.task_name = "MATH"
        self.sep = None
        self._init_query = None
        self._next_state_terminated = None

        # loading template
        # with open(os.path.join(
        #         CURRENT_DIR, f"prompts/MATH/decompose/decompose_template.json"),
        #         "r") as f:
        #     decompose_template = json.load(f)
        #     self.question_index = decompose_template["index"]
        # self.critic_prompt_template = read_json(
        #     os.path.join(CURRENT_DIR, f"prompts/MATH/generation_config.json"))
        self.cot_task_desc = cot_task_desc_str
        self.critique_task_desc = critique_task_desc_str
        self.rewrite_task_desc = rewrite_task_desc_str

        self.cot_format_str = cot_format_str
        self.critique_format_str = critique_format_str
        self.rewrite_format_str = rewrite_format_str

        if reset:
            self.reset(update_legal_action=True)
        # self.rewrite_prompt_template = None

    def check_attribute(self):
        assert hasattr(self, "cot_task_desc")
        assert hasattr(self, "critique_task_desc")
        assert hasattr(self, "rewrite_task_desc")
        assert hasattr(self, "cot_format_str")
        assert hasattr(self, "critique_format_str")
        assert hasattr(self, "rewrite_format_str")

    @property
    def stop_str(self):
        return STOP_STR

    @property
    def answer(self):
        return ""

    @property
    def full_answer(self):
        return self.action_history[-1]

    def post_process_act(self, action: str):
        if not action.endswith(self.sep):
            action = action.strip() + self.sep

        return action

    def set_problem(self, idx):
        self.math_problem = self.math_problems[idx]

    def get_state(self):
        # if no solution has been generated yet, generate the initial query
        if len(self.action_history) == 0:
            ret = self.cot_task_desc + self.cot_format_str.format(
                question=self.question
            )
        else:
            ret = (
                self.cot_task_desc
                + self.cot_format_str.format(question=self.question)
                + self.action_history[-1]
            )
        return ret

    def reset(self, update_legal_action=True):
        """
        reset the environment, and generate the first solution to the question
        Args:
            update_legal_action:

        Returns:

        """
        assert update_legal_action, print("Need to first update legal action")
        self.set_problem(idx=0)
        self.action_history = []
        self.review_history = []

        if update_legal_action:
            cnt = 0
            while cnt < 3:
                cnt += 1
                try:
                    self._legal_actions, api_completion_token = (
                        self.generate_first_response()
                    )
                    break
                except NoLegalActionException as e:
                    if cnt == 3:
                        raise ResetException
        info = {"api_completion_token": api_completion_token}
        return None, info

    def generate_first_response(self) -> (List[Dict], int):
        first_cot_prompt = self.cot_task_desc + self.cot_format_str.format(
            question=self.math_problem["question"]
        )
        result = self.llm_gen_fn(
            input_str=first_cot_prompt,
            config=LMCallingConfig(
                n=self.num_first_answer,
                stop_str=self.sep,
                include_stop_str_in_output=True,
                max_new_tokens=self.max_new_tokens_answer,
                **self.config["generation_config"]
            ),
        )

        texts = result.text
        logps_avg_by_len = result.logp_avg_by_len
        token_len = result.num_tokens

        # for i in texts:
        #     print(f"\n {i}")

        _legal_actions = [
            {
                "action": action,
                "prob": prob,
                "num_token": n_token,
                "finish_reason": finish_reason,
                "from_review": "",
            }
            for action, prob, n_token, finish_reason in zip(
                texts, logps_avg_by_len, token_len, result.finish_reason
            )
        ]
        self._next_state_terminated = dict(zip(texts, [False] * len(texts)))
        return _legal_actions, result.completion_tokens

    def step(self, action, update_legal_action=True):
        """

        Args:
            action: the chosen action, which is the refined solution in this case, need to record this
            update_legal_action:

        Returns:

        """
        self.action_history.append(action)  # recording all the select full answer

        reward = self.get_reward()
        terminated, truncated, info = (
            self.get_done_and_info()
        )  # terminated or truncated when reach maximim depth
        # update legal actions
        if not (terminated or truncated) and update_legal_action:
            cnt = 0
            while cnt < 3:
                cnt += 1
                try:
                    self._legal_actions, api_completion_token = (
                        self.update_legal_actions()
                    )
                    info["api_completion_token"] = api_completion_token
                except NoLegalActionException as e:
                    if cnt == 3:
                        terminated = True
                        reward = 0
                        self._legal_actions = None
                        info["winner"] = 2
                        info["api_completion_token"] = 0
                    else:
                        pass
        else:
            self._legal_actions = None
            if info["winner"] == 1:
                reward = 1.0
            info["api_completion_token"] = 0
        return None, reward, terminated, truncated, info

    def update_legal_actions(self):
        """
        Given the current state (the completed solution to Q), a critic LLM generate review and
        a proposer LLM rewrite the answer, which is the new updated legal action
        Returns:

        """
        # retrive current answer
        assert len(self.action_history) > 0
        current_action = self.action_history[-1]

        # review
        review_prompt = self.critique_task_desc + self.critique_format_str.format(
            question=self.math_problem["question"], answer=current_action
        )
        result = self.llm_gen_fn(
            input_str=review_prompt,
            config=LMCallingConfig(
                n=self.num_reviews,
                stop_str=self.sep,
                include_stop_str_in_output=True,
                max_new_tokens=self.max_new_tokens_review,
                **self.config["generation_config"]
            ),
        )

        review_texts = result.text
        # for i in review_texts:
        #     print(f"\n review = {i}")
        # logps_avg_by_len = result.logp_avg_by_len
        # token_len = result.num_tokens

        new_action_text = []
        from_review_text = []
        new_prob_list = []
        tokens_num_list = []
        final_reason_list = []
        total_completion = result.completion_tokens
        for text in review_texts:
            rewrite_prompt = self.rewrite_task_desc + self.rewrite_format_str.format(
                question=self.math_problem["question"],
                answer=current_action,
                review=text,
            )

            result = self.llm_gen_fn(
                input_str=rewrite_prompt,
                config=LMCallingConfig(
                    n=self.num_rewrite,
                    stop_str=self.sep,
                    include_stop_str_in_output=True,
                    max_new_tokens=self.max_new_tokens_answer,
                    **self.config["generation_config"]
                ),
            )

            new_action_text += result.text
            new_prob_list += result.logp_avg_by_len
            tokens_num_list += result.num_tokens
            final_reason_list += result.finish_reason
            total_completion += result.completion_tokens
            from_review_text.append(text)

        _legal_actions = [
            {
                "action": action,
                "prob": prob,
                "num_token": n_token,
                "finish_reason": finish_reason,
                "from_review": f_review,
            }
            for action, prob, n_token, finish_reason, f_review in zip(
                new_action_text,
                new_prob_list,
                tokens_num_list,
                final_reason_list,
                from_review_text,
            )
        ]

        self._next_state_terminated = dict(
            zip(new_action_text, [False] * len(new_action_text))
        )

        return _legal_actions, total_completion

    def get_done_and_info(self):
        info = {"winner": 0}
        # done when reaches maximum length
        truncated = terminated = len(self.action_history) >= self.config["max_length"]
        assert len(self.action_history) <= self.config["max_length"]
        if terminated or truncated:
            if self._is_correct(self.action_history[-1]):
                info["winner"] = 1
            else:
                info["winner"] = 2
            return terminated, truncated, info
        return terminated, truncated, info

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
