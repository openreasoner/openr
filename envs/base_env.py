import abc
from typing import Dict, List, Optional
import numpy as np
import copy
import pdb
import torch
from distributed.utils import print_with_rank
from transformers import PreTrainedTokenizer
from reason.inference.lm_call import LMCallingConfig, ConcatedLMGenResult

INVALID_ANS = "[invalid]"


class NoLegalActionException(Exception):
    pass


class ResetException(Exception):
    pass


class BaseEnv(abc.ABC):
    """Basic environment to use for MCTS"""

    @abc.abstractmethod
    def reset(self, update_legal_action: bool):
        raise NotImplementedError

    @abc.abstractmethod
    def step(self):
        raise NotImplementedError

    @abc.abstractproperty
    def legal_actions(self):
        raise NotImplementedError

    @abc.abstractmethod
    def copy(self):
        raise NotImplementedError

    @staticmethod
    def build_query_str(
        cot_task_desc: Optional[str],
        cot_examples: Optional[str],
        problem_format_str: str,
        problem_input: str,
        is_few_shot: bool = False,
    ):
        """a wrap function that wrap the problem text with certrain format
        e.g. prompt_str = "Input: " + join_numbers(" ", xs) + "\nSteps:\n"
        >>> query_str = Game24Env.build_query_str("1 1 1 1")
        >>> print(query_str)
        >>> Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.
        Input: 1 1 1 1
        Steps:

        >>>
        """

        ret = ""
        if cot_task_desc:
            ret += cot_task_desc + "\n"
        if is_few_shot:
            ret += cot_examples + "\n"
        ret += problem_format_str.format(question=problem_input)

        return ret

    @staticmethod
    def build_response_str(
        answer_str: str, tokenizer: PreTrainedTokenizer, add_eos_token: bool
    ):
        raise NotImplementedError


class CoTEnv(BaseEnv):
    """The basic environment for solving natural language problems using CoT"""

    sep: str

    @property
    def stop_str(self):
        return NotImplementedError

    def _is_correct(self, completion) -> bool:
        raise NotImplementedError

    def get_reward(self):
        """To implement based on learned reward model"""
        raise NotImplementedError

    def __init__(
        self,
        config,
        math_problems,
        llm_gen_fn,
        task_desc_str: str,
        cot_example_str: str,
        problem_format_str: str,
        reset=True,
    ):
        self.config = config
        self.mcts_mode = "play_with_bot_mode"
        self.math_problems = math_problems
        self.llm_gen_fn = llm_gen_fn
        self.action_history = None
        self.math_problem = None
        self._legal_actions = None
        self.is_few_shot = config.get("is_few_shot", False)

        self._task_desc_str = task_desc_str
        self._cot_example_str = cot_example_str
        self._problem_format_str = problem_format_str

        prefixes = []
        if self._task_desc_str is not None:
            prefixes.append(self._task_desc_str)
        if self.is_few_shot:
            prefixes.append(self._cot_example_str)
        if len(prefixes) > 0:
            self.task_prefix = "\n".join(prefixes)
        else:
            self.task_prefix = None

        if reset:
            self.reset(update_legal_action=True)

    def reset(self, update_legal_action=True):
        # reset environment to problem idx
        self.set_problem(idx=0)
        self.action_history = []
        self._init_query = self.build_query_str(
            cot_examples=self._cot_example_str,
            cot_task_desc=self._task_desc_str,
            problem_format_str=self._problem_format_str,
            problem_input=self.math_problem["question"],
            is_few_shot=self.is_few_shot
        )
        if update_legal_action:
            cnt = 0
            while cnt < 3:
                cnt += 1
                try:
                    self._legal_actions, api_completion_token = self.update_legal_actions()
                    break
                except NoLegalActionException as e:
                    if cnt == 3:
                        raise ResetException
        info = {"api_completion_token": api_completion_token}
        return self.get_state(), info 

    def step(self, action, update_legal_action=True):
        self.action_history.append(action)
        state = self.get_state()
        reward = self.get_reward()
        terminated, truncated, info = self.get_done_and_info()
        # update legal actions
        if not (terminated or truncated) and update_legal_action:
            try:
                self._legal_actions, api_completion_token = self.update_legal_actions()
                info["api_completion_token"] = api_completion_token
            except NoLegalActionException as e:
                terminated = True
                reward = 0
                self._legal_actions = None
                info["winner"] = 2
        else:
            self._legal_actions = None
            if info["winner"] == 1:
                reward = 1.0
            info["api_completion_token"] = 0
        return state, reward, terminated, truncated, info

    def get_state(self):
        ret = self._init_query + self.sep.join(self.action_history)
        if len(self.action_history) > 0:
            ret += self.sep
        return ret

    def update_legal_actions(self):
        result: ConcatedLMGenResult = self.llm_gen_fn(
            input_str=self.get_state(),
            config=LMCallingConfig(
                n=self.config["max_actions"],
                stop_str=self.sep,
                **self.config["generation_config"]
            ),
        )
        texts = result.text
        logps_avg_by_len = result.logp_avg_by_len
        token_len = result.num_tokens
        text_list, prob_list, num_token_list = [], [], []

        for i in range(len(texts)):
            if len(texts[i]) > 0 and texts[i] not in text_list:
                text_list.append(texts[i])
                prob_list.append(logps_avg_by_len[i])
                num_token_list.append(token_len[i])

        if len(prob_list) == 0:
            print_with_rank(
                "state: {}".format(self.get_state())
            )
            print_with_rank("gen_result: {}".format(result))
            raise NoLegalActionException("No possible action have been generated.")

        prob_list = np.exp(prob_list)
        prob_list = np.array(prob_list)
        # normalize probability
        prob_list = prob_list / np.sum(prob_list)

        _legal_actions = [
            {"action": action, "prob": prob, "num_token": n_token}
            for action, prob, n_token in zip(text_list, prob_list, num_token_list)
        ]
        # print(num_token_list, sum(num_token_list), len(num_token_list), result.completion_tokens)

        return _legal_actions, result.completion_tokens

    def set_problem(self, idx):
        self.math_problem = self.math_problems[idx]

    @property
    def question(self):
        return self._init_query

    @property
    def answer(self):
        return self.sep.join(self.action_history)

    def get_done_and_info(self):
        info = {"winner": 0}
        # done when reaches maximum length or LLM generates stop words
        terminated = self.stop_str in self.action_history[-1]

        truncated = len(self.action_history) >= self.config["max_length"]
        assert len(self.action_history) <= self.config["max_length"] + (
            2 if self.task_prefix is not None else 1
        ), "action history length: {}, max length: {}".format(
            len(self.action_history),
            self.config["max_length"] + (2 if self.task_prefix is not None else 1),
        )

        if terminated or truncated:
            if self._is_correct(self.action_history[-1]):
                info["winner"] = 1
            else:
                info["winner"] = 2
            return terminated, truncated, info
        return terminated, truncated, info

    def copy(self):
        env = self.__class__(
            self.config,
            self.math_problems,
            self.llm_gen_fn,
            self._task_desc_str,
            self._cot_example_str,
            self._problem_format_str,
            reset=False,
        )
        env.math_problem = copy.deepcopy(self.math_problem)
        env._legal_actions = copy.deepcopy(self._legal_actions)
        env.action_history = copy.deepcopy(self.action_history)
        env._init_query = copy.deepcopy(self._init_query)
        return env

    @property
    def legal_actions(self):
        return self._legal_actions
