import abc
from typing import Dict, List, Optional
import numpy as np
import copy
import pdb
import torch
from distributed.utils import print_with_rank
from transformers import PreTrainedTokenizer

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
        sep: str,
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

        # THIS is because CoTEnv.answer/get_state() append "\n"
        ret += sep
        return ret

    @staticmethod
    def build_response_str(
        answer_str: str, tokenizer: PreTrainedTokenizer, add_eos_token: bool
    ):
        raise NotImplementedError


class CoTEnv(BaseEnv):
    """The basic environment for solving natural language problems using CoT"""

    sep: str

    @staticmethod
    def build_response_str(
        answer_str: str, tokenizer: PreTrainedTokenizer, add_eos_token: bool
    ):
        if add_eos_token:
            # if text ends with </s>, remove it so the follwing strip can remove \n
            #  and in latter add_traj, \n and </s> will be added again
            if answer_str.endswith(tokenizer.eos_token):
                answer_str = answer_str.replace(tokenizer.eos_token, "")
            answer_str = answer_str.strip()
            answer_str += tokenizer.eos_token
        return answer_str

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
        tokenizer,
        task_desc_str: str,
        cot_example_str: str,
        problem_format_str: str,
        reset=True,
    ):
        self.config = config
        self.mcts_mode = "play_with_bot_mode"
        self.math_problems = math_problems
        self.llm_gen_fn = llm_gen_fn
        self.tokenizer = tokenizer
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
        self.action_history = self.init_action_history()
        if update_legal_action:
            cnt = 0
            while cnt < 3:
                cnt += 1
                try:
                    self._legal_actions = self.update_legal_actions()
                    break
                except NoLegalActionException as e:
                    if cnt == 3:
                        raise ResetException
        return self.get_state()

    def step(self, action, update_legal_action=True):
        self.action_history.append(action)
        state = self.get_state()
        reward = self.get_reward()
        terminated, truncated, info = self.get_done_and_info()
        # update legal actions
        if not (terminated or truncated) and update_legal_action:
            try:
                self._legal_actions = self.update_legal_actions()
            except NoLegalActionException as e:
                terminated = True
                reward = 0
                self._legal_actions = None
                info["winner"] = 2
        else:
            self._legal_actions = None
            if info["winner"] == 1:
                reward = 1.0
        return state, reward, terminated, truncated, info

    def get_state(self):
        return "\n".join(self.action_history) + "\n"

    def init_action_history(self):
        # add the first prompted questions
        return ([self.task_prefix] if self.task_prefix is not None else []) + [
            # f"Question: {self.math_problem['question']}\nAnswer: Let's think step by step"
            self._problem_format_str.format(question=self.math_problem["question"])
        ]

    def update_legal_actions(self):
        def reduce_prob_list(prob_list: List[List]) -> List:
            ans_list = []
            for scores in prob_list:
                ans_list.append(np.exp(np.mean(scores)))
            return ans_list

        prefix = (
            (self.action_history[0] + "\n") if self.task_prefix is not None else None
        )
        act_hist_start_i = 0 if self.task_prefix is None else 1
        unprefixed_state = "\n".join(self.action_history[act_hist_start_i:]) + "\n"
        texts, logps = self.llm_gen_fn(
            static_prompt=prefix,
            prompt=unprefixed_state,
            num_sequence=self.config["max_actions"],
            stop=[13, self.tokenizer.eos_token_id],
            **self.config["generation_config"],
        )

        text_list, prob_list = [], []

        for i in range(len(texts)):
            if len(texts[i]) > 0 and texts[i] not in text_list:
                text_list.append(texts[i])
                prob_list.append(logps[i])

        if len(prob_list) == 0:
            print_with_rank(
                "{} {} {}".format(prefix, act_hist_start_i, unprefixed_state)
            )
            raise NoLegalActionException("No possible action have been generated.")

        prob_list = reduce_prob_list(prob_list)
        prob_list = np.array(prob_list)
        # normalize probability
        prob_list = prob_list / np.sum(prob_list)
        # set add special tokens as False to remove bos/eos tokens
        num_token_list = [
            len(self.tokenizer.encode(txt, add_special_tokens=False))
            for txt in text_list
        ]
        _legal_actions = [
            {"action": action, "prob": prob, "num_token": n_token}
            for action, prob, n_token in zip(text_list, prob_list, num_token_list)
        ]

        return _legal_actions

    def set_problem(self, idx):
        self.math_problem = self.math_problems[idx]

    @property
    def question(self):
        return (
            "\n".join(self.action_history[:1]) + "\n"
            if self.task_prefix is None
            else "\n".join(self.action_history[:2]) + "\n"
        )

    @property
    def answer(self):
        return (
            "\n".join(self.action_history[1:]) + "\n"
            if self.task_prefix is None
            else "\n".join(self.action_history[2:]) + "\n"
        )

    def get_done_and_info(self):
        info = {"winner": 0}
        # done when reaches maximum length or LLM generates stop words
        terminated = self.stop_str in self.action_history[-1]

        truncated = len(self.action_history) >= self.config["max_length"] + (
            2 if self.task_prefix is not None else 1
        )
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
            self.tokenizer,
            self._task_desc_str,
            self._cot_example_str,
            self._problem_format_str,
            reset=False,
        )
        env.math_problem = copy.deepcopy(self.math_problem)
        env._legal_actions = copy.deepcopy(self._legal_actions)
        env.action_history = copy.deepcopy(self.action_history)
        return env

    @property
    def legal_actions(self):
        return self._legal_actions
