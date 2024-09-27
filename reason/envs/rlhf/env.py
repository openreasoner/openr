import copy
import re
from typing import List
import numpy as np
from transformers import PreTrainedTokenizer
from envs.base_env import TokenEnv
from .prompt import PROBLEM_FORMAT_STR, SEP

class RLHF_TokenEnv(TokenEnv):
    sep=SEP
    def __init__(
        self,
        config,
        problems,
        llm_forward_fn,
        tokenizer,
        reward_fn,
        task_desc_str= None,
        cot_example_str = None,
        problem_format_str = PROBLEM_FORMAT_STR,
        reset=True,
    ):
        super().__init__(
            config,
            problems,
            llm_forward_fn,
            tokenizer,
            task_desc_str,
            cot_example_str,
            problem_format_str,
            reset,
        )
        self.reward_fn = reward_fn

    @staticmethod
    def build_response_str(
        answer_str: str, tokenizer: PreTrainedTokenizer, add_eos_token: bool
    ):
        if (
            add_eos_token
            and len(tokenizer.encode(answer_str, add_special_tokens=False)) < 64
        ):
            answer_str += tokenizer.eos_token
        return answer_str
    
    @property
    def stop_str(self):
        return self.tokenizer.eos_token

    # def init_action_history(self):
    #     # add the first prompted questions
    #     return ([self.task_prefix] if self.task_prefix is not None else []) + [
    #         self._problem_format_str.format(self.problem['prompt'])
    #     ]

    def step(self, action, update_legal_action=True):
        terminated = False
        if not self.stop_str == action:
            # remove the final stop string like eos token
            self.action_history.append(action)
        else:
            terminated = True
        state = self.get_state()
        truncated = len(self.action_history) >= self.config["max_length"] + (
            2 if self.task_prefix is not None else 1
        )
        reward = self.get_reward(terminated, truncated)
        # update legal actions
        if not (terminated or truncated) and update_legal_action:
            self._legal_actions = self.update_legal_actions()
        else:
            self._legal_actions = None
        return state, reward, terminated, truncated, {'winner': None, 'reward': reward}
    
    def get_reward(self, terminated, truncated):
        """To implement based on learned reward model"""
        if terminated or truncated:
            reward = self.reward_fn(self.question, self.answer)
        else:
            reward = 0
        return reward
    
    def copy(self):
        env = self.__class__(
            self.config,
            self.problems,
            self.llm_forward_fn,
            self.tokenizer,
            self.reward_fn,
            self._task_desc_str,
            self._cot_example_str,
            self._problem_format_str,
            reset=False,
        )
        env.problem = copy.deepcopy(self.problem)
        env._legal_actions = copy.deepcopy(self._legal_actions)
        env.action_history = copy.deepcopy(self.action_history)
        return env


if "__name__" == '__main__':
    pass