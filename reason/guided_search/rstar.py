"""
rStar Implementation
"""
import copy
import json
import math

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Type
from distributed.utils import print_rank_0, print_with_rank
from envs.base_env import CoTEnv
import pdb
from tqdm import tqdm
import heapq
from copy import deepcopy
from tqdm import tqdm
from enum import Enum, unique

from .tree import Node, LanguageNode, SearchTree
from envs.rstar.rstar_utils import *


@unique
class Node_Type(Enum):
    USER_QUESTION = "USER_QUESTION"
    REPHRASED_USER_QUESTION = "REPHRASED_USER_QUESTION"
    DIRECT_ANSWER = "DIRECT_ANSWER"
    SUBQUESTION = "SUBQUESTION"
    RE_SUBANSWER = "RE_SUBANSWER"
    OST_STEP = "OST_STEP"


class RstarLanguageNode(MCTS_Node):
    def __init__(
        self,
        parent: "Reasoning_MCTS_Node",
        depth: int,
        node_type: Node_Type,
        verbose: bool = False,
        # --- For instantiating root node ---
        node_value: float = None,
        disable_a5: bool = None,
        user_question: str = None,
        max_depth_allowed: int = None,
        disable_a1: bool = None,

        # --- For instantiating REPHRASED_USER_QUESTION node ---
        rephrased_user_question: str = None,
        # ------------------------------------------------------
        expected_answer: str = None,
        # --- For instantiating DIRECT_ANSWER node ---
        direct_answer: str = None,
        # --------------------------------------------
        # --- For instantiating SUBQUESTION node ---
        subquestion: str = None,
        subanswer: str = None,
        is_new_subquestion: bool = None,
        # ------------------------------------------
        # --- For instantiating RE_SUBANSWER node ---
        re_subanswer: str = None,
        # -------------------------------------------
        # --- For instantiating OST_STEP node ---
        ost_step: str = None,
        question_index: int = None,
    ) -> None:
        super().__init__()

        #! sanity checks
        try:
            assert depth is not None
            assert node_type is not None
            if node_value is not None:
                assert node_value > 0, breakpoint()

            if node_type is Node_Type.USER_QUESTION:
                assert depth == 0
                assert all(
                    attr is None
                    for attr in [
                        parent,
                        node_value,
                        rephrased_user_question,
                        direct_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        re_subanswer,
                        ost_step,
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [disable_a5, user_question, expected_answer, max_depth_allowed, disable_a1]
                )
            elif node_type is Node_Type.REPHRASED_USER_QUESTION:
                assert depth == 1
                assert all(
                    attr is None
                    for attr in [
                        node_value,
                        disable_a5,
                        user_question,
                        expected_answer,
                        direct_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        re_subanswer,
                        ost_step,
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
                assert all(attr is not None for attr in [parent, rephrased_user_question])
            elif node_type is Node_Type.DIRECT_ANSWER:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        disable_a5,
                        user_question,
                        expected_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        re_subanswer,
                        ost_step,
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
                assert all(attr is not None for attr in [parent, node_value, direct_answer])
            elif node_type is Node_Type.SUBQUESTION:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        disable_a5,
                        user_question,
                        expected_answer,
                        direct_answer,
                        re_subanswer,
                        ost_step,
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
                assert all(
                    attr is not None for attr in [parent, node_value, subquestion, subanswer, is_new_subquestion]
                )
            elif node_type is Node_Type.RE_SUBANSWER:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        disable_a5,
                        user_question,
                        expected_answer,
                        direct_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        ost_step,
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
                assert all(attr is not None for attr in [parent, node_value, re_subanswer])
            elif node_type is Node_Type.OST_STEP:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        node_value,
                        disable_a5,
                        user_question,
                        rephrased_user_question,
                        expected_answer,
                        direct_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        re_subanswer,
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
                assert all(attr is not None for attr in [parent, ost_step])
        except AssertionError:
            print(f"Instantiating node with type {node_type} failed!")
            breakpoint()
            exit()

        #! attributes
        self.parent = parent  # if parent is None, then the node is the root
        self.children: List["Reasoning_MCTS_Node"] = []
        self.depth = depth
        self.node_type = node_type
        self.node_value = node_value
        self.direct_answer = direct_answer
        self.subquestion = subquestion
        self.subanswer = subanswer
        self.is_new_subquestion = is_new_subquestion
        self.re_subanswer = re_subanswer
        self.ost_step = ost_step

        ## additional parameter
        self._visit_count = 0
        self._value_sum = 0     # self.node_value

        if parent is None:  # root
            self.verbose = verbose
            self.user_question = user_question
            self.expected_answer = expected_answer
            self.disable_a5 = disable_a5
            self.question_index = question_index
            self.max_depth_allowed = max_depth_allowed
            self.disable_a1 = disable_a1
        else:  # inherit from parent
            self.verbose = parent.verbose
            self.user_question = parent.user_question
            self.expected_answer = parent.expected_answer
            self.disable_a5 = parent.disable_a5
            self.question_index = parent.question_index
            self.max_depth_allowed = parent.max_depth_allowed
            self.disable_a1 = parent.disable_a1

        #! keep track of paraphrasing
        if node_type is Node_Type.USER_QUESTION:
            self.paraphrased = False
        elif node_type is Node_Type.REPHRASED_USER_QUESTION:
            self.paraphrased = True
            self.user_question = rephrased_user_question
        else:
            assert parent is not None
            self.paraphrased = parent.paraphrased

        #! record number of subquestions till now
        if parent is None:  # root
            self.subquestion_counter = 0
        else:
            if node_type is Node_Type.SUBQUESTION and is_new_subquestion:
                self.subquestion_counter = parent.subquestion_counter + 1
            else:
                self.subquestion_counter = parent.subquestion_counter

        #! record number of one-step thought steps till now
        if parent is None:  # root
            self.ost_step_counter = 0
        else:
            if node_type is Node_Type.OST_STEP:
                self.ost_step_counter = parent.ost_step_counter + 1
            else:
                self.ost_step_counter = parent.ost_step_counter

        #! record solution trace from root to the current node. key: subquestion id
        if parent is None:  # root
            assert self.node_type is Node_Type.USER_QUESTION
            self.solution_trace: Dict[int, Dict[str, str]] = {0: {"user_question": user_question, "ost_step": {}}}
        else:
            assert self.node_type is not Node_Type.USER_QUESTION
            self.solution_trace = deepcopy(parent.solution_trace)

            if node_type is Node_Type.REPHRASED_USER_QUESTION:
                self.solution_trace[0]["user_question"] = rephrased_user_question
            elif node_type is Node_Type.DIRECT_ANSWER:
                assert self.subquestion_counter in self.solution_trace.keys()
                assert self.subquestion_counter == parent.subquestion_counter
                self.solution_trace[self.subquestion_counter]["direct_answer"] = {
                    "text": direct_answer,
                    "value": node_value,
                }
            elif node_type is Node_Type.SUBQUESTION:
                assert is_new_subquestion and self.subquestion_counter == parent.subquestion_counter + 1
                self.solution_trace[self.subquestion_counter] = {
                    "subquestion": subquestion,
                    "subanswer": {"text": subanswer, "value": node_value},
                    "ost_step": {},
                }
            elif node_type is Node_Type.RE_SUBANSWER:
                assert parent.subquestion is not None
                assert self.subquestion_counter == parent.subquestion_counter
                assert self.solution_trace[self.subquestion_counter]["subquestion"] == parent.subquestion
                self.solution_trace[self.subquestion_counter]["subanswer"] = {"text": re_subanswer, "value": node_value}
            elif node_type is Node_Type.OST_STEP:
                assert "ost_step" in self.solution_trace[self.subquestion_counter].keys()
                self.solution_trace[self.subquestion_counter]["ost_step"][self.ost_step_counter] = ost_step

    def __str__(self) -> str:
        type2str = {
            Node_Type.USER_QUESTION: "U",
            Node_Type.REPHRASED_USER_QUESTION: "RU",
            Node_Type.DIRECT_ANSWER: "DA",
            Node_Type.SUBQUESTION: "SQ",
            Node_Type.RE_SUBANSWER: "RS",
            Node_Type.OST_STEP: "TS",
        }
        return f"{type2str[self.node_type]}-{self.id}"


    def is_valid_leaf_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type
        return (
            self.node_type is Node_Type.SUBQUESTION and reach_terminal_subquestion(self.subquestion, self.user_question)
        ) or self.node_type is Node_Type.DIRECT_ANSWER

    def is_valid_solution_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type or OST_STEP type
        return (
            (
                self.node_type is Node_Type.SUBQUESTION
                and reach_terminal_subquestion(self.subquestion, self.user_question)
            )
            or (self.node_type is Node_Type.OST_STEP and reach_terminal_ost_step(self.ost_step))
            or self.node_type is Node_Type.DIRECT_ANSWER
        )

    @property
    def value(self) -> float:
        """
        Overview:
            The value of the current node.
        Returns:
            - output (:obj:`Int`): Current value, used to compute ucb score.
        """
        if self._visit_count == 0:
            # if not visited, return the initial value
            return self._value_sum
        return self._value_sum / self._visit_count

    def update(self, value: float) -> None:
        """
        Overview:
            Updata the current node information, such as visit_count and value_sum.
        Arguments:
            - value (:obj:`Int`): The value of the node.
        """
        self._visit_count += 1
        self._value_sum += value


class RstarSearchTree(SearchTree):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.parent2children: Dict[RstarLanguageNode, List[RstarLanguageNode]] = dict()
        self.explored_nodes = set()
        self.N: Dict[RstarLanguageNode, int] = defaultdict(lambda: 0)  # total visit count for each node
        self.Q: Dict[MCTS_Node, float] = defaultdict(lambda: 0.0)  # total reward of each node
        self.weight_scheduler = 'const'
        self.mcts_exploration_weight = 2.0

    @override
    def _select_child(self,
                      node: RstarLanguageNode,
                      simulate_env: Type[CoTEnv],
                      rollout_id: int) -> Tuple[RstarLanguageNode, bool]:

        # for select, if there is unexplored children, select it randomly. if all children nodes
        # have been explored, select UCB, a leaf node means it has no children, return True
        # when there is unexplored node

        if node not in self.parent2children.keys():
            return node, True

        # if there are children unexplored
        unexplored = [n for n in self.parent2children[node] if n not in self.explored_nodes]
        if unexplored:
            next_node = random.choice(unexplored)
            return next_node, True

        # if all have been explord, from parent2children dict, select one node with highest UCB score
        next_node = max(
            self.parent2children[node],
            key=lambda n: self._compute_uct(
                parent_node=node, node=n, rollout_id=rollout_id
            )
        )

        return next_node, False

    def _compute_uct(self,
                     parent_node: RstarLanguageNode,
                     node: RstarLanguageNode,
                     rollout_id: int):
        "Upper confidence bound for trees"
        if parent_node is None:  # invalid UCT: the node is the root
            return 666
        else:
            if self.N[node] == 0:  # invalid UCT: the node has not been explored yet
                return 999
            else:
                weight = self._get_weight(rollout_id)
                return self.Q[node] / self.N[node] + weight * math.sqrt(math.log(self.N[parent_node]) / self.N[node])


    def _get_weight(self, rollout_id: int):
        # start with exploration weight, end with 0.1 * exploration weight
        if self.weight_scheduler == "exp":
            return self.mcts_exploration_weight * (0.1 ** (rollout_id / self.num_path))
        elif self.weight_scheduler == "lin":
            return self.mcts_exploration_weight * (1 - 0.9 * (rollout_id / self.num_path))
        elif self.weight_scheduler == "const":
            return self.mcts_exploration_weight



    def rstar_mcts(
        self,
        simulate_env: Type[CoTEnv],
        num_path: int,
        reward_model_fn: Optional[Callable] = None,
        select_by_prior: bool = False,
    ) -> List[Dict]:
        api_call_completion_tokens = 0
        simulate_env.reset()
        # api_call_completion_tokens += info["api_completion_token"]
        if self.root is None:
            root = RstarLanguageNode(
                                    parent=None,
                                    depth=0,
                                    node_type=Node_Type.USER_QUESTION,
                                    disable_a5=False,
                                    user_question=simulate_env.math_problem['question'],
                                    expected_answer=simulate_env.math_problem['answer'],
                                    max_depth_allowed=5,
                                    disable_a1=False,
                                    )
            # updated_node = simulate_env.try_update_legal_action(node=root)          # creating children on root
            # api_call_completion_tokens += info["api_completion_token"]

            # self._expand_leaf_node(root, simulate_env, reward_model_fn)     # this we do expansion in simulated env already, here we just compute the
            self.root = root

        traj_list = []

        self.num_path = num_path
        model_solutions = []
        model_all_solutions = []
        model_rollout_nodes = []

        for i_path in tqdm(range(num_path), desc="Running MCTS"):
            node = self.root
            # node.set_rollout_id(i_path)
            # node.set_unique_id()        # set unique id

            env_copy = simulate_env.copy()
            done = False
            path = []
            while not done:
                # this is the whole process of navigating from root to a terminate node, along the way,
                # there are explored nodes with children where we do UCB, and there are unexplored nodes where we
                # expand its children through legal_action_update,
                # for select, if there is unexplored children, select it randomly. if all children nodes
                # have been explored, select UCB
                next_node, is_leaf = self._select_child(node, env_copy, i_path)     # find a leaf node or
                                                                            # simulate the remaining
                path.append(next_node)
                _, _, terminated, truncated, info = env_copy.step(
                    next_node
                )       # checking terminal condition
                # update legal action (expand) when the current code is not a leaf (no children)
                # no env.step only use for checking termination when is not leaf, and update legal action
                # when the current node is leaf

                done = terminated or truncated
                if not done and is_leaf:     # expand when encounter a leaf node one step further
                    next_node_children = env_copy.try_update_legal_action(node = next_node)
                    next_node_children.set_rollout_id(i_path)
                    self.parent2children[next_node] = next_node_children
                    # self._expand_leaf_node()

                if done:
                    self.explored_nodes.add(next_node)

            else:
                # boostrapping
                reward = next_node.calculate_reward()
                for node in reversed(path):
                    self.Q[node] += reward
                    self.N[node] += 1
                    self.explored_nodes.add(node)


            traj_data = {
                "path_idx": i_path,
                "text": env_copy.answer,
                "value": reward,
                "api_completion_tokens": api_call_completion_tokens,
                "tree_completion_tokens": self._completion_tokens,
            }

            traj_list.append(traj_data)

            # reset api_call_completion_tokens
            # api_call_completion_tokens = 0

            model_rollout_nodes.append(next_node)
            _, best_solution, _, chosen_node, all_solution_nodes, all_solutions = stochastic_find_best_solution(
                self.root, env_copy.evaluator, enable_potential_score=False
            )

            model_solutions.append(best_solution)
            model_all_solutions.append(all_solutions)


        return []

