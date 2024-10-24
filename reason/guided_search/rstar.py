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
                done = env_copy.is_terminal(
                    next_node
                )       # checking terminal condition
                # update legal action (expand) when the current code is not a leaf (no children)
                # no env.step only use for checking termination when is not leaf, and update legal action
                # when the current node is leaf

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

