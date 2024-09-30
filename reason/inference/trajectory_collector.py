from typing import Optional
from envs.base_env import CoTEnv
from reason.reranking.vote_utils import MAJORITY_VOTE
from reason.mcts.tree import SearchTree
from reason.mcts.utils import get_root
import time


def _mcts_rollout_v1(
    mcts: SearchTree,
    env: CoTEnv,
    policy_forward_value,
    n_rollout: int,
    reset_total_tree: bool,
    sample: bool,
    clear_total_tree: bool,
):
    """MCTS.GET_NEXT_ACTION"""
    output_episodes = []
    num_generated_token = 0
    env.reset(True)
    mcts.root = None
    done = False
    for i in range(n_rollout):
        while not done:
            action, _, current_node = mcts.get_next_action(
                env,
                policy_forward_fn=policy_forward_value,
                sample=sample,
                return_tree=True,
            )
            mcts.root = current_node.children[action]
            next_state, reward, terminated, truncated, info = env.step(
                action, update_legal_action=len(mcts.root.children) == 0
            )
            done = terminated or truncated

            if not done and len(mcts.root.children) > 0:
                env._legal_actions = [
                    {"action": a, "prob": None} for a in mcts.root.children.keys()
                ]

        num_generated_token = mcts.num_generated_token

        traj_data = {
            "path_idx": i,
            "text": env.answer.strip(),  # drop the last "\n"
            "value": mcts.root.value,
            "num_generated_token": num_generated_token,
        }
        output_episodes.append(traj_data)

        assert not (reset_total_tree and clear_total_tree)  # cannot be both true
        if reset_total_tree:
            if i < n_rollout - 1:
                mcts.root = None
                env.reset(update_legal_action=True)
        else:
            mcts.root = get_root(current_node)
            if clear_total_tree:
                mcts.clear_node(mcts.root)
            env.reset(update_legal_action=False)
            env._legal_actions = [
                {"action": a, "prob": None} for a in mcts.root.children.keys()
            ]
        done = False

    return output_episodes


def _mcts_rollout_v2(
    mcts: SearchTree,
    env: CoTEnv,
    policy_forward_value,
    n_rollout: int,
    max_simulation: Optional[int],
    max_token: Optional[int],
):
    """MCTS.ROLLOUT"""

    output_list, num_simulation, root = mcts.rollout(
        env,
        n_rollout,
        policy_forward_value,
        max_num_simulation=max_simulation,
        max_token=max_token,
        return_tree=True,
    )

    # texts = [x["text"].strip() for x in output_list]
    # values = [x["value"] for x in output_list]
    # num_generated_token = mcts.num_generated_token

    return output_list  # texts, values, num_generated_token
