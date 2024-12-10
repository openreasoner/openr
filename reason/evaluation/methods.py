from dataclasses import dataclass
import functools
from typing import Dict
from reason.inference.lm_call import LMCallingConfig, LanguageModelCallingFunction
from reason.inference.rm_call import RewardModelCallingFunction
from reason.evaluation.evaluator import SolutionOutput, Task, TreeSearchSolutionOutput
from reason.guided_search.tree import SearchTree
from reason.guided_search.rstar import RstarSearchTree
from reason.guided_search.critic_mcts import CriticSearchTree


@dataclass
class BasicConfig:
    task_name: str


@dataclass
class CoTConfig(BasicConfig):
    pass


def cot(
    config: CoTConfig,
    gen_config: LMCallingConfig,
    problem_inst: Dict[str, str],
    llm_call: LanguageModelCallingFunction,
    rm_call: RewardModelCallingFunction,
) -> SolutionOutput:
    gen_config = LMCallingConfig(
        n=1,
        temperature=0,
        top_k=1,
        top_p=1.0,
        max_new_tokens=gen_config.max_new_tokens,
    )
    config.num_sequence = 1
    return best_of_n(config, gen_config, problem_inst, llm_call, rm_call)


@dataclass
class BestOfNConfig(BasicConfig):
    num_sequence: int = 32


def best_of_n(
    config: BestOfNConfig,
    gen_config: LMCallingConfig,
    problem_inst: Dict[str, str],
    lm_call: LanguageModelCallingFunction,
    rm_call: RewardModelCallingFunction,
) -> SolutionOutput:
    if gen_config.max_new_tokens < 256:
        print("Warning: max_new_tokens is less than 256")

    gen_config.n = config.num_sequence
    task = Task(task_name=config.task_name)
    prompt = task.prompt_fn(problem_inst["question"])
    output = lm_call(prompt, gen_config)
    completion_tokens = output.num_tokens
    return SolutionOutput(
        solutions=output.text,
        completion_tokens=completion_tokens,
    )


@dataclass
class TreeSearchConfig(BasicConfig):
    # construction config
    tree_max_width: int = 10
    tree_max_depth: int = 10
    # node config
    init_critic_value: bool = True

    def __post_init__(self):
        assert self.tree_max_width > 0, \
            "Tree width must be greater than 0"
        assert self.tree_max_depth > 0, \
            "Tree depth must be greater than 0"

@dataclass
class BeamSearchConfig(TreeSearchConfig):
    beam_size: int = 1

    def __post_init__(self):
        super().__post_init__()
        assert self.beam_size > 0, \
            "Beam size must be greater than 0"
        assert self.init_critic_value, \
            "BeamSearch should set init_critic_value to True"

def beam_search(
    config: BeamSearchConfig,
    gen_config: LMCallingConfig,
    problem_inst: Dict[str, str],
    lm_call: LanguageModelCallingFunction,
    rm_call: RewardModelCallingFunction,
) -> SolutionOutput:
    task = Task(task_name=config.task_name)
    env = task.env_fn(
        config={
            "max_actions": config.tree_max_width,
            "max_length": config.tree_max_depth,
            "stop_str": "The answer is ",
            "generation_config": {
                "max_new_tokens": gen_config.max_new_tokens,
                "temperature": gen_config.temperature,
                "top_p": gen_config.top_p,
                "top_k": gen_config.top_k,
            },
        },
        math_problems=[
            {
                "question": problem_inst["question"],
                "answer": task.extract_groundtruth(problem_inst["answer"]),
            }
        ],
        llm_gen_fn=lm_call,
        # TODO(ziyu): set sep by lm_call.lm_step_tag
    )

    search_tree = SearchTree(cfg={})
    rm_call_fn = functools.partial(rm_call, lm_step_tag=lm_call.lm_step_tag)
    traj_list = search_tree.beam_search(
        env, config.beam_size, config.tree_max_depth, rm_call_fn
    )
    return TreeSearchSolutionOutput(
        solutions=[t["text"] for t in traj_list],
        completion_tokens=[t["api_completion_tokens"] for t in traj_list],
        tree_completion_tokens=[t["tree_completion_tokens"] for t in traj_list],
    )

@dataclass
class MCTSBaseConfig(TreeSearchConfig):
    # PUCT hparams
    pb_c_base: float = 19652
    pb_c_init: float = 1.25

@dataclass
class VanilaMCTSConfig(MCTSBaseConfig):
    # rollout step strategy, if `select_by_prior` is False,
    #  then select by the initial critic value
    # otherwise, random choice by the prior probability
    select_by_prior: bool = False 
    num_path: int = 1
    
    def __post_init__(self):
        super().__post_init__()
        if not self.select_by_prior:
            assert self.init_critic_value, \
                "VanilaMCTS with greedy as rollout method should set init_critic_value to True"
        assert self.num_path > 0

def vanila_mcts(
    config: VanilaMCTSConfig,
    gen_config: LMCallingConfig,
    problem_inst: Dict[str, str],
    lm_call: LanguageModelCallingFunction,
    rm_call: RewardModelCallingFunction
):
    task = Task(task_name=config.task_name)
    env = task.env_fn(
        config={
            "max_actions": config.tree_max_width,
            "max_length": config.tree_max_depth,
            "stop_str": "The answer is ",
            "generation_config": {
                "max_new_tokens": gen_config.max_new_tokens,
                "temperature": gen_config.temperature,
                "top_p": gen_config.top_p,
                "top_k": gen_config.top_k,
            },
        },
        math_problems=[
            {
                "question": problem_inst["question"],
                "answer": task.extract_groundtruth(problem_inst["answer"]),
            }
        ],
        llm_gen_fn=lm_call,
    )

    search_tree = SearchTree(
        cfg={
            "pb_c_base": config.pb_c_base,
            "pb_c_init": config.pb_c_init,
            "init_critic_value": config.init_critic_value,
        }
    )
    rm_call_fn = functools.partial(rm_call, lm_step_tag=lm_call.lm_step_tag)
    traj_list = search_tree.vanila_mcts(
        simulate_env=env,
        num_path=config.num_path,
        reward_model_fn=rm_call_fn,
        select_by_prior=config.select_by_prior
    )
    return TreeSearchSolutionOutput(
        solutions=[t["text"] for t in traj_list],
        completion_tokens=[t["api_completion_tokens"] for t in traj_list],
        tree_completion_tokens=[t["tree_completion_tokens"] for t in traj_list],
    )


@dataclass
class RStarMCTSConfig(MCTSBaseConfig):
    # rollout step strategy, if `select_by_prior` is False,
    #  then select by the initial critic value
    # otherwise, random choice by the prior probability
    select_by_prior: bool = False
    num_path: int = 1

    def __post_init__(self):
        super().__post_init__()
        if not self.select_by_prior:
            assert self.init_critic_value, \
                "VanilaMCTS with greedy as rollout method should set init_critic_value to True"
        assert self.num_path > 0

def rstar_mcts(
        config: RStarMCTSConfig,
        gen_config: LMCallingConfig,
        problem_inst: Dict[str, str],
        lm_call: LanguageModelCallingFunction,
        rm_call: RewardModelCallingFunction
):
    task = Task(task_name=config.task_name)
    env = task.env_fn(
        config={
            "max_actions": config.tree_max_width,
            "max_length": config.tree_max_depth,
            "stop_str": "The answer is ",
            "generation_config": {
                "temperature": gen_config.temperature,
                "top_p": gen_config.top_p,
                "top_k": gen_config.top_k,      # this is fixed for each llm call
            },
        },
        math_problems=[
            {
                "question": problem_inst["question"],
                "answer": task.extract_groundtruth(problem_inst["answer"]),
            }
        ],
        llm_gen_fn=lm_call,
    )

    search_tree = RstarSearchTree(
        cfg={
            "pb_c_base": config.pb_c_base,
            "pb_c_init": config.pb_c_init,
            "init_critic_value": config.init_critic_value,
        }
    )
    rm_call_fn = functools.partial(rm_call, lm_step_tag=lm_call.lm_step_tag)
    traj_list = search_tree.rstar_mcts(
        simulate_env=env,
        num_path=config.num_path,
        reward_model_fn=rm_call_fn,
        select_by_prior=config.select_by_prior
    )
    return TreeSearchSolutionOutput(
        solutions=[t["text"] for t in traj_list],
        completion_tokens=[t["api_completion_tokens"] for t in traj_list],
        tree_completion_tokens=[t["tree_completion_tokens"] for t in traj_list],
    )


@dataclass
class CriticMCTSConfig(MCTSBaseConfig):
    # rollout step strategy, if `select_by_prior` is False,
    #  then select by the initial critic value
    # otherwise, random choice by the prior probability
    select_by_prior: bool = False
    num_path: int = 1

    def __post_init__(self):
        super().__post_init__()
        if not self.select_by_prior:
            assert self.init_critic_value, \
                "VanilaMCTS with greedy as rollout method should set init_critic_value to True"
        assert self.num_path > 0


def critic_mcts(
        config: CriticMCTSConfig,
        gen_config: LMCallingConfig,
        problem_inst: Dict[str, str],
        lm_call: LanguageModelCallingFunction,
        rm_call: RewardModelCallingFunction
):
    task = Task(task_name=config.task_name)
    env = task.env_fn(
        config={
            "max_actions": config.tree_max_width,
            "max_length": config.tree_max_depth,
            "stop_str": "The answer is ",
            "generation_config": {
                "temperature": gen_config.temperature,
                "top_p": gen_config.top_p,
                "top_k": gen_config.top_k,  # this is fixed for each llm call
            },
        },
        math_problems=[
            {
                "question": problem_inst["question"],
                "answer": task.extract_groundtruth(problem_inst["answer"]),
            }
        ],
        llm_gen_fn=lm_call,
    )

    search_tree = CriticSearchTree(
        cfg={
            "pb_c_base": config.pb_c_base,
            "pb_c_init": config.pb_c_init,
            "init_critic_value": config.init_critic_value,
        }
    )
    rm_call_fn = functools.partial(rm_call, lm_step_tag=lm_call.lm_step_tag)
    traj_list = search_tree.critic_mcts(
        simulate_env=env,
        num_path=config.num_path,
        reward_model_fn=rm_call_fn,
        select_by_prior=config.select_by_prior
    )
    return TreeSearchSolutionOutput(
        solutions=[t["text"] for t in traj_list],
        completion_tokens=[t["api_completion_tokens"] for t in traj_list],
        tree_completion_tokens=[t["tree_completion_tokens"] for t in traj_list],
        metadata={
            "review": [[f"Path {i}'s Review"] + t['review_path'] for i, t in enumerate(traj_list)],
            "rewrite": [[f"Path {i}'s Rewrite"] + t['rewrite_path'] for i, t in enumerate(traj_list)]
        }
    )