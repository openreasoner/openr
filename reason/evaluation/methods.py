from dataclasses import dataclass
import random
import functools
from typing import Dict
from reason.inference.lm_call import LMCallingConfig, LanguageModelCallingFunction
from reason.inference.rm_call import RewardModelCallingFunction
from reason.evaluation.evaluator import SolutionOutput, Task, TreeSearchSolutionOutput
from reason.guided_search.tree import SearchTree
from reason.guided_search.rstar import RstarSearchTree
from reason.evaluation.utils import read_txt


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

@dataclass        # TODO: no need to load every time...right?
class DotsConfig(BasicConfig):
    depth: int = 5
    num_sequence: int = 1

    empty_prompt: str = ""
    rewriting_prompt: str = read_txt("reason/evaluation/dots_prompts/MATH/rewriting_prompt.txt")
    decomposition_prompt: str = read_txt("reason/evaluation/dots_prompts/MATH/decomposition_prompt.txt")
    cot_prompt: str = read_txt("reason/evaluation/dots_prompts/MATH/cot_prompt.txt")
    pot_prompt: str = read_txt("reason/evaluation/dots_prompts/MATH/pot_prompt.txt")
    self_verification_prompt: str = read_txt("reason/evaluation/dots_prompts/MATH/self_verification_prompt.txt")

    analysis_layer_prompts = [empty_prompt, rewriting_prompt, decomposition_prompt]
    solution_layer_prompts = [cot_prompt, pot_prompt]
    verification_layer_prompts = [empty_prompt, self_verification_prompt]

def dots(
    config: DotsConfig,
    gen_config: LMCallingConfig,
    problem_inst: Dict[str, str],
    lm_call: LanguageModelCallingFunction,
    rm_call: RewardModelCallingFunction,
) -> SolutionOutput:
    if gen_config.max_new_tokens < 256:
        print("Warning: max_new_tokens is less than 256")

    gen_config.n = config.num_sequence
    task = Task(task_name=config.task_name)
    # prompt = task.prompt_fn(problem_inst["question"])
    prompt = problem_inst["question"] + "\n"

    # analysis layer
    analysis_prompt = random.choice(config.analysis_layer_prompts)
    if analysis_prompt != "": # EMPTY actions means bypass
        output = lm_call(prompt+analysis_prompt, gen_config)
        prompt += (output.text[0] + "\n")

    step_num = 0
    completion_tokens = 0
    while step_num <= config.depth:
        step_num += 1

        # solution layer
        solution_prompt = random.choice(config.solution_layer_prompts)
        output = lm_call(prompt+solution_prompt, gen_config)    # TODO: PoT run the code
        completion_tokens += output.num_tokens[0]
        prompt += (output.text[0] + "\n")

        # verification layer
        verification_prompt = random.choice(config.verification_layer_prompts)
        if verification_prompt == "":     # EMPTY actions means bypass
            continue
        output = lm_call(prompt+verification_prompt, gen_config)
        prompt += (output.text[0] + "\n")
        completion_tokens += output.num_tokens[0]

        if "The answer is: correct" in output.text[0]:
            break

    return SolutionOutput(
        solutions=[prompt],
        completion_tokens=[completion_tokens],
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
