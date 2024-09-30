from pathlib import Path
from typing import Dict, List, Optional
import torch.distributed as dist
from argparse_utils import str2bool
from distributed.utils import (
    print_rank_0,
    print_with_rank,
    init_distributed,
    gather_scalar,
)
from envs import get_env_datasets, get_default_query_str_builder
from inference.trajectory_collector import _mcts_rollout_v1
from inference.value import value_fn
from llm.text_generation import llm_gen_with_logp_fastchat_vllm
from inference.value import _value_inference_fastchat
from model import load_critic_model
from reason.mcts.tree import SearchTree
from inference.evaluation.vote_utils import (
    AGG_FN_MAP,
    MAJORITY_VOTE,
    ORM_VOTE,
    ORM_MAX,
)
from envs.base_env import INVALID_ANS
from transformers import AutoTokenizer
import torch
from functools import partial
import json
import jsonlines
import time
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from argparse import ArgumentParser
import os
import importlib
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


CHOSEN_AGGR_METHODS = [MAJORITY_VOTE, ORM_VOTE, ORM_MAX]


def judge_ans(
    problem_str: str,
    extracted_groundtruth: str,
    output_list: List[str],
    v_list: List[float],
    aggration_mode: str,
    extract_answer_fn,
    judge_correct_fn,
):
    ans_list = [extract_answer_fn(txt) for txt in output_list]
    valid_ans_list, valid_v_list = [], []
    for i, ans in enumerate(ans_list):
        if ans != INVALID_ANS:
            valid_ans_list.append(ans)
            valid_v_list.append(v_list[i])

    if len(valid_ans_list) == 0:
        return 0

    # score_normalization: this is only necessary for [-1, 1] values
    valid_v_list = np.array(valid_v_list)
    valid_v_list -= valid_v_list.min()
    valid_v_list /= valid_v_list.max() + 1e-3
    valid_v_list = valid_v_list.tolist()
    aggregated_ans = AGG_FN_MAP[aggration_mode](valid_ans_list, valid_v_list)

    return (
        1 if judge_correct_fn(problem_str, extracted_groundtruth, aggregated_ans) else 0
    )


def get_correct_proportion(
    problem_str: str,
    extracted_groundtruth: str,
    output_list: List[str],
    extract_answer_fn,
    judge_correct_fn,
):
    correct_list = [
        (
            1.0
            if judge_correct_fn(
                problem_str, extracted_groundtruth, extract_answer_fn(txt)
            )
            else 0.0
        )
        for txt in output_list
    ]
    if len(correct_list) > 0:
        return np.mean(correct_list).item()
    else:
        return 0.0


@dataclass
class SearchArgs:
    # temperature used for llm generation in CoT(-SC) and MCTS tree expansion
    temperature: float = 1.0
    # COT-SC number
    k_maj: int = 100

    # MCTS aggregation number
    num_mcts_aggregation: int = 5

    # which tree search methods to use
    #  ["mcts.get_next_action", "mcts.rap", "mcts.rollout", "dfs", "bfs"]
    # "mcts.get_next_action" is MCTS-$\alpha$ in paper
    # "mcts.rap" is MCTS in paper
    # "mcts.rollout" is MCTS-Rollout in paper
    rollout_method: str = None

    # Tree Search building configs
    max_length: int = 8
    max_action: int = 6

    # general mcts hyperparameters for MCTS-alpha, MCTS, MCTS-Rollout
    # Tree basic configs
    pb_c_init: float = 10

    # MCTS-alpha hyperparamerters
    num_simulations: int = 10
    reset_total_tree: bool = False
    mcts_sample: bool = False
    clear_tree: bool = False

    # MCTS-Rollout Hyperparameters
    max_simulation: Optional[int] = None
    max_token: Optional[int] = None

    # DFS hyperparameters
    prune_ratio: Optional[float] = None
    prune_value: Optional[float] = None

    # if set method to be mcts.rap and set this to be True, then
    #  it samples with llm's prior on the tree space, which is
    #  CoT-SC-Tree
    select_by_prior: bool = False

    seed: int = 7


if __name__ == "__main__":
    TEST_NO_TERMINAL = int(os.getenv("TEST_NO_TERMINAL", 0))
    TEST_WITH_TERMINAL = int(os.getenv("TEST_WITH_TERMINAL", 0))
    TEST_COT_GREEDY = int(os.getenv("TEST_COT_GREEDY", 0))
    TEST_COT_SC = int(os.getenv("TEST_COT_SC", 0))
    assert TEST_NO_TERMINAL + TEST_WITH_TERMINAL + TEST_COT_SC + TEST_COT_GREEDY > 0

    parser = ArgumentParser()
    parser.add_argument("--critic_model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--state_dict_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--env_name", type=str, default="gsm8k")
    parser.add_argument("--test", type=str2bool, default=True)
    parser.add_argument("--is_few_shot", type=str2bool, default=False)
    parser.add_argument("--controller_addr", type=str, default="http://0.0.0.0:28778")
    config = parser.parse_args()

    TREE_MAX_LENGTH = 8
    TREE_MAX_ACTIONS = 6
    # RANDOM_SEEDS = [x * 10009 + 7 for x in [0, 1, 2]]
    args_list = [
        {
            "temperature": 1.0,
            "max_length": TREE_MAX_LENGTH,
            "max_action": TREE_MAX_ACTIONS,
            "pb_c_init": 3,
            "num_simulations": 5,
            "k_maj": 10,
            "num_mcts_aggregation": 1,
            "max_simulation": None,
            "max_token": 51200,
            "rollout_method": "mcts.get_next_action",
            "select_by_prior": True,
            "reset_total_tree": False,
            "mcts_sample": False,
            "clear_tree": True,
            "prune_ratio": 0.7,
            "prune_value": None,
            "seed": 7,
        },
    ]

    task_module = importlib.import_module(f"envs.{config.env_name}")
    extract_answer = task_module.extract_answer
    extract_groundtruth = task_module.extract_groundtruth
    judge_correct = task_module.judge_correct

    save_dir = Path(config.save_dir) / config.env_name

    local_rank, world_size = init_distributed()

    print_rank_0("ENV: {}, test set: {}".format(config.env_name, config.test))
    train_ds, test_ds = get_env_datasets(config.env_name)
    if not config.test:
        test_ds = train_ds

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)

    # fake value
    def policy_forward_value(x):
        return np.array([0.0] * len(x))

    llm_gen_fn = partial(
        llm_gen_with_logp_fastchat_vllm, controller_addr=config.controller_addr
    )

    def prompt_fn(problem_input: str):
        return get_default_query_str_builder(config.env_name)(
            problem_input, is_few_shot=config.is_few_shot
        )

    def cot_direct_output(args, problem_inst, stop, **kwargs):
        prompt = prompt_fn(problem_inst["question"])
        max_new_tokens = kwargs.pop("max_new_tokens", 256)
        max_new_tokens = max(256, max_new_tokens)
        texts, logps = llm_gen_fn(
            static_prompt=None,
            prompt=prompt,
            num_sequence=1,
            stop=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        extracted_groundtruth = extract_groundtruth(problem_inst["answer"])
        value_list = policy_forward_value(
            [prompt + txt + task_module.SEP for txt in texts]
        ).tolist()

        output_list = [
            {"path_idx": i, "text": txt, "value": v}
            for i, (txt, v) in enumerate(zip(texts, value_list))
        ]

        return (
            judge_ans(
                problem_inst["question"],
                extracted_groundtruth,
                texts,
                value_list,
                MAJORITY_VOTE,
                extract_answer,
                judge_correct,
            ),
            output_list,
        )

    def cot_sc_output(args, problem_inst, stop, **kwargs):
        prompt = prompt_fn(problem_inst["question"])
        max_new_tokens = kwargs.pop("max_new_tokens", 256)
        max_new_tokens = max(256, max_new_tokens)
        texts, logps = llm_gen_fn(
            static_prompt=None,
            prompt=prompt,
            num_sequence=args.k_maj,
            stop=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            max_batch_size=50,
            **kwargs,
        )
        extracted_groundtruth = extract_groundtruth(problem_inst["answer"])
        value_list = []
        for i in range(0, len(texts), 25):
            value_list.extend(
                policy_forward_value(
                    [prompt + txt + task_module.SEP for txt in texts[i : i + 25]]
                ).tolist()
            )
        judge_results = {
            f"{k}@{args.k_maj}": judge_ans(
                problem_inst["question"],
                extracted_groundtruth,
                texts,
                value_list,
                k,
                extract_answer,
                judge_correct,
            )
            for k in CHOSEN_AGGR_METHODS
        }
        judge_results["c%"] = get_correct_proportion(
            problem_inst["question"],
            extracted_groundtruth,
            texts,
            extract_answer,
            judge_correct,
        )

        output_list = [
            {"path_idx": i, "text": txt, "value": v}
            for i, (txt, v) in enumerate(zip(texts, value_list))
        ]

        return judge_results, output_list

    def mcts_multi_search(
        args: "SearchArgs", problem, no_terminal_reward=True, tree_path=None
    ):
        env = task_module.Env(
            config={
                "max_actions": args.max_action,
                "max_length": args.max_length,
                "stop_str": "The answer is ",
                "generation_config": {
                    "max_new_tokens": 64,
                    "do_sample": True,
                    "temperature": args.temperature,
                    "top_p": 1.0,
                    "top_k": 100,
                    "return_dict_in_generate": True,
                    "output_scores": True,
                    "use_cache": True,
                },
            },
            math_problems=[
                {
                    "question": problem["question"],
                    "answer": extract_groundtruth(problem["answer"]),
                }
            ],
            llm_gen_fn=llm_gen_fn,
            tokenizer=tokenizer,
        )
        # llm_gen_fn=partial(llm_gen_with_logp_v1, model, tokenizer),
        cfg = {
            "num_simulations": args.num_simulations,
            "pb_c_base": 19652,
            "pb_c_init": args.pb_c_init,
            "root_dirichlet_alpha": 0.3,
            "root_noise_weight": 0.25,
            "no_terminal_reward": no_terminal_reward,
        }
        if tree_path and tree_path.exists():
            mcts = SearchTree.from_json(cfg, tree_path, reset_visit_info=True)
        else:
            mcts = SearchTree(cfg=cfg)

        if args.rollout_method == "mcts.rollout":
            assert args.max_token is not None and args.max_simulation is None
            output_list, _, _ = mcts.rollout(
                env,
                args.num_mcts_aggregation,
                policy_forward_value,
                max_num_simulation=args.max_simulation,
                max_token=args.max_token,
                return_tree=True,
            )
        elif args.rollout_method == "mcts.get_next_action":
            output_list = _mcts_rollout_v1(
                mcts,
                env,
                policy_forward_value,
                args.num_mcts_aggregation,
                args.reset_total_tree,
                sample=args.mcts_sample,
                clear_total_tree=args.clear_tree,
            )
            prompt = prompt_fn(problem["question"])
            texts = [o["text"] for o in output_list]
            if len(texts) > 0:
                value_list = policy_forward_value(
                    # add a .strip() in case mistakes happens when copy this line to other place
                    [prompt + txt.strip() + task_module.SEP for txt in texts]
                ).tolist()
            else:
                value_list = []
            for o, v in zip(output_list, value_list):
                o["value"] = v

        elif args.rollout_method == "mcts.rap":
            output_list = mcts.vanila_mcts(
                env,
                args.num_mcts_aggregation,
                policy_forward_value,
                args.select_by_prior,
            )
        elif args.rollout_method == "mcts.beam_search":
            output_list = mcts.beam_search(
                env, args.num_mcts_aggregation, args.max_length, policy_forward_value
            )
        elif args.rollout_method == "mcts.dfs":
            # here the num_mcts_aggregation is the step_limit which indicate how many nodes
            # will be visited in the tree.
            output_list = mcts.dfs(
                env,
                args.num_mcts_aggregation,
                policy_forward_value,
                prune_value=args.prune_value,
                prune_ratio=args.prune_ratio,
            )
        else:
            raise ValueError("Unknow rollout method: {}".format(args.rollout_method))

        texts = [o["text"] for o in output_list]
        value_list = [o["value"] for o in output_list]

        extracted_groundtruth = extract_groundtruth(problem["answer"])
        judge_results = {
            f"{k}@{args.num_mcts_aggregation}": judge_ans(
                problem["question"],
                extracted_groundtruth,
                texts,
                value_list,
                k,
                extract_answer,
                judge_correct,
            )
            for k in CHOSEN_AGGR_METHODS
        }
        judge_results["c%"] = get_correct_proportion(
            problem["question"],
            extracted_groundtruth,
            texts,
            extract_answer,
            judge_correct,
        )

        if output_list and args.rollout_method != "mcts.rollout":
            num_token = output_list[-1]["num_generated_token"]
        else:
            num_token = mcts.num_generated_token
        judge_results["#token"] = num_token
        return mcts, judge_results, output_list

    def test_problem(
        args,
        idx,
        problem_inst,
        cot_writer,
        cot_sc_writer,
        mcts_no_term_writer,
        mcts_w_term_writer,
    ):
        results = {}

        def save_fn(writer, output, result: Dict):
            if writer is not None:
                obj = {
                    "i": idx,
                    "question": problem_inst["question"],
                    "groundtruth": problem_inst["answer"],
                    "output": output,
                    "result": result,
                }
                writer.write(obj)

        if TEST_NO_TERMINAL:
            # save_tree_path = save_dir / f"tmp_tree"
            # if not save_tree_path.exists():
            #     save_tree_path.mkdir(parents=True)
            mcts, r_no_terminal, no_terminal_episodes = mcts_multi_search(
                args, problem_inst, True
            )
            # json.dump(
            #     get_root(mcts.root).to_json(),
            #     open(save_tree_path / f"tree_{idx}.json", "w"),
            #     indent=2,
            # )
            save_fn(mcts_no_term_writer, no_terminal_episodes, r_no_terminal)
            results["w/o-terminal"] = r_no_terminal

        if TEST_WITH_TERMINAL:
            _, r_with_terminal, with_terminal_episodes = mcts_multi_search(
                args, problem_inst, False
            )
            save_fn(mcts_w_term_writer, with_terminal_episodes, r_with_terminal)
            results["w/-terminal"] = r_with_terminal

        if TEST_COT_GREEDY:
            r_cot_greedy, cot_episodes = cot_direct_output(
                args,
                problem_inst,
                stop=tokenizer.eos_token_id,
                max_new_tokens=256,
                temperature=args.temperature,
                top_k=1,
            )
            save_fn(cot_writer, cot_episodes, r_cot_greedy)
            results["cot-greedy"] = r_cot_greedy

        if TEST_COT_SC:
            r_cot_sc, cot_sc_episodes = cot_sc_output(
                args,
                problem_inst,
                stop=tokenizer.eos_token_id,
                max_new_tokens=256,
                temperature=args.temperature,
                top_k=100,
            )
            save_fn(cot_sc_writer, cot_sc_episodes, r_cot_sc)
            results["cot-sc"] = r_cot_sc
        return results

    def _result_str(results, cnt, join_str="\n"):
        res = ""
        for k, v in results.items():
            if isinstance(v, int):
                res += f"{k}: {v/cnt:.2%}"
            elif isinstance(v, dict):
                res += f"{k}: "
                res += ", ".join(
                    [
                        (
                            f"{sub_k}: {sub_v/cnt:.2f}"
                            if sub_k == "#token"
                            else f"{sub_k}: {sub_v/cnt:.2%}"
                        )
                        for sub_k, sub_v in v.items()
                    ]
                )
            else:
                raise ValueError
            res += join_str
        res += f"cnt: {cnt}"
        return res

    for i_arg, cur_args in enumerate(args_list):
        args = SearchArgs(**cur_args)
        seed = args.seed
        setup_seed(seed)
        writer_dir = save_dir / (f"args{i_arg}_seed{seed}/")

        if local_rank == 0:
            print("Search args: {}, SEED={}".format(args, seed))
            if not writer_dir.exists():
                writer_dir.mkdir(parents=True)
            json.dump(cur_args, open(writer_dir / "args.json", "w"))

        if TEST_COT_GREEDY:
            cot_save_path = writer_dir / "cot"
            if local_rank == 0 and not cot_save_path.exists():
                cot_save_path.mkdir(parents=True)
            dist.barrier()
            cot_writer = jsonlines.open(cot_save_path / f"{local_rank}.jsonl", "a")
        else:
            cot_writer = None
        if TEST_COT_SC:
            cot_sc_save_path = writer_dir / "cot_sc"
            if local_rank == 0 and not cot_sc_save_path.exists():
                cot_sc_save_path.mkdir(parents=True)
            dist.barrier()
            cot_sc_writer = jsonlines.open(
                cot_sc_save_path / f"{local_rank}.jsonl", "a"
            )
        else:
            cot_sc_writer = None

        if TEST_NO_TERMINAL:
            mcts_no_term_save_path = writer_dir / "no_terminal_reward"
            if local_rank == 0 and not mcts_no_term_save_path.exists():
                mcts_no_term_save_path.mkdir(parents=True)
            dist.barrier()
            mcts_no_term_writer = jsonlines.open(
                mcts_no_term_save_path / f"{local_rank}.jsonl", "a"
            )
        else:
            mcts_no_term_writer = None

        if TEST_WITH_TERMINAL:
            mcts_w_term_save_path = writer_dir / "with_terminal_reward"
            if local_rank == 0 and not mcts_w_term_save_path.exists():
                mcts_w_term_save_path.mkdir(parents=True)
            dist.barrier()
            mcts_w_term_writer = jsonlines.open(
                mcts_w_term_save_path / f"{local_rank}.jsonl", "a"
            )
        else:
            mcts_w_term_writer = None

        cnt = 0
        correct_cnt_dict = dict()
        t0 = time.time()
        for i in (pbar := tqdm(range(len(test_ds)), disable=(local_rank != 0))):
            if i % world_size == local_rank:
                results = test_problem(
                    args,
                    i,
                    test_ds[i],
                    cot_writer,
                    cot_sc_writer,
                    mcts_no_term_writer,
                    mcts_w_term_writer,
                )
                for k, v in results.items():
                    if isinstance(v, int):
                        if k not in correct_cnt_dict:
                            correct_cnt_dict[k] = 0
                        correct_cnt_dict[k] += v
                    elif isinstance(v, dict):
                        if k not in correct_cnt_dict:
                            correct_cnt_dict[k] = dict()
                        for sub_k, sub_v in v.items():
                            if sub_k not in correct_cnt_dict[k]:
                                correct_cnt_dict[k][sub_k] = 0

                            correct_cnt_dict[k][sub_k] += sub_v
                cnt += 1
                results_strs = _result_str(correct_cnt_dict, cnt, join_str="; ")
                pbar.set_description(results_strs)

        print_with_rank(results_strs)

        cnt_list = gather_scalar(cnt, local_rank, world_size)

        gathered_results = {}
        for k, v in correct_cnt_dict.items():
            if isinstance(v, int):
                gathered_list = gather_scalar(int(v), local_rank, world_size)
                if local_rank == 0:
                    gathered_results[k] = sum(gathered_list)
            elif isinstance(v, dict):
                gathered_results[k] = {}
                for sub_k, sub_v in v.items():
                    gathered_list = gather_scalar(float(sub_v), local_rank, world_size)
                    if local_rank == 0:
                        gathered_results[k][sub_k] = sum(gathered_list)
            else:
                raise ValueError

        if local_rank == 0:
            total_cnt = sum(cnt_list)
            t1 = time.time()
            total_results_strs = _result_str(gathered_results, total_cnt)

            print(cur_args)
            print("TOTAL RESULTS:\n", total_results_strs)
            print("Time: {}".format(t1 - t0))
