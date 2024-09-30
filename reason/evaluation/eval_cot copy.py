from pathlib import Path
from typing import Dict, List, Optional, Union
import torch.distributed as dist
from config.config_utils import str2bool
from distributed.utils import (
    print_rank_0,
    print_with_rank,
    init_distributed,
    gather_scalar,
)
from envs import get_env_datasets, get_default_query_str_builder
from reason.inference.trajectory_collector import _mcts_rollout_v1
from reason.inference.text_generation import llm_gen_with_logp_fastchat_vllm
from reason.inference.value import _value_inference_fastchat
from reason.mcts.tree import SearchTree
from reason.reranking.vote_utils import (
    AGG_FN_MAP,
    MAJORITY_VOTE,
    ORM_VOTE,
    ORM_MAX,
    PRM_MIN_VOTE,
    PRM_MIN_MAX,
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
from multiprocessing import Pool
import tree


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


CHOSEN_AGGR_METHODS = [MAJORITY_VOTE, PRM_MIN_MAX, PRM_MIN_VOTE]


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

    if "orm" in aggration_mode:
        # score_normalization: this is only necessary for [-1, 1] values
        valid_v_list = np.array(valid_v_list)
        valid_v_list -= valid_v_list.min()
        valid_v_list /= valid_v_list.max() + 1e-3
        valid_v_list = valid_v_list.tolist()
    aggregated_ans = AGG_FN_MAP[aggration_mode](valid_ans_list, valid_v_list)

    return (
        1 if judge_correct_fn(problem_str, extracted_groundtruth, aggregated_ans) else 0
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="CoT")
    parser.add_argument("--LM", type=str, required=True)
    parser.add_argument("--RM", type=str, required=True)
    # parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--env_name", type=str, default="gsm8k")
    parser.add_argument("--test", type=str2bool, default=True)
    parser.add_argument("--is_few_shot", type=str2bool, default=False)
    parser.add_argument("--controller_addr", type=str, default="http://0.0.0.0:28778")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_sequence", type=int, default=1)
    config = parser.parse_args()

    setup_seed(config.seed)

    task_module = importlib.import_module(f"envs.{config.env_name}")
    extract_answer = task_module.extract_answer
    extract_groundtruth = task_module.extract_groundtruth
    judge_correct = task_module.judge_correct

    # save_dir = Path(config.save_dir) / config.env_name
    train_ds, test_ds = get_env_datasets(config.env_name)
    if not config.test:
        test_ds = train_ds

    llm_gen_fn = partial(
        llm_gen_with_logp_fastchat_vllm,
        config.LM,
        controller_addr=config.controller_addr,
    )

    def prompt_fn(problem_input: str):
        return get_default_query_str_builder(config.env_name)(
            problem_input, is_few_shot=config.is_few_shot
        )

    # # fake value
    # def rm_call(x):
    #     return np.array([0.0] * len(x)).tolist()

    def rm_call(input_str: Union[str, List[str]]) -> Union[List[List[int]], List[int]]:
        return _value_inference_fastchat(config.RM, input_str, config.controller_addr)

    def cot_direct_output(problem_inst, **kwargs):
        prompt = prompt_fn(problem_inst["question"])
        max_new_tokens = kwargs.pop("max_new_tokens", 256)
        max_new_tokens = max(256, max_new_tokens)
        gen_result = llm_gen_fn(
            prompt=prompt,
            num_sequence=config.num_sequence,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        return gen_result

    def analyze_output(problem_inst, gen_result):
        texts = gen_result.text
        extracted_groundtruth = extract_groundtruth(problem_inst["answer"])
        prompt = prompt_fn(problem_inst["question"])
        if len(texts) > 1:
            value_list = rm_call([prompt + txt + task_module.SEP for txt in texts])
        else:
            value_list = [[0]]
        output_list = [
            {"path_idx": i, "text": txt, "value": v}
            for i, (txt, v) in enumerate(zip(texts, value_list))
        ]
        res = {
            agg_method: judge_ans(
                problem_inst["question"],
                extracted_groundtruth,
                texts,
                value_list,
                agg_method,
                extract_answer,
                judge_correct,
            )
            for agg_method in (
                CHOSEN_AGGR_METHODS if config.num_sequence > 1 else [MAJORITY_VOTE]
            )
        }
        return res, output_list

    def save_fn(writer, idx, problem_inst, result: Dict):
        if writer is not None:
            obj = {
                "i": idx,
                "question": problem_inst["question"],
                "groundtruth": problem_inst["answer"],
                "result": result,
            }
            writer.write(obj)

    def fn(problem_inst):
        gen_result = cot_direct_output(
            problem_inst,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
        )
        r_cot, cot_episodes = analyze_output(problem_inst, gen_result)
        return {"res": r_cot, "episodes": cot_episodes}

    # test_ds = [test_ds[i] for i in range(10)]
    results = Pool(32).imap(fn, test_ds)
    results = list(tqdm(results, total=len(test_ds)))

    print(
        "Experiment:",
        config.exp_name,
        tree.map_structure(lambda *xs: np.mean(xs), *[r["res"] for r in results]),
    )
    print(config)

    with jsonlines.open("tmp_results.jsonl", "w") as writer:
        for i, (problem_inst, result) in enumerate(zip(test_ds, results)):
            save_fn(writer, i, problem_inst, result)
