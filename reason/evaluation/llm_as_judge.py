from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from torch.optim.optimizer import required
from collections import defaultdict

from config.config_utils import str2bool
from reason.inference.lm_call import LMCallingConfig, VLLMRemoteCaller
from reason.inference.rm_call import (
    RMRemoteCaller,
    DummyRewardModelCaller,
    RewardModelBaseConfig,
    RemoteRewardModelConfig,
)
from reason.evaluation.evaluator import (
    SolutionOutput,
    Task,
    RemoteMathEvaluator,
    RemoteLLMEvaluator,
    CHOSEN_AGGR_METHODS,
)
from reason.reranking.vote_utils import AGG_FN_MAP
import torch
from functools import partial
import json
import jsonlines
import time
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import os
import importlib
import random
from multiprocessing import Pool
import tree
from ray.util.actor_pool import ActorPool
from reason.evaluation.methods import *
import ray

from reason.evaluation.utils import setup_seed, load_record_data


def LLM_as_Judge(cfg, llm_gen_fn, gen_cfg, datasets, save_dir, num_seq):
    """
    cfg: global config
    llm_gen_fn: cot calling function
    gen_cfg: config for reward generation,
    dataset: List[Dict], each dict contains question, groundtruth, question_idx, path_idx, value, gen_answer
    num_seq: int, number of searching sequences
    """
    task_answer_extractor = importlib.import_module(
        f"envs.{cfg.task_name}"
    ).extract_answer
    record_writer = jsonlines.open(
        os.path.join(save_dir, f"reviewed_record_{cfg.LM}.jsonl"), mode="w"
    )
    actor_pool = ActorPool(
        [RemoteLLMEvaluator.remote(llm_gen_fn, gen_cfg) for _ in range(cfg.num_worker)]
    )
    res_q = actor_pool.map_unordered(
        lambda p, x: p.evaluate_problem.remote(x), datasets
    )

    all_collection = defaultdict(list)
    for i, (problem_inst, score, review_text) in enumerate(
        tqdm(res_q, total=len(datasets))
    ):
        obj = {
            "question_idx": problem_inst["question_idx"],
            "path_idx": problem_inst["path_idx"],
            "answer": problem_inst["gen_answer"],
            "value": problem_inst["value"],
            "score": score,
            "review_text": review_text,
        }
        all_collection[problem_inst["question_idx"]].append(obj)

    record_writer.write(all_collection)

    # compute aggregation
    score_board = {}
    normalize = False
    for agg_method in CHOSEN_AGGR_METHODS if num_seq > 1 else [CHOSEN_AGGR_METHODS[0]]:
        total_score = []
        for p in all_collection.values():
            ans_list = [task_answer_extractor(i["answer"]) for i in p]
            value_list = [i["value"] for i in p]
            score_list = [i["score"] for i in p]

            if "orm" in agg_method and normalize:
                valid_v_list = np.array(value_list)
                valid_v_list -= valid_v_list.min()
                valid_v_list /= valid_v_list.max() + 1e-3
                valid_v_list = valid_v_list.to_list()
            else:
                valid_v_list = value_list

            aggregated_ans = AGG_FN_MAP[agg_method](ans_list, valid_v_list)
            aggregated_score = score_list[ans_list.index(aggregated_ans)]
            total_score.append(aggregated_score)

        score_board[agg_method] = total_score

    avg_res = {k: sum(v) / len(v) for k, v in score_board.items()}
    print(f"Avg = {avg_res}")
    json.dump(
        avg_res,
        open(os.path.join(save_dir, f"reviewed_avg_results_{cfg.LM}.json"), "w"),
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--LM", type=str, required=True, help="LM name in the service")
    parser.add_argument(
        "--LM_addr",
        type=str,
        default="http://0.0.0.0:28778",
        help="LM address in the service",
    )
    parser.add_argument("--data_path", type=str, required=True, help="Data path")
    parser.add_argument("--task_name", type=str, required=True, help="Name of the task")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--stop_str", type=str, default=None)
    parser.add_argument(
        "--num_worker", type=int, default=32, help="Number of parallel processes"
    )
    parser.add_argument("--local", action="store_true", default=False)

    config = parser.parse_args()
    setup_seed(config.seed)
    if config.local:
        print("Run in pure local mode for debug onlu")
        config.num_worker = 1
        ray.init(local_mode=True)

    llm_gen_fn = VLLMRemoteCaller(config.LM, config.max_new_tokens)
    gen_config = LMCallingConfig(
        max_new_tokens=config.max_new_tokens, stop_str=config.stop_str
    )

    # data loading
    record_data, n_seq = load_record_data(config.data_path)

    LLM_as_Judge(
        config,
        llm_gen_fn,
        gen_config,
        record_data,
        os.path.dirname(config.data_path),
        n_seq,
    )
