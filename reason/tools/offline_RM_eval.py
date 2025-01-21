"""
Evaluate record.jsonl file on the given RM
"""

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

BASE_DIR = Path(__file__).parent.parent.parent.resolve()

class RMEvaluator:
    def __init__(self, rm_call: RewardModelCallingFunction, lm_step_tag: str):
        self.rm_call = rm_call
        self.lm_step_tag = lm_step_tag

    def evaluate_problem(self, problem_inst):
        question = problem_inst['question']
        gen_answers = [i['text'] for i in problem_inst['output']]
        input_list = [(question, a) for a in gen_answers]
        value_list = self.rm_call(input_list, lm_step_tag=self.lm_step_tag)

        output_list = [
            {"path_idx": i, "text": txt, "value": v}
            for i, (txt, v) in enumerate(zip(gen_answers, value_list))
        ]
        return problem_inst, output_list

@ray.remote
class RemoteRMEvaluator(RMEvaluator):
    def __init__(self, rm_call, lm_step_tag):
        super().__init__(rm_call, lm_step_tag)


def RM_evaluate(cfg, rm_fn, lm_step_tag, datasets, save_dir):
    if cfg.save:
        record_writer = jsonlines.open(os.path.join(save_dir, f"record_{cfg.RM}.jsonl"), mode="w")

    actor_pool = ActorPool(
        [
            RemoteRMEvaluator.remote(rm_fn, lm_step_tag) for _ in range(cfg.num_worker)
        ]
    )
    res_q = actor_pool.map_unordered(
        lambda p, x: p.evaluate_problem.remote(x), datasets
    )

    for i, (problem_inst, output) in enumerate(tqdm(res_q, total=len(datasets))):
        if cfg.save:
            obj = {
                "question": problem_inst["question"],
                "groundtruth": problem_inst['groundtruth'],
                "category": problem_inst['category'] if "category" in problem_inst else "",
                "result": {},
                "output": output,
            }

            record_writer.write(obj)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--RM", type=str, required=True)
    parser.add_argument("--RM_config", type=str, default="reason/resource/mistral/shepherd_prm_config.json")
    parser.add_argument("--RM_addr", type=str, default="http://0.0.0.0:28777")
    parser.add_argument("--data", type=str, required=True, help="record data dir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_worker", type=str, default=32)
    parser.add_argument("--LM_config", type=str,
                        default="reason/resource/qwen2.5/config.json",
                        help="LM name of the generated text")
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--local", action="store_true", default=False)

    config = parser.parse_args()

    with open(os.path.join(BASE_DIR, config.LM_config), "r", encoding="utf-8") as file:
        lm_cfg = json.load(file)
    with open(os.path.join(BASE_DIR, config.RM_config), "r", encoding="utf-8") as file:
        rm_cfg = json.load(file)

    lm_step_tag = lm_cfg["lm_step_tag"]
    lm_stop_tag = lm_cfg["lm_stop_tag"]
    prm_step_tag = rm_cfg["prm_step_tag"]
    prm_format_str = rm_cfg["problem_format_str"]

    if config.local:
        print("run in pure local mode for debug only")
        config.num_worker = 1
        ray.init(local_mode=True)

    # RM
    if config.RM == "dummy":
        rm_config = RewardModelBaseConfig(
            step_tag=prm_step_tag, format_str=prm_format_str
        )
        rm_call = DummyRewardModelCaller(rm_config)
    else:
        rm_config = RemoteRewardModelConfig(
            step_tag=prm_step_tag,
            format_str=prm_format_str,
            model_name=config.RM,
            controller_addr=config.RM_addr,
            lm_stop_str=lm_stop_tag,
        )
        rm_call = RMRemoteCaller(rm_config)

    # load data
    record_data = []
    with jsonlines.open(config.data, mode="r") as reader:
        for obj in reader:
            record_data.append(obj)

    # RM evaluation





