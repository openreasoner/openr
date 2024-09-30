from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from config.config_utils import str2bool
from reason.inference.lm_call import LMCallingConfig, VLLMRemoteCaller
from reason.inference.rm_call import RMRemoteCaller
from reason.evaluation.evaluator import SolutionOutput, Task, MathEvaluator
import torch
from functools import partial
import json
import jsonlines
import time
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import os
import random
from multiprocessing import Pool
import tree
from ray.util.actor_pool import ActorPool
from reason.evaluation.methods import *
import ray


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--LM", type=str, required=True)
    parser.add_argument("--RM", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--task_name", type=str, default="gsm8k")
    parser.add_argument("--test", type=str2bool, default=True)
    parser.add_argument("--is_few_shot", type=str2bool, default=False)
    parser.add_argument("--controller_addr", type=str, default="http://0.0.0.0:28778")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_sequence", type=int, default=1)
    parser.add_argument("--num_worker", type=int, default=32)
    config = parser.parse_args()

    setup_seed(config.seed)
    # ray.init(local_mode=True)

    llm_gen_fn = VLLMRemoteCaller(config.LM, config.controller_addr)
    rm_call = RMRemoteCaller(config.RM, config.controller_addr)

    task = Task(task_name=config.task_name, is_few_shot=config.is_few_shot)

    def parallel_evaluate_test_dataset(
        method_name: str, solver_fn: Callable
    ) -> List[Dict[str, Any]]:
        if config.save_dir is not None:
            datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = (
                Path(config.save_dir) / task.task_name / method_name / datetime_str
            )
            save_dir.mkdir(parents=True)
            record_writer = jsonlines.open(save_dir / f"record.jsonl", mode="w")
        else:
            record_writer = None

        results = []
        actor_pool = ActorPool(
            [
                MathEvaluator.remote(config.task_name, llm_gen_fn, rm_call)
                for _ in range(config.num_worker)
            ]
        )
        test_ds = task.test_ds
        # test_ds = [test_ds[i] for i in range(32)]
        res_q = actor_pool.map_unordered(
            lambda p, x: p.evaluate_problem.remote(x, solver_fn), test_ds
        )
        for i, (problem_inst, result, output) in enumerate(
            tqdm(res_q, total=len(test_ds))
        ):
            results.append(result)
            if record_writer:
                obj = {
                    # "i": i,
                    "question": problem_inst["question"],
                    "groundtruth": problem_inst["answer"],
                    "result": result,
                    "output": output,
                }
                record_writer.write(obj)
        avg_res = (tree.map_structure(lambda *xs: np.mean(xs), *results),)
        if record_writer:
            json.dump(avg_res, open(save_dir / "avg_result.json", "w"))
        print("Method: {}. Average result: {}".format(method_name, avg_res))
        return results

    solver_fns = {"cot": cot, "best_of_n": best_of_n}

    gen_config = LMCallingConfig(
        n=config.num_sequence,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        max_new_tokens=config.max_new_tokens,
    )
    if config.method == "cot":
        cot_config = CoTConfig(config.task_name)
        solver_fn = partial(cot, cot_config, gen_config)
    elif config.method == "best_of_n":
        best_of_config = BestOfNConfig(
            config.task_name, num_sequence=config.num_sequence
        )
        solver_fn = partial(best_of_n, best_of_config, gen_config)
    elif config.method == "beam_search":
        beam_search_config = BeamSearchConfig(
            task_name=config.task_name,
            tree_max_length=10,
            tree_max_width=10,
            beam_size=2,
        )
        solver_fn = partial(beam_search, beam_search_config, gen_config)
    else:
        raise ValueError(f"Unknown method: {config.method}")

    parallel_evaluate_test_dataset(config.method, solver_fn)
