from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from config.config_utils import str2bool
from reason.inference.lm_call import LMCallingConfig, VLLMRemoteCaller
from reason.inference.rm_call import (
    RMRemoteCaller,
    DummyRewardModelCaller,
    RewardModelBaseConfig,
    RemoteRewardModelConfig,
)
from reason.evaluation.evaluator import SolutionOutput, Task, RemoteMathEvaluator
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


def parallel_evaluate_test_dataset(
    method_name: str, solver_fn: Callable, save_dir: Optional[Path] = None
) -> List[Dict[str, Any]]:
    if save_dir is not None:
        record_writer = jsonlines.open(save_dir / f"record.jsonl", mode="w")
    else:
        record_writer = None

    test_ds = task.test_ds
    # test_ds = [test_ds[i] for i in range(32)]

    results = []
    if config.resume_dir is not None:
        answered_questions = set()
        with jsonlines.open(Path(config.resume_dir) / "record.jsonl", "r") as reader:
            cnt = 0
            for obj in reader:
                results.append(obj["result"])
                answered_questions.add(obj["question"])
                if record_writer is not None:
                    record_writer.write(obj)
                    cnt += 1
        print(f"Resumed {cnt} questions from {config.resume_dir}")
        total_cnt = len(test_ds)
        test_ds = [
            problem_inst
            for problem_inst in test_ds
            if problem_inst["question"] not in answered_questions
        ]
        new_cnt = len(test_ds)
        print(
            f"After resuming, there are {new_cnt}/{total_cnt} new questions to answer."
        )

    actor_pool = ActorPool(
        [
            RemoteMathEvaluator.remote(config.task_name, llm_gen_fn, rm_call)
            for _ in range(config.num_worker)
        ]
    )
    res_q = actor_pool.map_unordered(
        lambda p, x: p.evaluate_problem.remote(x, solver_fn), test_ds
    )  # Distributes tasks from the test_ds dataset across the worker pool asynchronously and
    # collects results in any order as they complete. Every worker has a new searching tree as we reset the
    # tree in solver_fn
    for i, (problem_inst, result, output) in enumerate(tqdm(res_q, total=len(test_ds))):
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--LM", type=str, required=True)
    parser.add_argument(
        "--LM_addr",
        type=str,
        default="http://0.0.0.0:28778",
        desc="locate which LM we use",
    )
    parser.add_argument(
        "--LM_config", type=str, default="reason/resource/qwen2.5/config.json"
    )
    parser.add_argument("--RM", type=str, default="dummy")
    parser.add_argument(
        "--RM_addr",
        type=str,
        default="http://0.0.0.0:28778",
        desc="locate which RM we use",
    )
    parser.add_argument("--RM_config", type=str, default="reason/resource/mistral")
    # task config
    parser.add_argument("--task_name", type=str, default="gsm8k")
    parser.add_argument("--test", type=str2bool, default=True)
    parser.add_argument("--is_few_shot", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    # method config
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--num_sequence", type=int, default=1)
    # LM gen config
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--stop_str", type=str, nargs="+", desc="customized stopping str for LM"
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="whether to use lora adapter for generation",
    )
    # Tree construction config
    parser.add_argument("--tree_max_depth", type=int, default=None)
    parser.add_argument("--tree_max_width", type=int, default=None)
    # ckpg config
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--resume_dir", type=str, default=None)
    # parallel config
    parser.add_argument("--local", action="store_true", default=False)
    parser.add_argument("--num_worker", type=int, default=32)
    config = parser.parse_args()

    setup_seed(config.seed)
    if config.local:
        print("run in pure local mode for debug only")
        config.num_worker = 1
        ray.init(local_mode=True)

    # load necessary config file
    with open(config.LM_config, "r", encoding="utf-8") as file:
        lm_cfg = json.load(file)
    with open(config.RM_config, "r", encoding="utf-8") as file:
        rm_cfg = json.load(file)

    prm_step_tag = rm_cfg.prm_step_tag
    prm_format_str = rm_cfg.problem_format_str
    lm_step_tag = lm_cfg.step_str

    llm_gen_fn = VLLMRemoteCaller(config.LM, config.LM_addr, lm_step_tag=lm_step_tag)
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
        )
        rm_call = RMRemoteCaller(rm_config)

    task = Task(
        task_name=config.task_name,
        is_few_shot=config.is_few_shot,
        cot_task_desc=lm_cfg.cot_task_desc,
        cot_examples=lm_cfg.cot_examples,
        problem_format_str=lm_cfg.problem_format_str,
    )

    solver_fns = {"cot": cot, "best_of_n": best_of_n}

    cfg_dict_record = dict()
    # XXX: qwen-2.5 requires add more stop words
    # not do it now.
    # stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    gen_config = LMCallingConfig(
        n=config.num_sequence,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        max_new_tokens=config.max_new_tokens,
        stop_str=config.stop_str,
        use_lora=config.use_lora,
    )
    cfg_dict_record["gen_config"] = gen_config.__dict__

    if config.method == "cot":
        method_config = CoTConfig(config.task_name)
        solver_fn = partial(cot, method_config, gen_config)
    elif config.method == "best_of_n":
        method_config = BestOfNConfig(
            config.task_name, num_sequence=config.num_sequence
        )
        solver_fn = partial(best_of_n, method_config, gen_config)
    elif config.method == "beam_search":
        method_config = BeamSearchConfig(
            task_name=config.task_name,
            tree_max_depth=config.tree_max_depth,
            tree_max_width=config.tree_max_width,
            beam_size=config.num_sequence,
        )
        solver_fn = partial(beam_search, method_config, gen_config)
    elif config.method == "vanila_mcts":
        method_config = VanilaMCTSConfig(
            task_name=config.task_name,
            tree_max_depth=config.tree_max_depth,
            tree_max_width=config.tree_max_width,
            select_by_prior=False,
            num_path=config.num_sequence,
        )
        solver_fn = partial(vanila_mcts, method_config, gen_config)
    elif config.method == "rstar_mcts":
        method_config = VanilaMCTSConfig(
            task_name=config.task_name,
            tree_max_depth=config.tree_max_depth,
            tree_max_width=config.tree_max_width,
            select_by_prior=False,
            num_path=config.num_sequence,
        )
        solver_fn = partial(rstar_mcts, method_config, gen_config)

    else:
        raise ValueError(f"Unknown method: {config.method}")
    cfg_dict_record["method"] = config.method
    cfg_dict_record["method_config"] = method_config.__dict__

    if config.save_dir is not None:
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(config.save_dir) / task.task_name / config.method / datetime_str
        save_dir.mkdir(parents=True)
        record_writer = jsonlines.open(save_dir / f"record.jsonl", mode="w")
        cfg_dict_record["LM"] = config.LM
        cfg_dict_record["RM"] = config.RM
        json.dump(cfg_dict_record, open(save_dir / "config.json", "w"))
    else:
        save_dir = None

    parallel_evaluate_test_dataset(config.method, solver_fn, save_dir)
