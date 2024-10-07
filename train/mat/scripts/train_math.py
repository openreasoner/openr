#!/usr/bin/env python
import sys
import os
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
sys.path.append("../../")
from mat.config import get_config
from mat.envs.math.math_env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from mat.runner.shared.math_runner import MathRunner as Runner
from mat.envs.math.math_env import MathEnv

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = MathEnv(rank=rank, dataset_name=all_args.dataset_name, dataset_path=all_args.dataset_path, mode="train")
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = MathEnv(rank=rank, dataset_name=all_args.dataset_name, dataset_path=all_args.dataset_path, mode="test")
            env.seed(all_args.seed + rank * 5000)
            return env
        return init_env
    return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--dataset_name', type=str, default='prealgebra', help="Which dataset to test on.")
    parser.add_argument('--dataset_path', type=str, help="Path to the dataset file.")
    parser.add_argument('--model_name_or_path', type=str, help="Name of the agent model or path to the agent model checkpoint.")
    parser.add_argument('--prm_model_name_or_path', type=str, default='', help="Name of the model or path to the process reward model checkpoint.")
    parser.add_argument('--max_new_tokens', type=int, default=96, help="max_new_tokens")
    parser.add_argument('--vacab_size', type=int, default=151936)
    parser.add_argument('--gradient_cp_steps', type=int, default=2)
    all_args = parser.parse_known_args(args)[0]

    return all_args

def build_run_dir(all_args):
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/scripts/results") / all_args.experiment_name / all_args.dataset_name / all_args.algorithm_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                            str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    return run_dir

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    all_args.num_env_steps = 50000
    all_args.episode_length = 8
    all_args.n_rollout_threads = 4
    all_args.log_interval = 1
    all_args.critic_lr = 5e-5
    all_args.lr = 1e-6
    print("algorithm: {}, dataset_name: {}".format(all_args.algorithm_name, all_args.dataset_name))

    run_dir = build_run_dir(all_args)

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args)

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": envs.n_agents,
        "run_dir": run_dir
    }

    runner = Runner(config)
    runner.run()
    # runner.eval(0)

    runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
