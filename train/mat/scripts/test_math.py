#!/usr/bin/env python
import sys
import os
import socket
import setproctitle
import numpy as np
import random
from pathlib import Path
import torch
sys.path.append("../../")
from mat.config import get_config
from mat.envs.math.math_env import MathEnv
from mat.agents.qwen_lora_agent import QwenLoRAgent


def parse_args(args, parser):
    parser.add_argument('--dataset_name', type=str, default='prealgebra', help="Which dataset to test on.")
    parser.add_argument('--dataset_path', type=str, help="Path to the dataset file.")
    parser.add_argument('--model_name_or_path', type=str, help="Name of the agent model or path to the agent model checkpoint.")
    parser.add_argument('--model_peft_path', type=str, default='', help="Which model to uese")
    parser.add_argument('--max_new_tokens', type=int, default=96, help="max_new_tokens")
    parser.add_argument('--vacab_size', type=int, default=151936)
    all_args = parser.parse_known_args(args)[0]

    return all_args

@torch.no_grad()
def eval(agent, eval_env, episodes):
    eval_episode = 0

    eval_obs = eval_env.reset()
    
    while True:
        eval_actions = agent.test_get_actions(eval_obs)
        eval_obs, eval_rewards, eval_dones, eval_infos = eval_env.step(eval_actions)

        if eval_dones[0]:
            eval_obs = eval_env.reset()
            eval_episode += 1
            
        if eval_episode >= episodes:
            break


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # seed
    random.seed(all_args.seed)
    np.random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)  # If you're using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    eval_env = MathEnv(rank=0, dataset_name=all_args.dataset_name, dataset_path=all_args.dataset_path, mode="test")

    agent = QwenLoRAgent(all_args.model_name_or_path, all_args.max_new_tokens, "APPO", all_args.model_peft_path)
    
    eval(agent, eval_env, 500)

if __name__ == "__main__":
    main(sys.argv[1:])