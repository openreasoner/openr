import numpy as np
import json
import jsonlines
import random
from copy import deepcopy
from mat.envs.math.prompts import IN_CONTEXT_EXAMPLE

# training data with mode="train" and testing data with mode="test"
def load_dataset(dataset_path, mode):
    if "jsonl" in dataset_path:
        with jsonlines.open(dataset_path) as reader:
            dataset = [line for line in reader]
    else:
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
    return dataset

class MathEnv:

    def __init__(self, rank, dataset_name, dataset_path, mode):
        
        self.rank = rank
        self.mode = mode
        self.dataset_name = dataset_name
        self.dataset = load_dataset(dataset_path=dataset_path, mode=mode)
        self.n_agents = 1
        self.max_step = 10
        self.step_count = 0
        
        self.problem = None
        self.label = None
        self.step_tag = "ки"
        self.current_state = None

    def reset(self):
        problem_answer_pair = random.choice(self.dataset)
        # problem_answer_pair = self.dataset[3]
        self.problem = problem_answer_pair["problem"]
        self.label = problem_answer_pair["final_answer"]
        self.current_state = IN_CONTEXT_EXAMPLE + self.problem + "\n"
        obs = np.array([self.current_state], dtype=np.object_)
        self.step_count = 0
        return obs
    
    def step(self, action):
        self.step_count += 1
        action = action[0]
        action = action.replace(self.step_tag, "").strip()
        self.current_state = self.current_state + action + " " + self.step_tag + "\n"
        # self.current_state = self.current_state + action.strip() + "\n"
        next_obs = np.array([self.current_state], dtype=np.object_)
        
        score = 0.0
        if "step" in action.lower() or "answer" in action.lower():
            score = 1.0
        if "answer" in action.lower():
            dones = np.ones((self.n_agents), dtype=bool)
        elif self.step_count >= self.max_step:
            dones = np.ones((self.n_agents), dtype=bool)
        else:
            dones = np.zeros((self.n_agents), dtype=bool)
        
        rewards = [score for _ in range(self.n_agents)]
        infos = {"state": self.current_state}
        return next_obs, rewards, dones, infos

    def seed(self, seed):
        np.random.seed(seed)