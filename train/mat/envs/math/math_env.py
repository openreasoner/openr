import numpy as np
import json
import random
from copy import deepcopy

# training data with mode="train" and testing data with mode="test"
def load_dataset(dataset_path, mode):
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
        self.problem = problem_answer_pair["problem"]
        self.label = problem_answer_pair["final_answer"]
        self.current_state = self.problem
        obs = np.array([self.problem], dtype=np.object_)
        self.step_count = 0
        return obs
    
    def step(self, action):
        self.step_count += 1
        action = action[0]
        self.current_state = self.current_state + "\n" + action + self.step_tag
        next_obs = np.array([self.current_state], dtype=np.object_)
        
        score = 0.0
        if "answer" in action.lower() or self.label in action.lower():
            dones = np.ones((self.n_agents), dtype=bool)
            if self.label in action.lower():
                score = 1.0
        elif self.step_count >= self.max_step:
            dones = np.ones((self.n_agents), dtype=bool)
        else:
            dones = np.zeros((self.n_agents), dtype=bool)
        
        rewards = [score for _ in range(self.n_agents)]
        infos = {"state": self.current_state}
        return next_obs, rewards, dones, infos

    def seed(self, seed):
        np.random.seed(seed)