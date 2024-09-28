import numpy as np
from copy import deepcopy


# training data with mode="train" and testing data with mode="test"
def load_dataset(dataset_name, mode):
    pass


class MathEnv:

    def __init__(self, rank, dataset_name, mode):

        self.rank = rank
        self.mode = mode
        self.dataset_name = dataset_name
        self.dataset = load_dataset(dataset_name=dataset_name, mode=mode)
        self.n_agents = 1
        self.max_step = 10
        self.step_count = 0

        self.problem = None
        self.label = None
        self.current_state = None

    def reset(self):
        # the problem and label should be a random math problem from dataset.
        self.problem = "---\n1+1=2\n---2+2=4\n---3+3=6\n---4+4=8\n---5+5=10\n---6+6="
        self.label = "12"

        self.current_state = self.problem
        obs = np.array([self.problem], dtype=np.object_)
        self.step_count = 0
        return obs

    def step(self, action):
        self.step_count += 1
        action = action[0]
        self.current_state = self.current_state + "\n" + action
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
        # infos = [{"status": status, "std": std} for _ in range(self.n_agents)]
        infos = {"state": self.current_state}
        return next_obs, rewards, dones, infos

    def seed(self, seed):
        np.random.seed(seed)
