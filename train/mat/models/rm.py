from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
from torch import nn
import numpy as np

class ProcessRM(nn.Module):

    def __init__(self, model_name_or_path):
        super().__init__()
        self.good_token = '+'
        self.bad_token = '-'
        self.step_tag = 'ки'

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.candidate_tokens = self.tokenizer.encode(f"{self.good_token} {self.bad_token}")[1:] # [648, 387]
        self.step_tag_id = self.tokenizer.encode(f"{self.step_tag}")[-1] # 12902
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).eval()
        print(f"successfully loaded PRM.")

    def get_reward(self, obs: list[np.ndarray[str]], actions: list[np.ndarray[str]]):
        inputs_for_prm = [f"{o} {a} {self.step_tag}" for o, a in zip(obs, actions)]
        input_ids = [torch.tensor([self.tokenizer.encode(ip)]) for ip in inputs_for_prm]

        step_scores = []
        with torch.no_grad():
            for input_id in input_ids:
                logits = self.model(input_id).logits[:, :, self.candidate_tokens]
                score = logits.softmax(dim=-1)[:, :, 0]
                step_score = score[input_id == self.step_tag_id][-1]
                step_scores.append(step_score)
            step_scores = np.array([[score.item()] for score in step_scores])
        return step_scores
