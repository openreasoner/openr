from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
from torch import nn
import numpy as np
from mat.envs.math.prompts import IN_CONTEXT_EXAMPLE

class ProcessRM(nn.Module):

    def __init__(self, model_name_or_path):
        super().__init__()
        self.good_token = '+'
        self.bad_token = '-'
        self.step_tag = 'ки'

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.pad_token_id = 0
        self.candidate_tokens = self.tokenizer.encode(f"{self.good_token} {self.bad_token}")[1:] # [648, 387]
        self.step_tag_id = self.tokenizer.encode(f"{self.step_tag}")[-1] # 12902
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto",).eval()
        print(f"successfully loaded PRM.")
        
        # from mat.envs.math.prompts import IN_CONTEXT_EXAMPLE
        # input_id = torch.tensor([self.tokenizer.encode(IN_CONTEXT_EXAMPLE)]).to("cuda")
        # print("input_id: ", input_id)
        
        # print("candidate_tokens: ", self.candidate_tokens)
        # print("step_tag_id: ", self.step_tag_id)
        # with torch.no_grad():
        #     logits = self.model(input_id).logits[:,:,self.candidate_tokens]
        #     print("logits: ", logits)
        #     scores = logits.softmax(dim=-1)[:,:,0] 
        #     print("scores: ", scores)
        #     step_scores = scores[input_id == self.step_tag_id]
        #     print("step_scores: ", step_scores)
        # exit()
            

    @torch.no_grad()
    def get_reward(self, obs: list[np.ndarray[str]], actions: list[np.ndarray[str]]):
        inputs_for_prm = []
        for o, a in zip(obs, actions):
            o = o[0].replace(IN_CONTEXT_EXAMPLE, "")
            a = a[0]
            inputs_for_prm.append(f"{o} {a} {self.step_tag}")
        # inputs_for_prm = [f"{o.replace(IN_CONTEXT_EXAMPLE, "")} {a} {self.step_tag}" for o, a in zip(obs, actions)]
        # input_ids = [torch.tensor([self.tokenizer.encode(ip)]).to("cuda") for ip in inputs_for_prm]
        input_ids = self.tokenizer(inputs_for_prm, return_tensors="pt", padding=True).to("cuda")
        logits = self.model(**input_ids).logits[:, :, self.candidate_tokens]
        score = logits.softmax(dim=-1)[:, :, 0]
        
        step_scores = []
        for i in range(np.shape(score)[0]):
            step_score = score[i][input_ids["input_ids"][i] == self.step_tag_id]
            last_step_score = step_score[-1]
            step_scores.append([last_step_score.item()])
        step_scores = np.array(step_scores)

        # step_scores = []
        # with torch.no_grad():
        #     for input_id in input_ids:
        #         logits = self.model(input_id).logits[:, :, self.candidate_tokens]
        #         score = logits.softmax(dim=-1)[:, :, 0]
        #         step_score = score[input_id == self.step_tag_id][-1]
        #         step_scores.append(step_score)
        #     step_scores = np.array([[score.item()] for score in step_scores])
        return step_scores
