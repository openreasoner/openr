import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import os
from transformers import set_seed
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List

# Set your Hugging Face token here
os.environ["HUGGINGFACE_HUB_TOKEN"] = "hf_yourkey"

# For reproducibility
SEED = 1234
set_seed(SEED)
random.seed(42)

class LM:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct", model_type: str = "hf", num_rollouts: int = 5, **model_args):
        self.model_type = model_type.lower()
        self.model_name = model_name
        
        self.max_tokens = 200
        self.temperature_range = [0.7, 1.0]
        self.num_rollouts = num_rollouts
        self.__dict__.update(model_args)
        print("Updated model args:", self.__dict__)
        
        if self.model_type == "vllm":
            raise NotImplementedError("VLLM is not implemented yet")
            from vllm import LLM, SamplingParams
            self.llm = LLM(model=model_name, enable_prefix_caching=True)
            self.SamplingParams = SamplingParams
        elif self.model_type == "hf":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="cuda"
            )
        elif self.model_type == "openai":
            import openai
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.model_type == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        else:
            raise ValueError("Invalid model_type. Choose 'vllm', 'hf', 'openai', or 'anthropic'.")
        
    def generate(self, question, partial_answer, num_rollouts=None):
        if num_rollouts is None:
            num_rollouts = self.num_rollouts
        prompt = question + partial_answer
        if self.model_type == "vllm":
            return self.generate_vllm(prompt, num_rollouts)
        elif self.model_type == "hf":
            return self.generate_hf(prompt, num_rollouts)
        elif self.model_type == "anthropic" or self.model_type == "openai":
            return self.generate_api(prompt, num_rollouts)

    def generate_hf(self, prompt, num_rollouts):
        inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')
        results = []
        for _ in range(num_rollouts):
            temperature = random.uniform(self.temperature_range[0], self.temperature_range[1])
            outputs = self.model.generate(
                **inputs, do_sample=True, max_new_tokens=self.max_tokens, temperature=temperature,
                num_return_sequences=1
            )
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            results.append(result)
        return results

    def generate_vllm(self, prompt, num_rollouts):
        raise NotImplementedError("VLLM is not implemented yet")
        temperature = random.choice(self.temperature_range)
        sampling_params = self.SamplingParams(
            temperature=temperature,
            top_p=1,
            max_tokens=self.max_tokens,
            n=num_rollouts,
            seed=SEED
        )
        outputs = self.llm.generate(prompt, sampling_params)
        result = [completion.text for output in outputs for completion in output.outputs]
        return result

    def generate_api(self, prompt: str, num_rollouts) -> List[str]:
        def send_request(prompt):
            temperature = random.choice(self.temperature_range)
            if self.model_type == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=temperature
                )
                output = response.choices[0].message.content
            elif self.model_type == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=temperature
                )
                output = response.content[0].text
            return output

        responses = []
        with ThreadPoolExecutor(max_workers=num_rollouts) as executor:
            futures = [executor.submit(send_request, prompt) for _ in range(num_rollouts)]
            for future in tqdm(as_completed(futures), total=len(futures)):
                responses.append(future.result())

        return responses
