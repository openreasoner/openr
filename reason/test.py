import sys
import os
from pathlib import Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from reason.inference.rm_call import RMRemoteCaller, RemoteRewardModelConfig
from reason.inference.lm_call import VLLMRemoteCaller, LMCallingConfig

image_path = "/home/siting/openr-mm/mathematics-0.png"
x = '<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>According to the following questions, which option in the image is correct? Give the answer from A, B, C and D.<|im_end|>\n<|im_start|>assistant\n'
print(x)

lm_call = VLLMRemoteCaller("Qwen2-VL-2B-Instruct")
y = lm_call(x, image_path=image_path, config=LMCallingConfig(n=1, temperature=0.0, max_new_tokens=2048, 
                                    #   stop_str=['\n\n'], 
                                      include_stop_str_in_output=True, is_multimodal=True))
print(y)

rm_call = RMRemoteCaller(config=RemoteRewardModelConfig(model_name="math-shepherd-mistral-7b-prm", controller_addr="http://0.0.0.0:28777", step_tag="ки\n", format_str="{question} {answer}"))
r = rm_call((x, y.text[0]), lm_step_tag="\n\n")
print(r)

import json

data = {
    "question": x,
    "image_path": image_path,
    "answer": y.text[0],
    "prm_value": r
}

# 将字典保存为 JSON 文件
with open("output.json", "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

import pdb; pdb.set_trace()