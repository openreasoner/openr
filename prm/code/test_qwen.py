from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from datasets import load_dataset
import argparse
import os

from peft import PeftModel,PeftConfig
from peft import get_peft_model, LoraConfig, TaskType
# Ensure bitsandbytes is available for 8-bit quantization
# import bitsandbytes as bnb
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

from torch.nn import BCEWithLogitsLoss
from transformers import DataCollatorWithPadding
from datasets import concatenate_datasets

parser = argparse.ArgumentParser()
parser.add_argument("--per_device_train_batch_size", type=int, default=2)
parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
parser.add_argument("--total_batch_size", type=int, default=256)
parser.add_argument("--learning_rate", type=int, default=1e-4)

args = parser.parse_args()


good_token = '+'
bad_token = '-'
step_tag = '\n\n\n\n\n' #ки
step_tag2 = '\n\n'

model_path = "../../models/Qwen/Qwen2.5-Math-7B-Instruct/"

# tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    add_eos_token=False, 
)

print(tokenizer.encode('a ки b'))
print(tokenizer.encode('a b'))

print(tokenizer.encode('a \n\n b'))
print(tokenizer.encode('a b'))
print(tokenizer.encode('a \n\n\n\n\n\n b'))
print(tokenizer.encode('a b'))


print(tokenizer.encode('a \n\n\n\n\n\n\n b'))
print(tokenizer.encode('a b'))

print(tokenizer.encode('a \n\n\n\n\n\n\n\n b'))
print(tokenizer.encode('a b'))


print(tokenizer.encode('a + b'))
print(tokenizer.encode('a b'))

print(tokenizer.encode('a - b'))
print(tokenizer.encode('a b'))

print(tokenizer.encode(' + -'))
print(tokenizer.encode('+-'))


# if USE_8bit is True:
#     model = prepare_model_for_int8_training(model)
print(tokenizer.eos_token_id)

tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
tokenizer.padding_side = "left"  # Allow batched inference


# tokenizer = AutoTokenizer.from_pretrained('peiyi9979/math-shepherd-mistral-7b-prm')
candidate_tokens = tokenizer.encode(f" {good_token} {bad_token}") # [488, 481]
print(candidate_tokens)
step_tag_id = tokenizer.encode(f" {step_tag}")[-1] # 76325
print('step_tag_id:',tokenizer.encode(f" {step_tag}"))
print('step_tag_id2:',tokenizer.encode(f"{step_tag2}"))
# model = AutoModelForCausalLM.from_pretrained('peiyi9979/math-shepherd-mistral-7b-prm').eval()
# model = AutoModelForCausalLM.from_pretrained(model_path).eval()
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # load_in_8bit=True,   # Enables 8-bit quantization
    device_map="auto",   # Automatically assigns the model to available GPUs/CPUs
    torch_dtype=torch.float16,  # Mixed precision for faster inference
    attn_implementation="flash_attention_2",
)

# for name,param in model.named_parameters():
#     print(name)
print(model)

# lora_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,  # LoRA for causal language modeling task
#     r=8,  # Rank of LoRA
#     lora_alpha=32,  # Alpha scaling factor for LoRA
#     lora_dropout=0.1,  # Dropout rate for LoRA layers
#     target_modules=["q_proj", "v_proj"],  # Apply LoRA to specific layers
# )

# model = get_peft_model(model, lora_config)

adapter_config = PeftConfig.from_pretrained('./prm_results_qwen/bs_256_lr_0.0001/checkpoint-3449')

# Wrap the pre-trained model with the LoRA fine-tuned weights
model = PeftModel.from_pretrained(model, './prm_results_qwen/bs_256_lr_0.0001/checkpoint-3449')

# model.to('cuda:0')
print(model.device)
question = "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
output1 = "Step 1: Janet's ducks lay 16 eggs per day. \n\n\n\n\n Step 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. \n\n\n\n\n Step 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. \n\n\n\n\n Step 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $18 every day at the farmers' market. The answer is: 18 \n\n\n\n\n" # 18 is right
output2 = "Step 1: Janet's ducks lay 16 eggs per day. \n\n\n\n\n Step 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. \n\n\n\n\n Step 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. \n\n\n\n\n Step 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $17 every day at the farmers' market. The answer is: 17 \n\n\n\n\n" # 17 is wrong


for output in [output1,output2]:
# for output in [output1, output2,output3]:
    input_for_prm = f"{question} {output}"
    input_id = torch.tensor([tokenizer.encode(input_for_prm)])
    # print(input_id)

    with torch.no_grad():
        logits = model(input_id).logits[:,:,candidate_tokens]
        # print(logits)
        scores = logits.softmax(dim=-1)[:,:,0] 
        # print(scores)
        step_scores = scores[input_id == step_tag_id]
        
        print(step_scores)
        print('aaaaaa')        
# tensor([0.9955, 0.9958, 0.9983, 0.9957])
# tensor([0.9955, 0.9958, 0.9983, 0.0240])

