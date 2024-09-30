from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from datasets import load_dataset

from peft import PeftModel
from peft import get_peft_model, LoraConfig, TaskType
# Ensure bitsandbytes is available for 8-bit quantization
import bitsandbytes as bnb

from torch.nn import BCEWithLogitsLoss
from transformers import DataCollatorWithPadding

good_token = '+'
bad_token = '-'
step_tag = 'ки'

model_path = "../../models/math_stepherd_prm/"

# tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    add_eos_token=False, 
)

# if USE_8bit is True:
#     model = prepare_model_for_int8_training(model)
print(tokenizer.eos_token_id)

tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
tokenizer.padding_side = "left"  # Allow batched inference


# tokenizer = AutoTokenizer.from_pretrained('peiyi9979/math-shepherd-mistral-7b-prm')
candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")[1:] # [648, 387]
step_tag_id = tokenizer.encode(f"{step_tag}")[-1] # 12902
print('step_tag_id:',tokenizer.encode(f"{step_tag}"))
# model = AutoModelForCausalLM.from_pretrained('peiyi9979/math-shepherd-mistral-7b-prm').eval()
# model = AutoModelForCausalLM.from_pretrained(model_path).eval()
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=True,   # Enables 8-bit quantization
    device_map="auto",   # Automatically assigns the model to available GPUs/CPUs
    torch_dtype=torch.float16  # Mixed precision for faster inference
)

# for name,param in model.named_parameters():
#     print(name)
print(model)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # LoRA for causal language modeling task
    r=8,  # Rank of LoRA
    lora_alpha=32,  # Alpha scaling factor for LoRA
    lora_dropout=0.1,  # Dropout rate for LoRA layers
    target_modules=["q_proj", "v_proj"],  # Apply LoRA to specific layers
)

model = get_peft_model(model, lora_config)

# model.to('cuda:0')
print(model.device)
question = "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
output1 = "Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $18 every day at the farmers' market. The answer is: 18 ки" # 18 is right
output2 = "Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $17 every day at the farmers' market. The answer is: 17 ки" # 17 is wrong

output3 = """Step 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $17 every day at the farmers' market. The answer is: 17 ки""" # 18 is right

# Tokenizing function
def preprocess_function(example):
    input = f"{example['question']} {example['process']} {example['label']}"
    tokenized_inputs = tokenizer(
        input, 
        truncation=True, 
        # padding='max_length', 
        max_length=2048,
    )
    length = len(tokenized_inputs['input_ids'])
    tokenized_inputs['labels'] = [-100] *(length-1) + tokenized_inputs['input_ids'][length-1:]
    tokenized_inputs['attention_mask'] = [1] *length
    return tokenized_inputs


DATA_PATH = {
    "train": 'test.json', 
    # "train": '/'.join([data_path, f"train/train_{args.K}_{args.train_type}.json"]), 
    # "val": '/'.join([args.data_path, f"valid/valid_{args.K}_{args.test_type}_sampled.json"]),
    "test": 'test.json',
}

dataset = load_dataset('json', data_files=DATA_PATH)
tokenized_datasets = dataset.map(preprocess_function)
tokenized_datasets['train'] = tokenized_datasets['train'].remove_columns(['question','process','label'])
tokenized_datasets['test'] = tokenized_datasets['test'].remove_columns(['question','process','label'])

print(tokenized_datasets['train']['input_ids'])
print(len(tokenized_datasets['train']['input_ids'][0]))

# Data collator for padding inputs dynamically
data_collator = DataCollatorWithPadding(tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,  # Enable mixed precision for better performance on supported hardware
    report_to="none",  # Set to "wandb" if you are using Weights and Biases for logging
)

# Define a custom metric function (e.g., accuracy for binary classification)
def compute_metrics(eval_pred):
    print(eval_pred)
    logits, labels = eval_pred
    predictions = torch.sigmoid(torch.tensor(logits)).numpy() > 0.5
    labels = torch.tensor(labels).numpy()
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

def preprocess_logits_for_metrics(logits,labels):
    print('aa')

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],  # Replace with a validation set if available
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained('./fine_tuned_math_shepherd_lora_8bit')
tokenizer.save_pretrained('./fine_tuned_math_shepherd_lora_8bit')



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
