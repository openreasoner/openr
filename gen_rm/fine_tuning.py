from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model

def preprocess_function(examples, tokenizer):
    # Tokenize the text and set labels
    inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

def train_genrm(model_name, tokenizer, output_dir):
    # model_suffix = model_name.split("/")[-1]
    # dataset_path = f"./dataset/organized_{model_suffix}_prm800k_train.json"
    dataset_path = "./dataset/organized_Llama-3.1-8B-Instruct_prm800k_train.json"

    # Load dataset with caching enabled to speed up future loads
    dataset = load_dataset("json", data_files=dataset_path, split="train[0:1000]")

    # Load model with efficient device mapping and caching
    standard_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        cache_dir="/shares/mxq6904/models",
    )

    # Configure LoRA with task-specific settings
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # Apply LoRA to the standard model
    model = get_peft_model(standard_model, lora_config)

    # Use a data collator for causal language modeling (no masking for LM)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Set training arguments with mixed precision training enabled if supported
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=100,
        save_total_limit=5,
        fp16=True,  # Enable mixed precision training to speed up training and reduce memory usage
    )

    # Process dataset using multi-processing to speed up tokenization
    dataset = dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True, num_proc=4, remove_columns=dataset.column_names)

    # Create a Trainer instance with the processed dataset
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Start the training process
    trainer.train()

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    # Load the tokenizer once and pass it to the train function
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token if not set by default

    # Call training function with model name, tokenizer, and output directory
    train_genrm(model_name, tokenizer, "./output")