import json
import os
import jinja2
from transformers import AutoTokenizer


prompt_template_for_GenRM = """
Task: Evaluate the following reasoning process and judge whether the last reasoning step is correct or not.

Instructions:
1. You will be presented with a reasoning-dense question (e.g., math questions) and a procedure of reasoning process.
2. You need to evaluate the reasoning process and judge whether the last reasoning step is correct or not.
3. Your answer should be either "Yes" or "No".

For example, you are given the following question and reasoning process:
Question: John has 3 apples. He gives 1 apple to Mary. How many apples does John have now?
Step 1: John has 3 apples.
Step 2: John gives 1 apple to Mary.
Step 3: John has 3-1=2 apples now.

Because the last reasoning step is correct, your answer should be "Yes".

Now, please evaluate the following reasoning process and judge whether the last reasoning step is correct or not.
Question: {{ question }}
{% for x in process %}
Step {{loop.index}}: {{x}}
{% endfor %}
"""
env = jinja2.Environment(loader = jinja2.FileSystemLoader('.'))

def organize_dataset(dataset_path, output_path, tokenizer):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    organized_dataset = []

    for single in dataset:
        question = single['question']
        process_list = single['process'].split('\n\n\n\n\n')[:-1]
        labels = single['label']

        if len(process_list) != len(labels):
            print("Warning: The number of reasoning steps does not match the number of labels.")
            continue

        for i in range(len(process_list)):
            temp_process_list = process_list[:i+1]
            prompt = env.from_string(prompt_template_for_GenRM).render(question=question, process=temp_process_list)
            label = "Yes" if labels[i] == "+" else "No"
            messages = [
                {"role": "system", "content": "You are a helpful assistant to judge the quality of reasoning process."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": label}
            ]
            tokenized_messages = tokenizer.apply_chat_template(messages, tokenize = False)
            organized_dataset.append(tokenized_messages)

    with open(output_path, 'w') as f:
        for i in organized_dataset:
            jsonl = {'text': i}
            f.write(json.dumps(jsonl) + '\n')

if __name__ == "__main__":
    tokenizer_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer_suffix = tokenizer_name.split("/")[-1]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    organize_dataset("./dataset/prm800k_train.json", 
                     f"./dataset/organized_{tokenizer_suffix}_prm800k_train.json", 
                     tokenizer)