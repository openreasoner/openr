---
title: Reward Models
parent: Usage
nav_order: 4
---


# Supervised Training for PRMs

In Process-supervision Reward Models (PRMs), the goal is to determine whether the sequence of solution process is currently on the right track, so it should output a binary indicator of correctness.

## Data Preprocessing

The datasets we used to train our PRM include [PRM800K](https://github.com/openai/prm800k), [Math-Shepherd](https://huggingface.co/datasets/peiyi9979/Math-Shepherd) and our dataset --- MATH-APS. These datasets are structured into three parts:

- Question:

```json
"question" : "Three pencils and a jumbo eraser cost $\\$1.24$. Five pencils and a jumbo eraser cost $\\$1.82$. No prices include tax. In cents, what is the cost of a pencil?"
```

- Process: the solution is broken down into multiple steps, with each step separated by a special step token represented as `\n\n\n\n\n`, indicating the end of a step, at which point the PRM can make predictions.

```json
"process" : 
"Step: 1: Let's call the price of a pencil p and the price of a jumbo eraser e. Then we can write two equations. \n\n\n\n\n
Step: 2: The first equation is $3p+e=124$. \n\n\n\n\n
Step: 3: To solve this system, let's subtract the first equation from the second equation. This will eliminate e. \n\n\n\n\n
Step: 4: $5p+e-3p-e=1.82-1.24$. \n\n\n\n\n
Step: 5: This simplifies to $2p=0.58$. So $p=0.29$. \n\n\n\n\n
Step: 6: We could also solve this system by substitution. \n\n\n\n\n"
```


- Label: corresponds to the classification for all the steps within the entire process, and it is either a `+` or a `-` based on the correctness of the process.

```json
"label" : ["+", "-", "+", "+", "+", "+"]
```

More details of data preprocessing can be found in [Datasets]({% link docs/datasets.md %}).

## Evaluation & Fine-tuning

Our method involves defining a special **step token**, denoted as `\n\n\n\n\n`, followed by
two additional tokens representing positive and negative feedback, denoted as `+` and `-`. We then use the LLM to predict the next token of the step token (implemeted by `preprocess_function()` in `prm/supervise/evaluate.py `). 

From the logits of the positive and negative tokens in the position, we apply softmax and use the score of the `+` token as the prediction result (retrieved by `preprocess_logits_for_metrics()` in `prm/supervise/evaluate.py `).

One we either evaluate or train through:
```python
# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],  # Replace with a validation set if available
    data_collator=data_collator,
    tokenizer=tokenizer,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    compute_metrics=compute_metrics,
)

trainer.evaluate()
trainer.train()
```
