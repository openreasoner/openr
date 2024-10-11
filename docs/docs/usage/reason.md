---
title: Reason
parent: Usage
nav_order: 5
---


# Reasoning
After trained the process reward model,
we implement how to do reasoning with diffrent search strategies

## Basic Command

To run inference using the Qwen 1.5B model as the policy model and Mistral-7B as the reward model on the MATH dataset, use the following command:

```python
python reason/evaluation/evaluate.py \
    --LM Qwen2.5-Math-1.5B-Instruct \
    --RM math-shepherd-mistral-7b-prm \
    --task_name MATH \
    --temperature 0.7 \
    --num_sequence 8 \
    --max_new_tokens 2048 \
    --method best_of_n
```
### Parameters

- **`--LM`**: Specifies the policy model. Available models include `Qwen2.5-Math-1.5B-Instruct` and others. Replace with the desired model path or name.
- **`--RM`**: Sets the reward model. In this example, `math-shepherd-mistral-7b-prm` is used, which can be replaced with any supported reward model.
- **`--task_name`**: Sets the dataset/task name. Examples include `MATH`, `gsm8k`, etc.
- **`--temperature`**: Controls the randomness of the model output. A higher temperature increases randomness, while a lower value makes the output more deterministic.
- **`--num_sequence`**: Number of sequences generated per inference step. Useful for methods like `best_of_n`, where multiple outputs are generated and evaluated.
- **`--max_new_tokens`**: Specifies the maximum number of new tokens to generate.
- **`--method`**: Chooses the search strategy. Options include:
  - `MCTS` - Monte Carlo Tree Search
  - `beam_search` - Beam Search
  - `best_of_n` - Select the best sequence from `n` generated options
- **`--tree_max_depth`**: Specifies the maximum depth of the search tree. Primarily useful when using tree-based methods like MCTS or beam search.
- **`--tree_max_width`**: Limits the number of child nodes each tree node can have. Useful for controlling memory usage and search complexity.
- **`--save_dir`**: Directory where model checkpoints or results will be saved. If not specified, results are not saved.
- **`--resume_dir`**: Directory containing previous checkpoints to resume training or evaluation from. If specified, the script will continue from the last saved state.
- **`--local`**: When set, runs the script in local debug mode, reducing parallelism and simplifying setup.
- **`--num_worker`**: Specifies the number of parallel worker processes. Useful for speeding up processing with large datasets or multi-agent environments.
