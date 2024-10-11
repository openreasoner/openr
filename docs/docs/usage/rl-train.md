---
title: RL Training
parent: Usage
nav_order: 4.5
---


# RL Training
With built MATH environment, we can train the language models using reinforcment learning algorithms.

## Basic Command
The following command demonstrates how to start training using the Qwen 1.5B model as the policy model and the Qwen 7B model as the reward model on the prealgebra dataset.



```python
python -u train_math.py \
    --dataset_name "prealgebra" \
    --dataset_path "./math_500.jsonl" \
    --model_name_or_path "./Qwen2.5-Math-1.5B" \         
    --prm_model_name_or_path "./Qwen2.5-Math-7B-Instruct" \
    --algorithm_name "APPO" \
    --num_mini_batch 4 \
    --ppo_epoch 1
```


## Supported Algorithms

This framework supports various reinforcement learning algorithms, including:

- **APPO**: Asynchronous Proximal Policy Optimization
- **GRPO**: Group Relative Policy Optimization
- **TPPO**: Policy Optimization with Action Decomposition

You can specify the algorithm using the `--algorithm_name` parameter.

## Customization Options

The framework offers a variety of customization options. For further details on the configuration parameters and their usage, please refer to the `train/mat/config.py` in the repository. This file provides a comprehensive list of all configuration options available within the framework, along with descriptions for each parameter.
.Here are some commonly used parameters:

- **`--dataset_name`**: Name of the dataset to be used (e.g., `prealgebra`).
- **`--dataset_path`**: Path to the dataset file.
- **`--model_name_or_path`**: Name or path of the policy model.
- **`--prm_model_name_or_path`**: Name or path of the process reward model.
- **`--algorithm_name`**: Specifies the algorithm to use (e.g., `APPO`).
- **`--num_mini_batch`**: Number of mini-batches per update.
- **`--ppo_epoch`**: Number of epochs for the PPO update.

The following parameters allow further customization:
### Environment-Specific Parameters



- **`--num_env_steps`**: Total number of environment steps for training.
- **`--episode_length`**: Maximum episode length in the replay buffer.
- **`--n_rollout_threads`**: Number of parallel environments for rollouts.

### Recurrent and Network Options

For recurrent policies or specific network configurations:

- **`--use_recurrent_policy`**: Enable recurrent policy usage.
- **`--hidden_size`**: Hidden layer size for actor/critic networks.
- **`--layer_N`**: Number of layers in the network.

### Evaluation and Logging

Options to control evaluation frequency and logging:

- **`--eval_interval`**: Interval between evaluations.
- **`--log_interval`**: Interval for logging during training.
- **`--save_interval`**: Interval for saving model checkpoints.
