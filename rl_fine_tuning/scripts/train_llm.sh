#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python -u train_math.py --seed 10 --dataset_name "prealgebra" --algorithm_name "APPO" --experiment_name "dev" --num_mini_batch 4 --ppo_epoch 1

