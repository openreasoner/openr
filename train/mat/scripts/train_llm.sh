#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python -u train_math.py --seed 10 \
                        --dataset_name "prealgebra" \
                        --dataset_path "/hpc2ssd/JH_DATA/spooler/qxiao183/workspace/muning/code/o1-dev/train/mat/envs/math/data/math_500.jsonl" \
                        --model_name_or_path "/hpc2ssd/JH_DATA/spooler/qxiao183/workspace/muning/models/Qwen2.5-Math-1.5B" \
                        --prm_type "Qwen" \
                        --prm_model_name_or_path "/hpc2ssd/JH_DATA/spooler/qxiao183/workspace/muning/models/Qwen2.5-Math-7B-Instruct/" \
                        --prm_checkpoint_path /hpc2ssd/JH_DATA/spooler/qxiao183/workspace/muning/prm/checkpoint-3449 \
                        --algorithm_name "APPO" \
                        --experiment_name "test_prm" \
                        --num_mini_batch 4 \
                        --ppo_epoch 1

