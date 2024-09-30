#!/bin/sh

CUDA_VISIBLE_DEVICES=0 
python -u train_math.py --seed 10 \
                        --dataset_name "prealgebra" \
                        --dataset_path "/home/jwliao/codes/Math-Agent/mat/envs/math/data/merged_precalculus_train.json" \
                        --model_name_or_path "/home/jwliao/models/Qwen2.5-Math-1.5B" \
                        --prm_name_or_path "/home/jwliao/models/math-shepherd-mistral-7b-prm" \
                        --algorithm_name "APPO" \
                        --experiment_name "dev" \
                        --num_mini_batch 4 \
                        --ppo_epoch 1

