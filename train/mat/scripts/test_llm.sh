#!/bin/sh


# CUDA_VISIBLE_DEVICES=0 python -u test_math.py  --peft_path ${peft_path} --variant "Washing Plate" --seed 10 --n_eval_rollout_threads 10 --eval_episodes 100

CUDA_VISIBLE_DEVICES=0 python -u test_math.py --seed 10 \
                        --dataset_name "prealgebra" \
                        --dataset_path "../envs/math/data/math_500.jsonl" \
                        --model_name_or_path "MODEL_PATH" \
                        --model_peft_path "PEFT_PATH"

