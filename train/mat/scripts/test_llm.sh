#!/bin/sh


# CUDA_VISIBLE_DEVICES=0 python -u test_math.py  --peft_path ${peft_path} --variant "Washing Plate" --seed 10 --n_eval_rollout_threads 10 --eval_episodes 100

CUDA_VISIBLE_DEVICES=0 python -u test_math.py --seed 10 \
                        --dataset_name "prealgebra" \
                        --dataset_path "/hpc2ssd/JH_DATA/spooler/qxiao183/workspace/muning/code/o1-dev/train/mat/envs/math/data/math_500.jsonl" \
                        --model_name_or_path "/hpc2ssd/JH_DATA/spooler/qxiao183/workspace/muning/models/Qwen2.5-Math-1.5B" \
                        --model_peft_path "/hpc2ssd/JH_DATA/spooler/qxiao183/workspace/muning/code/o1-dev/train/mat/scripts/results/ms_all/prealgebra/APPO/run1/models/episode_1000"

