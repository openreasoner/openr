python reason/evaluation/evaluate.py \
    --LM Qwen2.5-Math-1.5B-Instruct \
    --task_name MATH \
    --temperature 0.0 \
    --max_new_tokens 2048 \
    --save_dir results \
    --method cot \
    --num_worker 32 \
    --LM_addr http://0.0.0.0:28777 \
    --RM_addr http://0.0.0.0:28777