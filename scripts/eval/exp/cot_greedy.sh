python reason/evaluation/evaluate.py \
    --LM mistral-7b-sft \
    --task_name MATH \
    --temperature 0.0 \
    --max_new_tokens 1024 \
    --save_dir results \
    --method cot \
    --num_worker 32 \
    --controller_addr http://0.0.0.0:28777