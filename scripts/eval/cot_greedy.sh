python reason/evaluation/evaluate.py \
    --LM mistral-7b-sft \
    --RM math-shepherd-mistral-7b-prm \
    --task_name MATH \
    --temperature 0.0 \
    --max_new_tokens 1024 \
    --save_dir results \
    --controller_addr http://0.0.0.0:28777