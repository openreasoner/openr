python reason/evaluation/eval_cot.py \
    --LM mistral-7b-sft \
    --RM math-shepherd-mistral-7b-prm \
    --env_name MATH \
    --temperature 0.0 \
    --max_new_tokens 1024 \
    --controller_addr http://0.0.0.0:28777