python reason/evaluation/eval_cot.py \
    --LM mistral-7b-sft \
    --RM math-shepherd-mistral-7b-prm \
    --env_name MATH \
    --temperature 1.0 \
    --num_sequence 32 \
    --exp_name cot_rerank \
    --max_new_tokens 512 \
    --controller_addr http://0.0.0.0:28777