python reason/evaluation/evaluate.py \
    --LM mistral-7b-sft \
    --RM math-shepherd-mistral-7b-prm \
    --task_name MATH \
    --temperature 1.0 \
    --num_sequence 32 \
    --max_new_tokens 512 \
    --save_dir results \
    --controller_addr http://0.0.0.0:28777