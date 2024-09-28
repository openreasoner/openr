python reason/evaluation/eval_cot.py \
    --LM mistral-7b-sft \
    --RM math-shepherd-mistral-7b-prm \
    --tokenizer_path /hpc2ssd/JH_DATA/spooler/qxiao183/workspace/hf_models/tsllm/llama2-7b-game24-policy-hf \
    --env_name MATH \
    --temperature 1.0 \
    --num_sequence 10 \
    --exp_name cot_rerank \
    --max_new_tokens 512 \
    --controller_addr http://0.0.0.0:28777