python reason/evaluation/evaluate.py \
    --LM Qwen2.5-Math-1.5B-Instruct \
    --LM_config "reason/resource/qwen2.5/config.json" \
    --RM math-shepherd-mistral-7b-prm \
    --RM_config "reason/resource/mistral/shepherd_prm_config.json" \
    --task_name MATH \
    --temperature 0.7 \
    --num_sequence 4 \
    --max_new_tokens 2048 \
    --save_dir qwen_results \
    --method best_of_n \
    --num_worker 32 \
    --LM_addr http://0.0.0.0:28777 \
    --RM_addr http://0.0.0.0:28777

# math-shepherd-mistral-7b-prm
