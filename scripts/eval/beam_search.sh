python reason/evaluation/evaluate.py \
    --LM Qwen2.5-Math-1.5B-Instruct \
    --LM_config "reason/resource/qwen2.5/config.json" \
    --RM math-shepherd-mistral-7b-prm \
    --RM_config "reason/resource/mistral/shepherd_prm_config.json" \
    --task_name MATH \
    --test_data_path "envs/MATH/dataset/test500.jsonl" \
    --temperature 0.7 \
    --max_new_tokens 2048 \
    --num_sequence 1 \
    --tree_max_width 4 \
    --tree_max_depth 50 \
    --save_dir qwen1.5_results \
    --method beam_search \
    --num_worker 16 \
    --LM_addr http://0.0.0.0:28777 \
    --RM_addr http://0.0.0.0:28777 \
    --local