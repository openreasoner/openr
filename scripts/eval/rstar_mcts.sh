python reason/evaluation/evaluate.py \
    --LM_config "reason/resource/qwen2.5/config.yaml" \
    --RM_config "reason/resource/mistral/shepherd_prm_config.yaml" \
    --task_name rstar \
    --test_data_path "envs/MATH/dataset/test500.jsonl" \
    --max_new_tokens 2048 \
    --num_sequence 1 \
    --tree_max_width 4 \
    --tree_max_depth 50 \
    --save_dir debug \
    --method rstar_mcts \
    --num_worker 16 \
    --top_k 40 \
    --top_p 0.95 \
    --temperature 0.8 \
    --LM_addr http://0.0.0.0:28777 \
    --RM_addr http://0.0.0.0:28777 \
#    --local