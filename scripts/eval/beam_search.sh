python reason/evaluation/evaluate.py \
    --LM Qwen2.5-Math-7B-Instruct \
    --RM math-shepherd-mistral-7b-prm \
    --task_name MATH \
    --temperature 0.7 \
    --max_new_tokens 2048 \
    --num_sequence 1 \
    --tree_max_width 4 \
    --tree_max_depth 50 \
    --save_dir debug \
    --method beam_search \
    --num_worker 32 \
    --local \
    --controller_addr http://0.0.0.0:28777 \
    --resume_dir /hpc2ssd/JH_DATA/spooler/qxiao183/workspace/ziyu/open_reasoner/o1-dev/debug/MATH/beam_search/20241009_021324 


# math-shepherd-mistral-7b-prm
# Qwen2.5-Math-7B-PRM