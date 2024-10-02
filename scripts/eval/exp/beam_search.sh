# iterate i in 1, 2, 3, 5, 10, 20
for i in 1 2 3 5 10 20:
do
    python reason/evaluation/evaluate.py \
        --LM mistral-7b-sft \
        --RM math-shepherd-mistral-7b-prm \
        --task_name MATH \
        --temperature 1.0 \
        --max_new_tokens 1024 \
        --num_sequence $i \
        --tree_max_width 20 \
        --tree_max_depth 50 \
        --save_dir results \
        --method beam_search \
        --num_worker 32 \
        --controller_addr http://0.0.0.0:28777
done