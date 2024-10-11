
set -ex

LM_NAME=Qwen2.5-Math-1.5B-Instruct
# RM_NAME=checkpoint-6898
RM_NAME=math-shepherd-mistral-7b-prm
TASK_NAME=MATH

NUM_RAY_WORKER=32

CONTROLLER_ADDRESS=http://0.0.0.0:28777

# python reason/evaluation/evaluate.py \
#     --LM $LM_NAME \
#     --task_name $TASK_NAME \
#     --temperature 0.0 \
#     --max_new_tokens 2048 \
#     --save_dir results \
#     --method cot \
#     --num_worker $NUM_RAY_WORKER \
#     --controller_addr $CONTROLLER_ADDRESS

# python reason/evaluation/evaluate.py \
#     --LM $LM_NAME \
#     --RM $RM_NAME \
#     --task_name $TASK_NAME \
#     --temperature 0.7 \
#     --num_sequence 256 \
#     --max_new_tokens 2048 \
#     --save_dir results \
#     --method best_of_n \
#     --num_worker $NUM_RAY_WORKER \
#     --controller_addr $CONTROLLER_ADDRESS




for i in 1 4 16 64
do
    python reason/evaluation/evaluate.py \
    --LM $LM_NAME \
    --RM $RM_NAME \
    --task_name $TASK_NAME \
    --temperature 0.7 \
    --max_new_tokens 2048 \
    --num_sequence $i \
    --tree_max_width 4 \
    --tree_max_depth 50 \
    --save_dir results \
    --method beam_search \
    --num_worker $NUM_RAY_WORKER \
    --controller_addr $CONTROLLER_ADDRESS

# math-shepherd-mistral-7b-prm
done