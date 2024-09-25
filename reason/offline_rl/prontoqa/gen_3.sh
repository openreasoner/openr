set -e

K=100
T=0.7
N_WORKER=16
OUTPUT_DIR=./prontoqa/cot_sample/
CUDA_DEVICES=0,1,2,3,4,5,6,7

CT2_CACHE=$1
TOKENIZER_PATH=$2

python generate_data.py \
    -k $K \
    -t $T \
    --num_workers $N_WORKER \
    --gpu_ids $CUDA_DEVICES \
    --ct2_dir ${CT2_CACHE}/llama2_sft_ep1_ct2 \
    --tokenizer_path $TOKENIZER_PATH \
    --output_path ${OUTPUT_DIR}/prontoqa_train_cot_sample_offline_sft_k${K}_ep1.jsonl \
    --env_name prontoqa