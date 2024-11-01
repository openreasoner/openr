#!/bin/bash

# Set the model and other parameters
MODEL_NAME="/root/.cache/modelscope/hub/Qwen/Qwen2___5-Math-7B-Instruct"
DEVICE="cuda"
MAX_NEW_TOKENS=2048
TEMPERATURE=0.7
TOP_K=30
TOP_P=0.9
C_PUCT=0.125
ALPHA=0.5
BETA=0.9
LENGTH_SCALE=500
NUM_ROLLOUTS=16
MAX_SEARCH_COUNT=20
ROLLOUT_BUDGET=200
OUTPUT_DIR="output_results"

# Split files directory
SPLIT_DIR="output_directory"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR
mkdir -p  log

# Start the OmegaPRM process on each GPU with separate split files
for i in {1..8}
do
    SPLIT_FILE="$SPLIT_DIR/questions_part_${i}.json"
    GPU_ID=$((i-1))
    OUTPUT_FILE="$OUTPUT_DIR/results_part_${i}.json"
    LOG_FILE_PREFIX="log/omega_prm_gpu_$GPU_ID"

    # Run the OmegaPRM process in the background on the specified GPU
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 run_omegaprm.py \
        --question_file $SPLIT_FILE \
        --output_dir $OUTPUT_FILE \
        --model_name $MODEL_NAME \
        --device $DEVICE \
        --max_new_tokens $MAX_NEW_TOKENS \
        --temperature $TEMPERATURE \
        --top_k $TOP_K \
        --top_p $TOP_P \
        --c_puct $C_PUCT \
        --alpha $ALPHA \
        --beta $BETA \
        --length_scale $LENGTH_SCALE \
        --num_rollouts $NUM_ROLLOUTS \
        --max_search_count $MAX_SEARCH_COUNT \
        --rollout_budget $ROLLOUT_BUDGET \
        --log_file_prefix $LOG_FILE_PREFIX &
done

# Wait for all processes to finish
wait

echo "All OmegaPRM processes complete."
