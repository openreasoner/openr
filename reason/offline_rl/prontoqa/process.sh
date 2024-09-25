set -e
INPUT_DIR="prontoqa/cot_sample"
OUTPUT_DIR="prontoqa/processed"
mkdir -p $OUTPUT_DIR

N=50

file_prefix="prontoqa_train_cot_sample_offline_sft_k100_ep1"
python dedup.py \
    --input_path ${INPUT_DIR}/${file_prefix}.jsonl \
    --output_path ${OUTPUT_DIR}/${file_prefix}_dedup.jsonl
# split_two_test.py will choose files encswith "dedup"
python split_two_test.py \
    --train_data_prefix ${OUTPUT_DIR}/${file_prefix} \
    --train_num $N \
    --train_test_num 3
