set -e

INPUT_DIR="gsm8k_data/cot_sample"
OUTPUT_DIR="gsm8k_data/processed"
mkdir -p $OUTPUT_DIR

N=17

file_prefix="gsm8k_train_cot_sample_offline_sft_k100_ep1"
python dedup.py \
    --input_path ${INPUT_DIR}/${file_prefix}.jsonl \
    --output_path ${OUTPUT_DIR}/${file_prefix}_dedup.jsonl
python sample.py \
    --input_path ${OUTPUT_DIR}/${file_prefix}_dedup.jsonl \
    --output_path ${OUTPUT_DIR}/${file_prefix}_dedup_sample${N}.jsonl \
    -n $N


file_prefix="gsm8k_train_cot_sample_offline_sft_k100_ep2"
python dedup.py \
    --input_path ${INPUT_DIR}/${file_prefix}.jsonl \
    --output_path ${OUTPUT_DIR}/${file_prefix}_dedup.jsonl
python sample.py \
    --input_path ${OUTPUT_DIR}/${file_prefix}_dedup.jsonl \
    --output_path ${OUTPUT_DIR}/${file_prefix}_dedup_sample${N}.jsonl \
    -n $N

file_prefix="gsm8k_train_cot_sample_offline_sft_k100_ep3"
python dedup.py \
    --input_path ${INPUT_DIR}/${file_prefix}.jsonl \
    --output_path ${OUTPUT_DIR}/${file_prefix}_dedup.jsonl
# split_two_test.py will choose files encswith "dedup"
python split_two_test.py \
    --train_data_prefix ${OUTPUT_DIR}/${file_prefix} \
    --train_num $N \
    --train_test_num 3

python merge.py \
    --input_paths ${OUTPUT_DIR}/gsm8k_train_cot_sample_offline_sft_k100_ep1_dedup_sample17.jsonl \
    ${OUTPUT_DIR}/gsm8k_train_cot_sample_offline_sft_k100_ep2_dedup_sample17.jsonl \
    ${OUTPUT_DIR}/gsm8k_train_cot_sample_offline_sft_k100_ep3_dedup_sample17.jsonl \
    --output_path ${OUTPUT_DIR}/gsm8k_train_cot_sample_sft_k100_merged_dedup_sample17x3.jsonl