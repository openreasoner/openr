set -e
INPUT_DIR="./rlhf/cot_sample"
OUTPUT_DIR="./rlhf/processed"
mkdir -p $OUTPUT_DIR

python3 process.py --input_dir=$INPUT_DIR --output_dir=$OUTPUT_DIR
