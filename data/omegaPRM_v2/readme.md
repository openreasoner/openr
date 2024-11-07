# OmegaPRM Multi-GPU Runner

This script runs `OmegaPRM` on multiple GPUs, with each GPU handling a different part of the dataset for parallel processing.

## Steps to Use

1. **Split the Input Data**

   Use `process_json.py` to split your input JSON file into multiple parts for each GPU:

   ```bash
   python process_json.py --input_file questions.json --output_dir output_directory --num_splits 8
   ```
   
   2. **Run the Script**
   
      Use `run_omegaprm_multi_gpu.sh` to start processing with OmegaPRM on each GPU:
       ``` bash
      run_omegaprm_multi_gpu.sh
       ```
      Results are saved in `output_results`.
   
      **Note**: Before running, make sure to set the correct values for parameters in the script. Important parameters include:

        -`MODEL_NAME`: Path to the model (e.g., Hugging Face or vLLM model).

        -`MODEL_TYPE`: Set to "hf" for Hugging Face or "vllm" for vLLM support.
   
        -`Other parameters` like MAX_NEW_TOKENS, TEMPERATURE, TOP_K, and other hyperparameters according to your needs.
## Run on a Single GPU

```bash
CUDA_VISIBLE_DEVICES=0 python run_omegaprm.py \
    --question_file ../extracted_problems_and_answers.json \
    --output_dir output_results \
    --model_name "Your model name or path " \
    --model_type $MODEL_TYPE \
    --device "cuda" \
    --max_new_tokens 2048 \
    --temperature 0.7 \
    --top_k 30 \
    --top_p 0.9 \
    --c_puct 0.125 \
    --alpha 0.5 \
    --beta 0.9 \
    --length_scale 500 \
    --num_rollouts 16 \
    --max_search_count 20 \
    --rollout_budget 200 \
    --log_file_prefix "log/omega_prm_single_gpu"

```