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