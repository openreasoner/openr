#!/bin.bash

python reason/evaluation/llm_as_judge.py \
       --LM Qwen2-7B-Instruct  \
       --LM_addr http://0.0.0.0:28777 \
       --task_name MATH \
       --data_path xxx/record.jsonl \
       --max_new_tokens 2048 \
       --num_worker 64