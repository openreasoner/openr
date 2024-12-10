python reason/evaluation/evaluate.py \
    --LM Qwen2-VL-2B-Instruct\
    --RM math-shepherd-mistral-7b-prm\
    --task_name MATH \
    --temperature 0.0 \
    --max_new_tokens 2048 \
    --save_dir results \
    --method cot \
    --num_worker 32 \
    --controller_addr http://0.0.0.0:28777 \
    --is_multimodal True\
    --single_problem '{"question": "According to the following questions, which option in the image is correct? Give the answer from A, B, C and D.", "answer": "C", "image": "/home/siting/openr-mm/mathematics-0.png"}'
    # --single_problem '{"question": "What is the answer of 2+2?", "answer": "4"}'