CONTROLER_PORT=21101
echo PYTHON_EXECUTABLE=$(which python3)
PYTHON_EXECUTABLE=$(which python3)
WORKER_BASE_PORT=30010
MODEL_PATH=/data/ziyu/mcts_sep_train_results/llama2_game24_sft_0905/last_model_hf/

tmux start-server

tmux new-session -s FastChat -n controller -d
tmux send-keys "$PYTHON_EXECUTABLE -m fastchat.serve.controller --port ${CONTROLER_PORT}" Enter

echo "Wait 10 seconds ..."
sleep 10

echo "Starting workers"
for i in $(seq 0 0)
do
  WORKER_PORT=$((WORKER_BASE_PORT+i))
  tmux new-window -n worker_$i
  tmux send-keys "CUDA_VISIBLE_DEVICES=$i $PYTHON_EXECUTABLE -m fastchat.serve.vllm_worker --model-path $MODEL_PATH --controller-address http://localhost:$CONTROLER_PORT --port $WORKER_PORT --worker-address http://localhost:$WORKER_PORT --dtype bfloat16 --swap-space 32" Enter
done


# start value service

# CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.value_model_worker --model-path /data/ziyu/huggingface_hub_upload/llama2-7b-game24-value-sft-ep3/ --controller-address http://localhost:21101 --port 40010 --worker-address http://localhost:4001