CUDA_DEVICE_BASE=1

export PYTHONPATH=$(pwd)/../..

HOST_ADDR=0.0.0.0
CONTROLER_PORT=28777
echo PYTHON_EXECUTABLE=$(which python3)
PYTHON_EXECUTABLE=$(which python3)
WORKER_BASE_PORT=30010
MODEL_BASE=/hpc2ssd/JH_DATA/spooler/qxiao183/workspace/hf_models/tsllm
MODEL_PATH=$MODEL_BASE/llama2-7b-game24-policy-hf

tmux start-server

tmux new-session -s FastChat -n controller -d
tmux send-keys "$PYTHON_EXECUTABLE -m fastchat.serve.controller --port ${CONTROLER_PORT} --host $HOST_ADDR" Enter

echo "Wait 10 seconds ..."
sleep 10

echo "Starting workers"
for i in $(seq 0 0)
do
  WORKER_PORT=$((WORKER_BASE_PORT+i))
  tmux new-window -n policy_worker_$i
  tmux send-keys "CUDA_VISIBLE_DEVICES=$((i+CUDA_DEVICE_BASE)) $PYTHON_EXECUTABLE -m fastchat.serve.vllm_worker --model-path $MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT --dtype bfloat16 --swap-space 32" Enter
done


# start value service
tmux new-window -n value_worker
tmux send-keys "CUDA_VISIBLE_DEVICES=$((1+CUDA_DEVICE_BASE)) $PYTHON_EXECUTABLE -m fastchat.serve.value_model_worker --model-path $MODEL_BASE/llama2-7b-gsm8k-value/ --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port 40010 --worker-address http://$HOST_ADDR:4001" Enter