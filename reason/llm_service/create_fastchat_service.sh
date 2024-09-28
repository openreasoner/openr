set -e

HOST_ADDR=0.0.0.0
CONTROLER_PORT=28778
WORKER_BASE_PORT=30110

echo PYTHON_EXECUTABLE=$(which python3)
PYTHON_EXECUTABLE=$(which python3)

MODEL_BASE=/hpc2ssd/JH_DATA/spooler/qxiao183/workspace/hf_models/
# CUDA_DEVICE_BASE = 0
# POLICY_MODEL_NAME=peiyi9979/mistral-7b-sft
# VALUE_MODEL_NAME=peiyi9979/math-shepherd-mistral-7b-prm
CUDA_DEVICE_BASE=2
POLICY_MODEL_NAME=tsllm/llama2-7b-game24-policy-hf
VALUE_MODEL_NAME=tsllm/llama2-7b-gsm8k-value
MODEL_PATH=$MODEL_BASE/$POLICY_MODEL_NAME
VALUE_MODEL_PATH=$MODEL_BASE/$VALUE_MODEL_NAME

LOGDIR=logs_fastchat

tmux start-server
tmux new-session -s FastChat1 -n controller -d
tmux send-keys "export LOGDIR=${LOGDIR}" Enter
tmux send-keys "$PYTHON_EXECUTABLE -m fastchat.serve.controller --port ${CONTROLER_PORT} --host $HOST_ADDR" Enter

echo "Wait 10 seconds ..."
sleep 5

echo "Starting workers"
for i in $(seq 0 0)
do
  WORKER_PORT=$((WORKER_BASE_PORT+i))
  tmux new-window -n policy_worker_$i
  tmux send-keys "export LOGDIR=${LOGDIR}" Enter
  tmux send-keys "CUDA_VISIBLE_DEVICES=$((i+CUDA_DEVICE_BASE)) $PYTHON_EXECUTABLE -m fastchat.serve.vllm_worker --model-path $MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT --dtype bfloat16 --swap-space 32" Enter
done


# start value service
WORKER_PORT=$((WORKER_BASE_PORT+1))
tmux new-window -n value_worker
tmux send-keys "export LOGDIR=${LOGDIR}" Enter
tmux send-keys "CUDA_VISIBLE_DEVICES=$((1+CUDA_DEVICE_BASE)) $PYTHON_EXECUTABLE -m fastchat.serve.value_model_worker --model-path $VALUE_MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT" Enter