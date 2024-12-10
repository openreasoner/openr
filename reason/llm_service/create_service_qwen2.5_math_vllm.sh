set -e

HOST_ADDR=0.0.0.0
CONTROLER_PORT=28777
WORKER_BASE_PORT=30010

echo PYTHON_EXECUTABLE=$(which python3)
PYTHON_EXECUTABLE=$(which python3)

# MODEL_BASE=/mnt/nasdata/siting/models
CUDA_DEVICE_BASE=0
# POLICY_MODEL_NAME=Qwen/Qwen2.5-Math-1.5B-Instruct
# POLICY_MODEL_NAME=Qwen/Qwen2-VL-2B-Instruct
# POLICY_MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
# VALUE_MODEL_NAME=qwen_prm/checkpoint-6898/
# VALUE_MODEL_NAME=Qwen/Qwen2.5-Math-7B-PRM
# MODEL_PATH=$MODEL_BASE/$POLICY_MODEL_NAME
# VALUE_MODEL_PATH=$MODEL_BASE/$VALUE_MODEL_NAME
MODEL_PATH=/mnt/nasdata/siting/models/Qwen/Qwen2-VL-2B-Instruct
VALUE_MODEL_PATH=/mnt/nasdata/yanxue/llms/huggingface/math-shepherd-mistral-7b-prm

LOGDIR=logs_fastchat

tmux start-server
tmux new-session -s FastChat -n controller -d
tmux send-keys "export LOGDIR=${LOGDIR}" Enter
tmux send-keys "$PYTHON_EXECUTABLE -m fastchat.serve.controller --port ${CONTROLER_PORT} --host $HOST_ADDR" Enter

NUM_LM_WORKER=1
NUM_RM_WORKER=1

echo "Wait 5 seconds ..."
sleep 5

echo "Starting workers"
for i in $(seq 0 $((NUM_LM_WORKER-1)))
do
  WORKER_PORT=$((WORKER_BASE_PORT+i))
  tmux new-window -n policy_worker_$i
  tmux send-keys "export LOGDIR=${LOGDIR}" Enter
  tmux send-keys "CUDA_VISIBLE_DEVICES=$((i+CUDA_DEVICE_BASE)) $PYTHON_EXECUTABLE -m reason.llm_service.workers.vllm_worker --model-path $MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT" Enter
done


# start value service
for i in $(seq 0 $((NUM_RM_WORKER-1)))
do
  WORKER_PORT=$((i+WORKER_BASE_PORT+NUM_LM_WORKER))
  tmux new-window -n value_worker
  tmux send-keys "export LOGDIR=${LOGDIR}" Enter
  tmux send-keys "CUDA_VISIBLE_DEVICES=$((i+NUM_LM_WORKER+CUDA_DEVICE_BASE)) $PYTHON_EXECUTABLE -m reason.llm_service.workers.reward_model_worker --model-path $VALUE_MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT" Enter
done