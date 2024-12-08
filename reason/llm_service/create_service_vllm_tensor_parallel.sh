set -e

HOST_ADDR=0.0.0.0
CONTROLER_PORT=28777
WORKER_BASE_PORT=30010

echo PYTHON_EXECUTABLE=$(which python3)
PYTHON_EXECUTABLE=$(which python3)

MODEL_BASE=/hpc2ssd/JH_DATA/spooler/qxiao183/workspace/hf_models
POLICY_MODEL_NAME=Qwen/Qwen2.5-Math-1.5B-Instruct
MODEL_PATH=/mnt/nasdata/llms/qwen/Qwen2.5-32B-Instruct

LOGDIR=logs_fastchat

tmux start-server
tmux new-session -s FastChat -n controller -d
tmux send-keys "export LOGDIR=${LOGDIR}" Enter
tmux send-keys "$PYTHON_EXECUTABLE -m fastchat.serve.controller --port ${CONTROLER_PORT} --host $HOST_ADDR" Enter

NUM_LM_WORKER=1
GPU_IDX="0,1"
# Calculate the number of GPUs
NUM_GPUS=$(echo "$GPU_IDX" | tr -cd ',' | wc -c)
NUM_GPUS=$((NUM_GPUS + 1))

echo "Wait 5 seconds ..."
sleep 5

echo "Starting workers"
for i in $(seq 0 $((NUM_LM_WORKER-1)))
do
  WORKER_PORT=$((WORKER_BASE_PORT+i))
  tmux new-window -n policy_worker_$i
  tmux send-keys "export LOGDIR=${LOGDIR}" Enter
#  tmux send-keys "export VLLM_WORKER_MULTIPROC_METHOD=spawn" Enter    # https://github.com/vllm-project/vllm/issues/6152#issuecomment-2211709345
  tmux send-keys "CUDA_VISIBLE_DEVICES=$GPU_IDX $PYTHON_EXECUTABLE -m reason.llm_service.workers.vllm_worker --model-path $MODEL_PATH --num-gpus $NUM_GPUS --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT" Enter
done

