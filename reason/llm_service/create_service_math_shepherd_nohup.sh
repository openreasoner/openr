set -e

HOST_ADDR=0.0.0.0
CONTROLER_PORT=28777
WORKER_BASE_PORT=30010

echo PYTHON_EXECUTABLE=$(which python3)
PYTHON_EXECUTABLE=$(which python3)

MODEL_BASE=/hpc2ssd/JH_DATA/spooler/qxiao183/workspace/hf_models/
CUDA_DEVICE_BASE=0
POLICY_MODEL_NAME=peiyi9979/mistral-7b-sft
VALUE_MODEL_NAME=peiyi9979/math-shepherd-mistral-7b-prm
MODEL_PATH=$MODEL_BASE/$POLICY_MODEL_NAME
VALUE_MODEL_PATH=$MODEL_BASE/$VALUE_MODEL_NAME

LOGDIR=logs_fastchat
mkdir -p $LOGDIR  # 创建日志目录

# 启动控制器服务
echo "Starting controller..."
nohup $PYTHON_EXECUTABLE -m fastchat.serve.controller --port ${CONTROLER_PORT} --host $HOST_ADDR > $LOGDIR/controller.log 2>&1 &

NUM_LM_WORKER=1
NUM_RM_WORKER=1

echo "Wait 10 seconds ..."
sleep 10

# （policy worker
echo "Starting policy workers"
for i in $(seq 0 $((NUM_LM_WORKER-1)))
do
  WORKER_PORT=$((WORKER_BASE_PORT+i))
  echo "Starting policy worker $i on port $WORKER_PORT..."
  export CUDA_VISIBLE_DEVICES=$((i+CUDA_DEVICE_BASE))
  nohup $PYTHON_EXECUTABLE -m fastchat.serve.vllm_worker --model-path $MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT --dtype bfloat16 --swap-space 32 > $LOGDIR/policy_worker_$i.log 2>&1 &
done

#（value worker）
echo "Starting value workers"
for i in $(seq 0 $((NUM_RM_WORKER-1)))
do
  WORKER_PORT=$((i+WORKER_BASE_PORT+NUM_LM_WORKER))
  echo "Starting value worker $i on port $WORKER_PORT..."
  export CUDA_VISIBLE_DEVICES=$((i+NUM_LM_WORKER+CUDA_DEVICE_BASE))
  nohup $PYTHON_EXECUTABLE -m fastchat.serve.math_shepherd_model_worker --model-path $VALUE_MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT > $LOGDIR/value_worker_$i.log 2>&1 &
done

echo "All services started. Logs are stored in $LOGDIR."
