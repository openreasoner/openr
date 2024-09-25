set -e
# export TEST_NO_TERMINAL=1
# export TEST_WITH_TERMINAL=1
# export TEST_COT_GREEDY=1
export TEST_COT_SC=1

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

CT2_CACHE=$1
TOKENIZER_PATH=$2
# just None critic because we only use cot-sc sample
CRITIC_PATH="None"

# also don't forget to set k_maj as 50 in test_sft_and_v_rlhf.py
torchrun --nproc_per_node=8 --master-port 29503 ../test_sft_and_v_rlhf.py \
    --critic_model_path $CRITIC_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --ct2_dir $CT2_CACHE \
    --save_dir ./rlhf/cot_sample \
    --env_name rlhf
    --train