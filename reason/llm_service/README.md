# Deploy LMs and RMs via FastChat
The code about model workers are adapted from [FastChat official repo](https://github.com/lm-sys/FastChat).
## Requirements
```
pip3 install "fschat[model_worker,webui]"
sudo apt install -y tmux
```
## Examples
For example run Qwen2.5-Math LMs and PRMs we trained.
Set `NUM_LM_WORKER` and `NUM_RM_WORKER`, as well as the model_path config, see details in `create_service_qwen2.5_math_vllm.sh`
```
sh reason/llm_service/create_service_qwen2.5_math_vllm.sh
```