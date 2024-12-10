# OpenR Reasoning


## Environment
```
conda create -n open_reasoner python=3.10
conda activate open_reasoner
pip install -r requirements.txt
pip3 install  "fschat[model_worker,webui]"
```

## Running
`export PYTHONPATH=$(pwd)`
### Start LM & RM Services
Set `NUM_LM_WORKER` and `NUM_RM_WORKER`, as well as the model_path config, in `reason/llm_service/create_service_math_shepherd.sh`
```
sh reason/llm_service/create_service_math_shepherd.sh
```
or run model in tensor parallel mode
```angular2html
sh reason/llm_service/create_service_vllm_tensor_parallel.sh
```
⚠️ When setting up multiple LM/RM services, you can rename each service with input `--model-names $NAME` and call by `$NAME` in later execution.

### Run Inference
```
export PYTHONPATH=$(pwd)
sh scripts/eval/cot_greedy.sh
sh scripts/eval/cot_rerank.sh
sh scripts/eval/beam_search.sh
sh scripts/eval/vanila_mcts.sh
```

### Run LLM-as-Judge
After running inference script, log files can be found in saving directory containing `config.json`, `avg_result.json` and `record.json` files. Due to limited capabilities of rule-based answer checker, we can perform LLM-as-Judge on the generated reasoning pathway by running:
```angular2html
sh script/eval/llm_as_judge.sh
```