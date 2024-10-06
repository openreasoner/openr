# Open O1 dev


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

### Run Inference
```
export PYTHONPATH=$(pwd)
sh scripts/eval/cot_greedy.sh
sh scripts/eval/cot_rerank.sh
sh scripts/eval/beam_search.sh
```

### Run Experiment for me @(anjie)
see `scripts/eval/exp/`, run scripts in it.