# Open O1 dev


## Environment
```
conda create -n open_reasoner python=3.10
conda activate open_reasoner
pip install -r requirements.txt

cd .. 
git clone git@github.com:ziyuwan/FastChat_tsllm.git
cd FastChat_tsllm
pip3 install -e ".[model_worker,webui]"
```

## Running
`export PYTHONPATH=$(pwd)`
### Start LM & RM Services
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