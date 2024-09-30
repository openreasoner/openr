# Open O1 dev

## Features
- **Various thinking**: CoT greedy 
- **Scalable data**: 

## Evaluation
Todo: performance with more time spent thinking (test-time compute)

## Provided Datasets

[PRM800K](https://github.com/openai/prm800k)

[MATH-APS](https://huggingface.co/datasets/mengfang/MATH-APS)

## Getting Started

### Installation
Todo: libs

### Quickstart
This following starts the language model (LM) and reward model (RM) services required for running inference. 
Then it prepares and runs inference using different techniques.

Start LM & RM Services
```bash
sh reason/llm_service/create_service_math_shepherd.sh
```

Run Inference
```bash
export PYTHONPATH=$(pwd)
sh scripts/eval/cot_greedy.sh
sh scripts/eval/cot_rerank.sh
sh scripts/eval/beam_search.sh
```

## Two-week plan ?

- Inference (mcts-like)

- Data + PRM

- Training (RL-like) for LLM

## References

[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)

[Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)
