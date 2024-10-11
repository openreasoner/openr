---
title: Quick Start
nav_order: 2.5
---

# Quick Start
{: .no_toc }

Welcome to the **OpenR** Manual, designed to guide you through the process of training large language models (LLMs) to reason effectively. Here we provide a quick guide of how to successfully run the codebase of **OpenR**.

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Prerequisites

We have tested our code on machines with minimum 2 x A800 GPUs, each with 80GB memory. For optimal performance, it is recommended to run the project on machine with at least 80GB of GPU memory.

## Installation

### Create Environment using Conda 

1. Create and activate a new conda environment

```bash
conda create -n open_reasonser python=3.10
conda activate open_reasonser 
```

2. Intall dependencies

```bash
pip install -r requirements.txt
pip3 install  "fschat[model_worker,webui]"
pip install -U pydantic
cd envs/MATH/latex2sympy
pip install -e .
cd -
```


### Download Base Models

Before running the project, please ensure that all required base models are downloaded. The models used in this project include:

- `Qwen2.5-Math-1.5B-Instruct`, `Qwen2.5-Math-7B-Instruct`
- `Qwen2.5-Math-RM-72B`
- `peiyi9979/mistral-7b-sft`
- `peiyi9979/math-shepherd-mistral-7b-prm`

To download these models, please refer to the [Hugging Face model downloading tutorial](https://huggingface.co/docs/hub/models-downloading) for step-by-step guidance on downloading models from the Hugging Face Hub.

Ensure that all models are saved in their directories according to the project setup before proceeding.

## Start a LLM Service

Before running inference, please modify the following variables in the `reason/llm_service/create_service_math_shepherd.sh` script to set the appropriate base models for your usage:

- `$MODEL_BASE`: Set this to the directory where your models are stored.
- `$POLICY_MODEL_NAME`: Set this to the name of the policy model you wish to use.
- `$VALUE_MODEL_NAME`: Set this to the name of the value model you wish to use.
- `$NUM_LM_WORKER`: Set this to the number of language model (LM) workers to start.
- `$NUM_RM_WORKER`: Set this to the number of reward model (RM) workers to start.

```bash
sh reason/llm_service/create_service_math_shepherd.sh
```

## Run Inference

```bash
export PYTHONPATH=$(pwd)
sh scripts/eval/cot_greedy.sh
sh scripts/eval/cot_rerank.sh
sh scripts/eval/beam_search.sh
```

## Run Training

Before training, please modify the `$dataset_path`, `$model_name_or_path` and `$prm_name_or_path` in `train/mat/scripts/train_llm.sh`.
```bash
cd train/mat/scripts
bash train_llm.sh
```