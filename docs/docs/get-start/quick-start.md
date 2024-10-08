---
title: Quick Start
parent: Getting Started
nav_order: 3
---

# Quick Start
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

Before running inference, please modify the following variables in the `reason/llm_service/create_service_math_shepherd.sh` script to set the appropriate base models for your usage:

- `$MODEL_BASE`: Set this to the directory where your models are stored.
- `$POLICY_MODEL_NAME`: Set this to the name of the policy model you wish to use.
- `$VALUE_MODEL_NAME`: Set this to the name of the value model you wish to use.
- `$NUM_LM_WORKER`: Set this to the number of language model (LM) workers to start.
- `$NUM_RM_WORKER`: Set this to the number of reward model (RM) workers to start.

This following starts the language model (LM) and reward model (RM) services required for running inference. Then it prepares and runs inference using different techniques.
{: .fs-6 .fw-300 }


## Start LM & RM Services

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
