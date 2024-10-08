---
title: Installation
parent: Getting Started
nav_order: 2
---

# Installation
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---



##  Create Environment using Conda 

### Step 1: Create and activate a new conda environment

```bash
conda create -n open_reasonser python=3.10
```

```
conda activate open_reasonser 
```

### Step 2: Intall dependencies

```bash=
pip install -r requirements.txt
```



## Download  Base Models


Before running the project, please ensure that all required base models are downloaded. The models used in this project include:

- `Qwen2.5-Math-1.5B-Instruct`, `Qwen2.5-Math-7B-Instruct`
- `Qwen2.5-Math-RM-72B`
- `peiyi9979/mistral-7b-sft`
- `peiyi9979/math-shepherd-mistral-7b-prm`

To download these models, please refer to the [Hugging Face model downloading tutorial](https://huggingface.co/docs/hub/models-downloading) for step-by-step guidance on downloading models from the Hugging Face Hub.

Ensure that all models are saved in their directories according to the project setup before proceeding.


