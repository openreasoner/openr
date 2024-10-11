---
title: Datasets
nav_order: 4.6
---

# Datasets

## PRM800K

The original dataset can be found in [this GitHub repo](https://github.com/openai/prm800k). We
applied some preprocessing to the original dataset:

### Step tag

We use `\n\n\n\n\n` as the step tag, which is appended to each step.

### For binary classification

PRM800K has three labels: `good`, `neutral`, and `bad`, but we regard PRM training as a binary
classification problem. Hence, we treat steps labelled `neutral` or `good` in PRM800K as positive
ones and those labelled `bad` as negative ones.

### Multiple candidates for each step

Each step may contain one or more completions in the original PRM800K. We include all the possible
combinations of these candidates in the training data.

## MATH-APS

This dataset was collected by us following the procedure introduced [here]({% link
docs/usage/data.md %}).

### For binary classification

MATH-APS scores steps with values between 0 and 1, so we label steps whose score is greater than 0.5
as positive and others negative.

## Math-Shepherd

The original dataset can be found in [this Hugging Face dataset
repo](https://huggingface.co/datasets/peiyi9979/Math-Shepherd). It was used to train the
[Math-Shepherd](http://arxiv.org/abs/2312.08935) PRM.

### Step tag

Math-Shepherd uses `ки` as its step tag, so we replace them with `\n\n\n\n\n`, the step tag we
choose for our model.
