---
title: Datasets
nav_order: 4.6
---

# Datasets

## PRM800K

Used to train the process reward model (PRM) of OpenR. The original dataset can be found in [this
GitHub repo](https://github.com/openai/prm800k). We applied some preprocessing to the original
dataset:

### Step tag

We use the same step tag `ки` that [Math-Shepherd](https://arxiv.org/pdf/2312.08935.pdf) uses, so we
append `ки` to each step.

### Binary classification

PRM800K has three labels: `good`, `neutral`, and `bad`, but we regard PRM training as a binary
classification problem. Hence, we treat steps labelled `neutral` or `good` in PRM800K as positive
ones and those labelled `bad` as negative ones.

### Multiple candidates for each step

Each step may contain one or more completions in the original PRM800K. We include all the possible
combinations of these candidates in the training data.

## MATH-APS
