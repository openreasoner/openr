---
title: Data
parent: Usage
nav_order: 2
---


# Data

Here we introduce basic usage for annotated data generation in the context of training a process-supervision reward model. We resort to the method proposed by the paper [*Improve Mathematical Reasoning in Language Models by Automated Process Supervision*](https://arxiv.org/pdf/2406.06592) and implement in a fairly simple way (Data Acquisition section in the report). 

## Prerequisites

- **System Requirements**: Same as the root requirements

- **Dependencies**: (a separate more detailed dependenies list?)

- **Raw data**: this is the data you want to generate annotation for. In this repo, we provide an example dataset named `extracted_problems_and_answers.json`. (TODO: Description of the data? Copy the report here?). A snapshot will look like:

```json
[
    {
        "problem": "A board game spinner is divided into three parts labeled $A$, $B$  and $C$. The probability of the spinner landing on $A$ is $\\frac{1}{3}$ and the probability of the spinner landing on $B$ is $\\frac{5}{12}$.  What is the probability of the spinner landing on $C$? Express your answer as a common fraction.",
        "final_answer": "\\frac{1}{4}"
    },
    {
        "problem": "My school's math club has 6 boys and 8 girls.  I need to select a team to send to the state math competition.  We want 6 people on the team.  In how many ways can I select the team without restrictions?",
        "final_answer": "3003"
    },
    ...
]

```

## Runnning the Scripts

You can simply generate annotated data by running the following command:

```bash
python gen_data.py
```

it will load the data file and use a LLM to generate multiple response, followed by computation of Monte-Carlo estimate through binary search.

### Workflow

The workflow goes like this, for each piece of data:

1. Generate multiple outputs:
```python
root = State(problem, "", final_answer)
max_roll_num = 20
rollouts, corrs = getrollouts(root, max_roll_num)
```
where the querries and answers are wrapped as state in a tree, and the `getrollouts()` function will do rollouts and answer checking.

2. Compute MC estimate through binary search:
```python
mcst = cal_mc_bs(root)
```
```python
# Binary search for MC estimate
def cal_mc_bs(s, bs = 5):
    n = len(s.rollouts)
    subn = max(1,random.randint(n//2, n))
    mc = 0
    for i in range(bs):
        corr = 0
        sub = random.sample(s.rollouts, subn)
        for r in sub:
            if check_answer(s.a, r):
                corr += 1
        mc += corr * 1.0 / len(sub)
    return mc / bs 
```

3. If the computation indicate values, annotate the data through MCTS and save to local path:
```python
if sum(corrs) > 0 and sum(corrs) < max_roll_num: 
    print("Process annotation ...\n")
    filename = str(i+1) +'_states_list.json'
    process_annotation(problem, final_answer, states, filename)
```


## Customization

For customization you can

- Use your own dataset: you can replace `extracted_problems_and_answers.json` with your own dataset.

- Use other LLMs for rollout generation: replace `checkpoints` in function [`complete_answer()`](https://github.com/openreasoner/o1-dev/blob/7e1e42857ac0d5fce804181ca8dceed5f6c28f7d/data/utils.py#L13)

- Use more advanced search method to replace naive binary search