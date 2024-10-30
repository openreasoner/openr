## Re-implementing rStar MCTS

Original Paper: [Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers](https://arxiv.org/abs/2408.06195)
Original Repo: [https://github.com/zhentingqi/rStar](https://github.com/zhentingqi/rStar)

For the current version of rStar, we use a separated environment `rstar_env.py`. In the future,
we hope to rewrite it in a better way.

For inference, set up your LLM and RM, then run `scripts/rstar_mcts.sh`.
