<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->

<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
<!-- 
***[![MIT License][license-shield]][license-url]
-->

<!-- PROJECT LOGO -->

<br />
<div align="center">
  <a href="https://github.com/openreasoner/o1-dev/">
    <img src="figure/logo.png" alt="Logo" width="400">
  </a>

<!-- <h3 align="center">OpenR</h3> -->

<p align="center">
    <strong>OpenR</strong>: OpenR: An Open-Sourced Framework for Advancing Reasoning in Large Language Models
    <!-- <br />
    <a href="https://openreasoner.github.io/"><strong>Explore the docs »</strong></a>
    <br /> -->
    <br />
    <a href="https://arxiv.org/abs/xxxxx">Paper</a>
    ·
    <a href="https://colab.research.google.com/XXXXX">Demo</a>
    ·
    <a href="https://iamlilaj.github.io/OpenR-docs/">Docs</a>
    ·
    <a href="https://github.com/openreasoner/o1-dev/issues">Issue</a>
    ·
    <a href="https://medium.com/p/xxxxxx">Blog (Pytorch)</a>
    ·
    <a href="https://nips.cc/virtual/xxxxx">Video</a>
  </p>
</div>


<!-- TABLE OF CONTENTS -->

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#news-and-updates">News and Updates</a></li>
    <li><a href="#intro">Introduction</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#results">Benchmark Results</a></li>
    <li><a href="#model-zoo">Model Zoo</a></li>
    <li><a href="#contributing">Community</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- News and Updates -->

## News and Updates

- [10/06/2024] OpenR has been released!
  
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

Run Training

Before training, please modify the dataset_path, model_name_or_path and prm_name_or_path in train/mat/scripts/train_llm.sh.

```bash
cd train/mat/scripts
bash train_llm.sh
```

## Two-week plan ?

- Inference (mcts-like)

- Data + PRM

- Training (RL-like) for LLM

## References

[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)

[Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)
