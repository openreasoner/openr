---
title: Home
layout: home
nav_order: 1
---

# OpenReasoner


{: .no_toc }


OpenR: An Open-Sourced Framework for Advancing Reasoning in LLMs
{: .fs-6 .fw-300 }

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

<img src="../assets/images/logo.png" alt="Description" width="300" />

[Get started now](/docs/get-start/index.html){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View it on GitHub](https://github.com/openreasoner/o1-dev){: .btn .fs-5 .mb-4 .mb-md-0 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---


## News and Updates

<span style="color: #555; font-weight: bold;">[10/06/2024]</span> <span style="color: #007acc;">OpenR has been released!</span>


## Introduction

OpenAI o1 has demonstrated that leveraging reinforcement learning to inherently
integrate reasoning steps during inference can greatly improve a model’s reasoning
abilities. This is especially relevant as the field shifts from the traditional autoregressive approach to a more deliberate modeling of the slow-thinking process, achieved through step-by-step reasoning training. 

We attribute these enhanced reasoning capabilities to the combined use of **search**, **reinforcement learning**, and **process supervision** in LLMs. In this technical manual, we introduce OpenR, an open-source framework that integrates search, reinforcement learning, and process supervision to improve reasoning in LLMs. Our work is the first to provide an open-source framework demonstrating how the effective utilization of these techniques enables LLMs to achieve advanced reasoning capabilities, 

Similar to those showcased by OpenAI o1—such as improvements resulting from extended reasoning time. We also highlight the critical role of process reward models in enhancing reasoning performance. 

We evaluate OpenR on the MATH dataset using open-access datasets and search methodologies. Our experiments demonstrate significant improvements with increased test-time compute, along with enhanced reasoning abilities through the application of process reward models. Our code, models, and datasets are available at [xxx].

## About the Project

### Liscense


### Contributing




<!-- MARKDOWN LINKS & IMAGES -->

<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/openreasoner/o1-dev.svg?style=for-the-badge
[contributors-url]: https://github.com/openreasoner/o1-dev/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/openreasoner/o1-dev.svg?style=for-the-badge
[forks-url]: https://github.com/openreasoner/o1-dev/network/members
[stars-shield]: https://img.shields.io/github/stars/openreasoner/o1-dev.svg?style=for-the-badge
[stars-url]: https://github.com/openreasoner/o1-dev/stargazers
[issues-shield]: https://img.shields.io/github/issues/openreasoner/o1-dev.svg?style=for-the-badge
[issues-url]: https://github.com/openreasoner/o1-dev/issues

[license-shield]: https://img.shields.io/github/license/openreasoner/o1-dev.svg?style=for-the-badge
[license-url]: https://github.com/openreasoner/o1-dev/blob/main/LICENSE.txt