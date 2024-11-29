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

<!-- 
***[![MIT License][license-shield]][license-url]
-->

<!-- PROJECT LOGO -->

<br />
<div align="center">
  <a href="https://github.com/openreasoner/openr/">
    <img src="figure/openr_logo.png" alt="Logo" width="200">
  </a>
  
<h1 align="center" style="font-size: 30px;"><strong><em>OpenR</em></strong>: 专注大型语言模型进阶推理能力的开源框架</h1>
<p align="center">
    <a href="https://arxiv.org/abs/2410.09671">技术报告</a>
    ·
    <a href="https://github.com/openreasoner/openr/blob/main/reports/Tutorial-LLM-Reasoning-Wang.pdf">指南</a>
    ·
    <a href="https://github.com/openreasoner/openr">代码库</a>
    ·
    <a href="https://openreasoner.github.io/">文档</a>
    ·
    <a href="https://huggingface.co/datasets/openreasoner/MATH-APS">数据集</a>
    ·
    <a href="https://huggingface.co/openreasoner/Math-psa">模型文件</a>
    ·
    <a href="https://github.com/openreasoner/openr/issues">问答</a>
    ·
    <a href="https://www.modelscope.cn/studios/modelscope/OpenR_Inference">推理</a>
  </p>
    <p align="center">
     [ <a href="https://github.com/openreasoner/openr/blob/main/README.md">English</a> ][ <a href="https://github.com/openreasoner/openr/blob/main/README_zh.md">中文</a> ]
    </p>
</div>

---
[![GitHub contributors](https://img.shields.io/github/contributors/openreasoner/openr)][contributors-url]
[![arXiv](https://img.shields.io/badge/ArXiv-2410.09671-b31b1b.svg)](https://arxiv.org/pdf/2410.09671)
![GitHub License](https://img.shields.io/github/license/openreasoner/openr)
[![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/openreasoner/openr)][issues-url]
[![GitHub forks](https://img.shields.io/github/forks/openreasoner/openr)][forks-url]
[![GitHub Repo stars](https://img.shields.io/github/stars/openreasoner/openr)][stars-url]
[![HuggingFace Dataset](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/openreasoner)
[![X](https://img.shields.io/badge/openreasoner-%23000000.svg?logo=X&logoColor=white)](https://x.com/openreasoner)
[![WeChat](https://img.shields.io/badge/WeChat_Group-07C160?logo=wechat&logoColor=white)](#community)


<!-- TABLE OF CONTENTS -->

<details>
  <summary><span style="font-size: 1.5em;"><strong>目录</strong> 📖 </span></summary>
  <ol>
    <li><a href="#新闻与更新">新闻与更新</a></li>
    <li><a href="#功能">功能</a></li>
    <li><a href="#图表">图表</a></li>
    <li><a href="#数据集与模型">数据集与模型</a></li>
    <li>
      <a href="#快速入门">快速入门</a>
      <ul>
        <li><a href="#安装">安装</a></li>
        <li><a href="#快速开始">快速开始</a></li>
      </ul>
    </li>
    <li><a href="#用法">用法</a></li>
    <li><a href="#加入我们">加入我们</a></li>
    <li><a href="#联系方式">联系方式</a></li>
    <li><a href="#问答示例">问答示例</a></li>
    <li><a href="#社区">社区</a></li>
    <li><a href="#参考引用">参考引用</a></li>
  </ol>

</details>

<!-- News and Updates -->

## 新闻与更新
- **[24/10/2024]** ***OpenR*** 现已支持 **MCTS** 推理 ([#24](https://github.com/openreasoner/openr/pull/24))! 🌲
- **[15/10/2024]** 我们的报告已发布在 [**Arxiv**](https://arxiv.org/abs/2410.09671) 上! 
- **[12/10/2024]** ***OpenR*** 已经发布！ 🚀 


## 功能

<p align="center">
  <img src="./figure/logo_text.png" alt="Description" style="width: 300px; margin-left: 50px; float: right;">
</p>

<div style="display: flex; align-items: center;">
<ul style="list-style-type: none; padding: 0;">
    <li><strong>✅ 过程监督的数据生成 </strong></li>
    <li><strong>✅ 在线策略训练 </strong></li>
    <li><strong>✅ Generative 和 Discriminative 过程奖励模型的训练</strong></li>
    <li><strong>✅ 多种搜索策略 </strong></li>
    <li><strong>✅ Test-time 计算和 Scaling Law</strong></li>
</ul>
</div>

## 图表

<p align="center">
  <img src="./figure/compare_prm_by_boN.png" alt="PRM_Results" width="45%" />
  <img src="./figure/MATH_subsampled.png" alt="Inference_Results" width="45%" />
</p>

## 数据集与模型

[//]: # ([PRM800K]&#40;https://github.com/openai/prm800k&#41; &#40;Process Supervision Dataset&#41;)

[MATH-APS](https://huggingface.co/datasets/mengfang/MATH-APS) (我们发布的数据集)

[MATH-psa](https://huggingface.co/openreasoner/Math-psa) (我们发布的过程奖励模型)

## 快速入门


### 安装

```
conda create -n open_reasoner python=3.10
conda activate open_reasoner
pip install -r requirements.txt
pip3 install  "fschat[model_worker,webui]"
pip install -U pydantic
cd envs/MATH/latex2sympy
pip install -e .
cd -
```


### 下载基座模型

在运行项目之前，请确保已下载所有所需的基础模型。本项目使用的模型包括：

- `Qwen2.5-Math-1.5B-Instruct`, `Qwen2.5-Math-7B-Instruct`
- `peiyi9979/mistral-7b-sft`
- `peiyi9979/math-shepherd-mistral-7b-prm`

Huggingface 具体下载方式可参考 [Huggingface 下载教程](https://huggingface.co/docs/hub/models-downloading)

在继续之前，请确保所有模型已根据项目设置保存在各自的目录中。


### 快速开始

在运行推理之前，请修改`reason/llm_service/`目录下脚本中的以下变量，以设置适合您使用的基座模型：

- `$MODEL_BASE`: 设置为存储模型的目录路径。
- `$POLICY_MODEL_NAME`: 设置为您希望使用的策略模型的名称。
- `$VALUE_MODEL_NAME`: 设置为您希望使用的Value模型的名称。
- `$NUM_LM_WORKER`: 设置为要启动的语言模型（LM）进程的数量
- `$NUM_RM_WORKER`: 设置为要启动的奖励模型（RM）进程的数量。

接下来，我们将使用不同的技术运行推理。

#### 启动 LM 和 RM 服务

例如，要启动 Math Shepherd 模型的 LM 和 RM 服务，请运行以下命令：



```bash
sh reason/llm_service/create_service_math_shepherd.sh
```

关闭服务进程可以参考以下命令:
```bash
tmux kill-session -t {Your Session Name} # default is `FastChat`
```

## 用法

#### 运行 推理(Inference)


⚠️ 确保脚本中的输入参数(`--LM`, `--RM`)与待运行的进程中的变量(`$POLICY_MODEL_NAME`, `$VALUE_MODEL_NAME`)保持一致！



```bash
export PYTHONPATH=$(pwd)
sh scripts/eval/cot_greedy.sh

# Method: cot. Average result: ({'majority_vote': 0.734, 'total_completion_tokens': 559.13},)

sh scripts/eval/cot_rerank.sh

# Method: best_of_n. Average result: ({'majority_vote': 0.782, 
#                                       'prm_min_max': 0.772, 
#                                       'prm_min_vote': 0.792, 
#                                       'prm_last_max': 0.776, 
#                                       'prm_last_vote': 0.792, 
#                                       'total_completion_tokens': 4431.268},)

sh scripts/eval/beam_search.sh

# Method: beam_search. Average result: ({'majority_vote': 0.74, 'total_completion_tokens': 2350.492},)

sh scripts/eval/vanila_mcts.sh

```

#### 运行 训练(Training)

⚠️ 运行训练之前，请修改 `train/mat/scripts/train_llm.sh` 文件中的 `$dataset_path`, `$model_name_or_path` 和 `$prm_name_or_path` 项。

```bash
cd train/mat/scripts
bash train_llm.sh
```

#### 运行 PRM学习

```bash
cd prm/code

\\ single gpu
python finetune_qwen_single_gpu.py --model_path $YOUR_MODEL_PATH \
                                   --train_data_path $TRAIN_DATA_PATH \
                                   --test_data_path $TEST_DATA_PATH


\\ multi gpu
torchrun --nproc_per_node=2 finetune_qwen.py --model_path $YOUR_MODEL_PATH \
                                             --data_path $YOUR_DATA_FOLDER_PATH \
                                             --datasets both \
```

## 加入我们

> 您的每一份贡献对社区来说都是宝贵的。

感谢您对 ***OpenR*** 的关注！🥰 我们致力于发展开源社区，并十分欢迎大家的contribution。无论大小，您的努力都将帮助我们成长和进步。贡献不仅限于代码——解答问题、帮助他人、改进我们的文档、分享项目同样具有深远的影响。

欢迎查阅 [贡献指南](CONTRIBUTING.md) ! 

### 未来计划

- 更全面的强化学习训练和搜索方法的实验

- 更大规模的Prove-Verifier模型

- 支持自我提升训练功能

<!-- CONTACT -->

## 联系方式

***OpenR*** 社区由以下团队维护：

- **Openreasoner Team** (openreasoner@gmail.com)

## License

***OpenR*** is released under the MIT License.

## 欢迎引用

如果您觉得我们的资源对您有帮助，请引用我们的论文：

```
@article{wang2024openr,
  title={OpenR: An Open Source Framework for Advanced Reasoning with Large Language Models},
  author={Wang, Jun and Fang, Meng and Wan, Ziyu and Wen, Muning and Zhu, Jiachen and Liu, Anjie and Gong, Ziqin and Song, Yan and Chen, Lei and Ni, Lionel M and others},
  journal={arXiv preprint arXiv:2410.09671},
  year={2024}
}
```
十分感谢！

## 问答示例

### 对比 过程奖励模型（PRM）：Math-psa (Ours) V.S. Math-Shepherd 

<p align="center">
  <img src="./figure/QA/QA1.png" alt="QA 1" width="49%" />
  <img src="./figure/QA/QA2.png" alt="QA 2" width="49%" />
</p>


### 验证强化学习训练 （RL Training）

<p align="center">
  <img src="./figure/QA/QA3.png" alt="QA 3" width="49%" />
  <img src="./figure/QA/QA4.png" alt="QA 4" width="49%" />
</p>

### 探索 Test-time Computation

<p align="center">
  <img src="./figure/QA/QA5.png" alt="QA 5" width="70%" />
  <img src="./figure/QA/QA6.png" alt="QA 6" width="70%" />
  <img src="./figure/QA/QA7.png" alt="QA 7" width="70%" />
</p>


## 社区

**微信群聊**:

<img src="./figure/wechat_qrcode.jpg" width="30%" />



## 参考引用

### Inference-time Computing
[1] [Alphazero-like tree-search can guide large language model decoding and training.](https://arxiv.org/pdf/2309.17179)

[2] [Reasoning with language model is planning with world model.](https://arxiv.org/pdf/2305.14992)

[3] [Scaling LLM test-time compute optimally can be more effective than scaling model parameters](https://arxiv.org/pdf/2408.03314?)

[4] [Think before you speak: Training language models with pause tokens](https://arxiv.org/pdf/2310.02226)


### From Outcome Supervision to Process Supervision

[1] [Training verifiers to solve math word problems](https://arxiv.org/pdf/2110.14168)

[2] [Solving math word problems with process-and outcome-based feedback](https://arxiv.org/pdf/2211.14275)

[3] [Let’s verify step by step](https://arxiv.org/pdf/2305.20050)

[4] [Making large language models better reasoners with step-aware verifier](https://arxiv.org/pdf/2206.02336)

[5] [Ovm, outcome-supervised value models for planning in
mathematical reasoning](https://aclanthology.org/2024.findings-naacl.55.pdf)

[6] [Generative verifiers: Reward modeling as next-token prediction](https://arxiv.org/pdf/2408.15240)

### Data Acquisition

[1] [Star: Bootstrapping reasoning with reasoning](https://proceedings.neurips.cc/paper_files/paper/2022/file/639a9a172c044fbb64175b5fad42e9a5-Paper-Conference.pdf)

[2] [Quiet-star: Language models can teach themselves to think before speaking](https://arxiv.org/pdf/2403.09629)

[3] [Improve mathematical reasoning in language models by automated
process supervision](https://arxiv.org/pdf/2406.06592)

[4] [Shepherd: A critic for language model generation](https://arxiv.org/abs/2308.04592)

[5] [Math-shepherd: Verify and reinforce llms step-by-step without human annotations](https://aclanthology.org/2024.acl-long.510.pdf)

<!-- MARKDOWN LINKS & IMAGES -->

<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/openreasoner/openr.svg?style=for-the-badge
[contributors-url]: https://github.com/openreasoner/openr/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/openreasoner/openr.svg?style=for-the-badge
[forks-url]: https://github.com/openreasoner/openr/network/members
[stars-shield]: https://img.shields.io/github/stars/openreasoner/openr.svg?style=for-the-badge
[stars-url]: https://github.com/openreasoner/openr/stargazers
[issues-shield]: https://img.shields.io/github/issues/openreasoner/openr.svg?style=for-the-badge
[issues-url]: https://github.com/openreasoner/openr/issues

[license-shield]: https://img.shields.io/github/license/openreasoner/openr.svg?style=for-the-badge
[license-url]: https://github.com/openreasoner/openr/blob/main/LICENSE.txt