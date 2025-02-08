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
  
<h1 align="center" style="font-size: 30px;"><strong><em>OpenR</em></strong>: 大規模言語モデルによる高度な推論のためのオープンソースフレームワーク</h1>
<p align="center">
    <a href="https://arxiv.org/abs/2410.09671">論文</a>
    ·
    <a href="https://github.com/openreasoner/openr/blob/main/reports/Tutorial-LLM-Reasoning-Wang.pdf">チュートリアル</a>
    ·
    <a href="https://github.com/openreasoner/openr">コード</a>
    ·
    <a href="https://openreasoner.github.io/">ドキュメント</a>
    ·
    <a href="https://huggingface.co/datasets/openreasoner/MATH-APS">データ</a>
    ·
    <a href="https://huggingface.co/openreasoner/Math-psa">モデル</a>
    ·
    <a href="https://github.com/openreasoner/openr/issues">問題</a>
  </p>
    <p align="center">
     [ <a href="https://github.com/openreasoner/openr/blob/main/README.md">English</a> ][ <a href="https://github.com/openreasoner/openr/blob/main/README_zh.md">中文</a> ][ <a href="https://github.com/openreasoner/openr/blob/main/README_ja.md">日本語</a> ]
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
  <summary><span style="font-size: 1.5em;"><strong>目次</strong> 📖 </span></summary>
  <ol>
    <li><a href="#ニュースと更新">ニュースと更新</a></li>
    <li><a href="#機能">機能</a></li>
    <li><a href="#プロット">プロット</a></li>
    <li><a href="#提供されるデータセットとモデル">提供されるデータセットとモデル</a></li>
    <li>
      <a href="#始めに">始めに</a>
      <ul>
        <li><a href="#インストール">インストール</a></li>
        <li><a href="#クイックスタート">クイックスタート</a></li>
      </ul>
    </li>
    <li><a href="#使用法">使用法</a></li>
    <li><a href="#参加">参加</a></li>
    <li><a href="#連絡先">連絡先</a></li>
    <li><a href="#応答例">応答例</a></li>
    <li><a href="#コミュニティ">コミュニティ</a></li>
    <li><a href="#参考文献">参考文献</a></li>
  </ol>

</details>

<!-- News and Updates -->

## ニュースと更新
- **[2024年10月24日]** ***OpenR*** は **MCTS** 推論をサポートしました ([#24](https://github.com/openreasoner/openr/pull/24))! 🌲
- **[2024年10月15日]** 私たちのレポートが [**Arxiv**](https://arxiv.org/abs/2410.09671) に掲載されました! 
- **[2024年10月12日]** ***OpenR*** がリリースされました! 🚀 


## 機能

<p align="center">
  <img src="./figure/logo_text.png" alt="Description" style="width: 300px; margin-left: 50px; float: right;">
</p>

<div style="display: flex; align-items: center;">
<ul style="list-style-type: none; padding: 0;">
    <li><strong>✅ プロセス監督データ生成</strong></li>
    <li><strong>✅ オンラインポリシートレーニング</strong></li>
    <li><strong>✅ 生成的および識別的PRMトレーニング</strong></li>
    <li><strong>✅ 複数の検索戦略</strong></li>
    <li><strong>✅ テスト時の計算とスケーリング法則</strong></li>
</ul>
</div>

## プロット

<p align="center">
  <img src="./figure/compare_prm_by_boN.png" alt="PRM_Results" width="45%" />
  <img src="./figure/MATH_subsampled.png" alt="Inference_Results" width="45%" />
</p>

## 提供されるデータセットとモデル

[//]: # ([PRM800K]&#40;https://github.com/openai/prm800k&#41; &#40;Process Supervision Dataset&#41;)

[MATH-APS](https://huggingface.co/datasets/mengfang/MATH-APS) (私たちのデータセット)

[MATH-psa](https://huggingface.co/openreasoner/Math-psa) (私たちのプロセス報酬モデル)

## 始めに


### インストール

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


### ベースモデルのダウンロード

プロジェクトを実行する前に、必要なベースモデルがすべてダウンロードされていることを確認してください。このプロジェクトで使用されるモデルには以下が含まれます：

- `Qwen2.5-Math-1.5B-Instruct`, `Qwen2.5-Math-7B-Instruct`
- `peiyi9979/mistral-7b-sft`
- `peiyi9979/math-shepherd-mistral-7b-prm`

これらのモデルをダウンロードするには、[Hugging Faceモデルダウンロードチュートリアル](https://huggingface.co/docs/hub/models-downloading)を参照してください。

プロジェクトの設定に従って、すべてのモデルが各ディレクトリに保存されていることを確認してください。


### クイックスタート

推論を実行する前に、`reason/llm_service/`ディレクトリ内のスクリプトで以下の変数を変更して、使用するベースモデルを設定してください：

- `$MODEL_BASE`: モデルが保存されているディレクトリのパスを設定します。
- `$POLICY_MODEL_NAME`: 使用するポリシーモデルの名前を設定します。
- `$VALUE_MODEL_NAME`: 使用するバリューモデルの名前を設定します。
- `$NUM_LM_WORKER`: 起動する言語モデル（LM）ワーカーの数を設定します。
- `$NUM_RM_WORKER`: 起動する報酬モデル（RM）ワーカーの数を設定します。

次に、異なる技術を使用して推論を実行します。

#### LM & RM サービスの開始

例えば、Math ShepherdモデルのLMとRMサービスを開始するには、以下のコマンドを実行します：



```bash
sh reason/llm_service/create_service_math_shepherd.sh
```

サーバープロセスを終了するには、以下のコマンドを使用することをお勧めします：
```bash
tmux kill-session -t {Your Session Name} # デフォルトは`FastChat`
```

## 使用法

#### 推論の実行


⚠️ スクリプト内の入力パラメータ（`--LM`, `--RM`）が、保留中のワーカー内の変数（`$POLICY_MODEL_NAME`, `$VALUE_MODEL_NAME`）と一致していることを確認してください！



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

#### トレーニングの実行

⚠️ トレーニングを実行する前に、`train/mat/scripts/train_llm.sh`ファイル内の`$dataset_path`, `$model_name_or_path`および`$prm_name_or_path`を変更してください。

```bash
cd train/mat/scripts
bash train_llm.sh
```

#### PRM学習の実行

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

## 参加

> すべての貢献はコミュニティにとって価値があります。

***OpenR*** にご関心をお寄せいただきありがとうございます！🥰 私たちはオープンソースコミュニティに深くコミットしており、皆さんの貢献を歓迎します。あなたの努力は大小にかかわらず、私たちの成長と改善に役立ちます。貢献はコードに限らず、質問に答えたり、他の人を助けたり、ドキュメントを改善したり、プロジェクトを共有したりすることも同様に影響力があります。

[貢献ガイド](CONTRIBUTING.md) をご覧ください！ 

### 将来の計画

- RLトレーニングと検索戦略に関するより包括的な評価を追加

- Prove-Verifierモデルのサイズを拡大

- 自己改善トレーニングのサポート

<!-- CONTACT -->

## 連絡先

***OpenR*** コミュニティは以下のチームによって維持されています：

- **Openreasoner Team** (openreasoner@gmail.com)

## ライセンス

***OpenR*** はMITライセンスの下でリリースされています。

## 引用

私たちのリソースが役立つと感じた場合は、私たちの論文を引用してください：

```
@article{wang2024openr,
  title={OpenR: An Open Source Framework for Advanced Reasoning with Large Language Models},
  author={Wang, Jun and Fang, Meng and Wan, Ziyu and Wen, Muning and Zhu, Jiachen and Liu, Anjie and Gong, Ziqin and Song, Yan and Chen, Lei and Ni, Lionel M and others},
  journal={arXiv preprint arXiv:2410.09671},
  year={2024}
}
```
ありがとうございます！

## 応答例

### PRMの比較、Math-psa（私たちのもの）対Math-Shepherd 

<p align="center">
  <img src="./figure/QA/QA1.png" alt="QA 1" width="49%" />
  <img src="./figure/QA/QA2.png" alt="QA 2" width="49%" />
</p>


### RLトレーニングの正当化

<p align="center">
  <img src="./figure/QA/QA3.png" alt="QA 3" width="49%" />
  <img src="./figure/QA/QA4.png" alt="QA 4" width="49%" />
</p>

### テスト時の計算の探索

<p align="center">
  <img src="./figure/QA/QA5.png" alt="QA 5" width="70%" />
  <img src="./figure/QA/QA6.png" alt="QA 6" width="70%" />
  <img src="./figure/QA/QA7.png" alt="QA 7" width="70%" />
</p>


## コミュニティ

**WeChat**:

<img src="./figure/wechat_qrcode.jpg" width="30%" />



## 参考文献

### 推論時の計算
[1] [Alphazero-like tree-search can guide large language model decoding and training.](https://arxiv.org/pdf/2309.17179)

[2] [Reasoning with language model is planning with world model.](https://arxiv.org/pdf/2305.14992)

[3] [Scaling LLM test-time compute optimally can be more effective than scaling model parameters](https://arxiv.org/pdf/2408.03314?)

[4] [Think before you speak: Training language models with pause tokens](https://arxiv.org/pdf/2310.02226)


### 結果監督からプロセス監督へ

[1] [Training verifiers to solve math word problems](https://arxiv.org/pdf/2110.14168)

[2] [Solving math word problems with process-and outcome-based feedback](https://arxiv.org/pdf/2211.14275)

[3] [Let’s verify step by step](https://arxiv.org/pdf/2305.20050)

[4] [Making large language models better reasoners with step-aware verifier](https://arxiv.org/pdf/2206.02336)

[5] [Ovm, outcome-supervised value models for planning in
mathematical reasoning](https://aclanthology.org/2024.findings-naacl.55.pdf)

[6] [Generative verifiers: Reward modeling as next-token prediction](https://arxiv.org/pdf/2408.15240)

### データ取得

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
