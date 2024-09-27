import os

from pathlib import Path
from typing import List, Optional
import torch.distributed as dist
from distributed.utils import print_with_rank, init_distributed, gather_scalar
from transformers import AutoTokenizer, pipeline
import torch
from functools import partial
from envs import get_env_datasets, get_default_query_str_builder
from envs.rlhf.prompt import PROBLEM_FORMAT_STR
from inference.trajectory_collector import _mcts_rollout_v1, _mcts_rollout_v2
from inference.value import value_fn
import json
from llm.ct2_utils import load_ct2_model
from mcts.utils import get_root
from model import load_critic_model
from model.modeling_actor_critic import AutoModelForCausalLMWithValueHead
import torch
import jsonlines
from llm.text_generation import llm_gen_ct2
import time
import numpy as np
from tqdm import tqdm
from mcts.tree import MCTS
from dataclasses import dataclass
from argparse import ArgumentParser
import ctranslate2
from inference.value import value_fn_rlhf

import time
import importlib

@dataclass
class SearchArgs:
    temperature: float = 1.0 # sampling temperature
    num_mcts_aggregation: int = 5 # how many trajectories to sample
    max_length: int = 8 # max depth of tree
    pb_c_init: float = 10 # for mcts exploration

    # init_critic_value: bool = True # whether we use value function to initialize the tree node value
    rollout_method: str = None # which tree-search method we use
    # mask_no_terminal_reward_value: bool = False # whether we mask the non-terminal value
    prune_node_under_v: Optional[float] = None

    num_simulations: int = 10 # parameter for mcts-alpha

    # aggregation parameters
    reset_total_tree: bool = False # intra-tree
    clear_total_tree: bool = False # inter-tree
    mcts_sample: bool = False # whether to use sample in mcts-alpha

    max_simulation: Optional[int] = None # hyperparameter for mcts-alpha
    max_token: Optional[int] = None # hyperparameter for mct-rollout

    # DFS hyper parameter
    prune_ratio: Optional[float] = 0.7
    prune_value: Optional[float] = None

    # COT-SC-Tree equals to RAP with select_by_prior=True
    select_by_prior: bool = False

    # COT-SC number
    k_maj: int = 100
    max_new_tokens: int = 64


input_prompt_format = PROBLEM_FORMAT_STR

if __name__ == "__main__":
    TEST_NO_TERMINAL = int(os.getenv("TEST_NO_TERMINAL", 0))
    TEST_WITH_TERMINAL = int(os.getenv("TEST_WITH_TERMINAL", 0))
    TEST_COT_GREEDY = int(os.getenv("TEST_COT_GREEDY", 0))
    TEST_COT_SC = int(os.getenv("TEST_COT_SC", 0))
    assert TEST_NO_TERMINAL + TEST_WITH_TERMINAL + TEST_COT_SC + TEST_COT_GREEDY > 0

    parser = ArgumentParser()
    parser.add_argument("--ct2_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--critic_model_path",type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="Dahoas/synthetic-instruct-gptj-pairwise")
    parser.add_argument("--train", action='store_true', default=False)
    
    config = parser.parse_args()

    args_list = [
        {
            "temperature": 1.0,
            "max_length": 64,
            "pb_c_init": 3,
            "num_simulations": 10,
            "k_maj": 5,
            "num_mcts_aggregation": 1,
            "max_simulation": None,
            "max_token": 5000,
            "reset_total_tree": False,
            "clear_total_tree": True,
            "rollout_method": "mcts.get_next_action",
            "select_by_prior": False,
            "max_new_tokens": 64,
            "mcts_sample": False,
            "prune_ratio": 0.9,
            "prune_value": None
        }
    ]

    task_module = importlib.import_module(f"tsllm.envs.{config.env_name}")

    save_dir = Path(config.save_dir) / config.env_name

    local_rank, world_size = init_distributed()

    if config.env_name == "rlhf":
        task_dataset_kwargs = {"path": config.dataset_name, 'num_train_data': 30000}
        #assert config.train
        #task_dataset_kwargs = {"path": config.dataset_name, 'train_data_pre': 22500, 'train_data_post':30000}
    else:
        task_dataset_kwargs = {}

    train_ds, test_ds = get_env_datasets(config.env_name, **task_dataset_kwargs)
    if config.train:
        print_with_rank('Use training set')
        test_ds = train_ds

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    tokenizer.eos_token = "### End"

    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
    reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
    rank_model, reward_tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)
    rank_model = rank_model.bfloat16().to(f"cuda:{local_rank}")

    @torch.inference_mode()
    def reward_fn(question, answer):    
        inputs = reward_tokenizer(question, answer, return_tensors='pt').to(f"cuda:{local_rank}")
        score = rank_model(**inputs).logits[0].cpu().item()
        return score
    
    if not config.critic_model_path == 'None':
        critic = AutoModelForCausalLMWithValueHead.from_pretrained(config.critic_model_path).to(f"cuda:{local_rank}").bfloat16()
        policy_forward_value = partial(value_fn_rlhf, critic, tokenizer)
    else:
        critic = None
        policy_forward_value = None

    ############ CONVERT MODEL to CT2 files ###################

    dist.barrier()

    # PS：如果convert好了，上面两步都可以跳过

    ################ LOAD CT2 model ####################
    # ct2_generator = ctranslate2.Generator(ct2_dir,
    #                                       )
    # ct2_sp = spm.SentencePieceProcessor(os.path.join(ct2_dir, "tokenizer.model"))
    ct2_generator = ctranslate2.Generator(
        config.ct2_dir, device="cuda", device_index=local_rank, compute_type="float32"
    )

    def prompt_fn(problem_input: str):
        return input_prompt_format.format(question=problem_input)

    def cot_direct_output(args, problem_inst, stop, **kwargs):
        prompt = prompt_fn(problem_inst["question"])
        texts, logps = llm_gen_ct2(
            ct2_generator,
            tokenizer,
            static_prompt=None,
            prompt=prompt,
            num_sequence=1,
            stop=tokenizer.eos_token_id,
            **kwargs,
        )
        reward_list = [reward_fn(prompt, txt) for txt in texts]
        return reward_list[0], list(zip(texts, reward_list))

    def cot_sc_output(args, problem_inst, stop, **kwargs):
        prompt = prompt_fn(problem_inst["question"])
        texts, logps = llm_gen_ct2(
            ct2_generator,
            tokenizer,
            static_prompt=None,
            prompt=prompt,
            num_sequence=args.k_maj,
            stop=tokenizer.eos_token_id,
            max_batch_size=50,
            **kwargs,
        )
        reward_list = [reward_fn(prompt, txt) for txt in texts]
        judge_results = max(reward_list)
        return judge_results, list(zip(texts, reward_list))

    def llm_forward_fn():
        from llm.text_generation import llm_forward_ct2
        llm_gen_v2 = partial(llm_forward_ct2, ct2_generator, tokenizer)
        return llm_gen_v2
    
    def mcts_multi_search(args: "SearchArgs", problem_inst, no_terminal_reward=True):
        env = task_module.Env(
            config={
                "max_actions": 50,
                "sep": "",
                "max_length": args.max_length,
                "temperature": args.temperature,
            },
            problems=[
                {
                    "question": problem_inst["question"],
                }
            ],
            llm_forward_fn=llm_forward_fn(),
            tokenizer=tokenizer,
            reward_fn=reward_fn,
        )
        # llm_gen_fn=partial(llm_gen_with_logp_v1, model, tokenizer),
        mcts = MCTS(
            cfg={
                "num_simulations": args.num_simulations,
                "pb_c_base": 19652,
                "pb_c_init": args.pb_c_init,
                "root_dirichlet_alpha": 0.3,
                "root_noise_weight": 0.25,
                "no_terminal_reward": no_terminal_reward,
            }
        )

        if args.rollout_method == "mcts.rollout":
            assert args.max_token is not None and args.max_simulation is None
            output_list, _, _ = mcts.rollout(
                env,
                args.num_mcts_aggregation,
                policy_forward_value,
                max_num_simulation=args.max_simulation,
                max_token=args.max_token,
                return_tree=True,
            )
        elif args.rollout_method == "mcts.get_next_action":
            output_list = _mcts_rollout_v1(
                mcts,
                env,
                policy_forward_value,
                args.num_mcts_aggregation,
                args.reset_total_tree,
                sample=args.mcts_sample,
                clear_total_tree=args.clear_total_tree
            )
            prompt = prompt_fn(problem_inst["question"])
            # if len(texts) > 0:
            #     value_list = policy_forward_value(
            #         [prompt + txt + env.sep for txt in texts]
            #     ).tolist()
            # else:
            #     value_list = []
        elif args.rollout_method == "mcts.rap":
            output_list = mcts.rap(
                env,
                args.num_mcts_aggregation,
                policy_forward_value,
                args.select_by_prior,
            )
        elif args.rollout_method == "mcts.beam_search":
            output_list = mcts.beam_search(
                env, args.num_mcts_aggregation, args.max_length, policy_forward_value
            )
        elif args.rollout_method == "mcts.dfs":
            # here the num_mcts_aggregation is the step_limit which indicate how many nodes 
            # will be visited in the tree.
            output_list = mcts.dfs(
                env, 
                args.num_mcts_aggregation, 
                policy_forward_value, 
                prune_value=args.prune_value,
                prune_ratio=args.prune_ratio
            )
        else:
            raise ValueError("Unknow rollout method: {}".format(args.rollout_method))
        texts = [o["text"] for o in output_list]
        prompt = prompt_fn(problem_inst["question"])
        reward_list = [reward_fn(prompt, txt) for txt in texts]
        judge_results = {}
        if len(reward_list) == 0:
            # default reward as -1
            assert args.rollout_method == "mcts.rollout"
            reward_list = [-1] 
        judge_results[f"{args.rollout_method}@agg_{args.num_mcts_aggregation}"] = max(reward_list)

        if output_list and args.rollout_method != "mcts.rollout":
            num_token = output_list[-1]["num_generated_token"]
        else:
            num_token = mcts.num_generated_token
        judge_results["#token"] = num_token
        return mcts, judge_results, output_list
    
    def test_problem(
        args,
        idx,
        problem_inst,
        cot_writer,
        cot_sc_writer,
        mcts_no_term_writer,
        mcts_w_term_writer,
    ):
        results = {}

        def save_fn(writer, output, result: dict):
            if writer is not None:
                writer.write(
                    {
                        "i": idx,
                        "prompt": problem_inst["question"],
                        "output": output,
                        "result": result
                    }
                )

        if TEST_COT_GREEDY:
            r_cot_greedy, cot_episodes = cot_direct_output(
                args,
                problem_inst,
                stop=tokenizer.eos_token_id,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=1,
            )
            save_fn(cot_writer, cot_episodes, r_cot_greedy)
            results["cot-greedy"] = r_cot_greedy

        if TEST_WITH_TERMINAL:
            mcts, r_with_terminal, with_terminal_episodes = mcts_multi_search(
                args, problem_inst, False
            )
            # save_tree_path = save_dir / f"tmp_tree"
            # if not save_tree_path.exists():
            #     save_tree_path.mkdir(parents=True)
            # json.dump(
            #     get_root(mcts.root).to_json(),
            #     open(save_tree_path / f"{idx}_r{r_with_terminal}.json", "w"),
            #     indent=2,
            # )
            save_fn(mcts_w_term_writer, with_terminal_episodes, r_with_terminal)
            results["w/-terminal"] = r_with_terminal

        if TEST_COT_SC:
            r_cot_sc, cot_sc_episodes = cot_sc_output(
                args,
                problem_inst,
                stop=tokenizer.eos_token_id,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                top_k=50,
            )
            save_fn(cot_sc_writer, cot_sc_episodes, r_cot_sc)
            results["cot-sc"] = r_cot_sc
        return results

    def _result_str(results, cnt, join_str="\n"):
        res = ""
        for k, v in results.items():
            if isinstance(v, float):
                res += f"{k}: {v/cnt}"
            elif isinstance(v, dict):
                res += f"{k}: "
                res += ", ".join(
                    [
                        (
                            f"{sub_k}: {sub_v/cnt:.2f}"
                            if sub_k == "#token"
                            else f"{sub_k}: {sub_v/cnt}"
                        )
                        for sub_k, sub_v in v.items()
                    ]
                )
            else:
                raise ValueError
            res += join_str
        res += f"cnt: {cnt}"
        return res

    for i_arg, cur_args in enumerate(args_list):
        args = SearchArgs(**cur_args)
        if args.mcts_sample:
            print_with_rank('Use mcts sample mode, Make sure you are using code to generate rollout instead of test')

        writer_dir = save_dir / (f"args{i_arg}/")

        if local_rank == 0:
            print("Search args: {}".format(args))
            if not writer_dir.exists():
                writer_dir.mkdir(parents=True)
            json.dump(cur_args, open(writer_dir / "args.json", "w"))

        if TEST_COT_GREEDY:
            cot_save_path = writer_dir / "cot"
            if local_rank == 0 and not cot_save_path.exists():
                cot_save_path.mkdir(parents=True)
            dist.barrier()
            cot_writer = jsonlines.open(cot_save_path / f"{local_rank}.jsonl", "a")
        else:
            cot_writer = None
        if TEST_COT_SC:
            cot_sc_save_path = writer_dir / "cot_sc"
            if local_rank == 0 and not cot_sc_save_path.exists():
                cot_sc_save_path.mkdir(parents=True)
            dist.barrier()
            cot_sc_writer = jsonlines.open(
                cot_sc_save_path / f"{local_rank}.jsonl", "a"
            )
        else:
            cot_sc_writer = None

        if TEST_NO_TERMINAL:
            mcts_no_term_save_path = writer_dir / "no_terminal_reward"
            if local_rank == 0 and not mcts_no_term_save_path.exists():
                mcts_no_term_save_path.mkdir(parents=True)
            dist.barrier()
            mcts_no_term_writer = jsonlines.open(
                mcts_no_term_save_path / f"{local_rank}.jsonl", "a"
            )
        else:
            mcts_no_term_writer = None

        if TEST_WITH_TERMINAL:
            mcts_w_term_save_path = writer_dir / "with_terminal_reward"
            if local_rank == 0 and not mcts_w_term_save_path.exists():
                mcts_w_term_save_path.mkdir(parents=True)
            dist.barrier()
            mcts_w_term_writer = jsonlines.open(
                mcts_w_term_save_path / f"{local_rank}.jsonl", "a"
            )
        else:
            mcts_w_term_writer = None

        cnt = 0
        reward_dict = dict()
        t0 = time.time()
        for i in (pbar := tqdm(range(len(test_ds)), disable=(local_rank != 0))):
            if i % world_size == local_rank:
                results = test_problem(
                    args,
                    i,
                    test_ds[i],
                    cot_writer,
                    cot_sc_writer,
                    mcts_no_term_writer,
                    mcts_w_term_writer,
                )
                for k, v in results.items():
                    if isinstance(v, float):
                        if k not in reward_dict:
                            reward_dict[k] = 0
                        reward_dict[k] += v
                    elif isinstance(v, dict):
                        if k not in reward_dict:
                            reward_dict[k] = dict()
                        for sub_k, sub_v in v.items():
                            if sub_k not in reward_dict[k]:
                                reward_dict[k][sub_k] = 0

                            reward_dict[k][sub_k] += sub_v
                    else:
                        raise NotImplementedError
                cnt += 1
                results_strs = _result_str(reward_dict, cnt, join_str="; ")
                pbar.set_description(results_strs)

        print_with_rank(results_strs)

        cnt_list = gather_scalar(cnt, local_rank, world_size)

        gathered_results = {}
        for k, v in reward_dict.items():
            if isinstance(v, float):
                gathered_list = gather_scalar(v, local_rank, world_size)
                if local_rank == 0:
                    gathered_results[k] = sum(gathered_list)
            elif isinstance(v, dict):
                gathered_results[k] = {}
                for sub_k, sub_v in v.items():
                    gathered_list = gather_scalar(float(sub_v), local_rank, world_size)
                    if local_rank == 0:
                        gathered_results[k][sub_k] = sum(gathered_list)
            else:
                raise ValueError

        if local_rank == 0:
            total_cnt = sum(cnt_list)
            t1 = time.time()
            total_results_strs = _result_str(gathered_results, total_cnt)

            print(cur_args)
            print("TOTAL RESULTS:\n", total_results_strs)
            print("Time: {}".format(t1 - t0))
