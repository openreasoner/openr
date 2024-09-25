from typing import Dict, Optional, Union, Callable
import numpy as np
from tsllm.distributed.utils import print_with_rank
from tsllm.offline_rl.utils import load_jsonl
from transformers import PreTrainedTokenizer
from pathlib import Path
from torch.utils.data import Dataset
import jsonlines


def build_sft_data_component(
    jsonl_path: Union[Path, str],
    q2idx_dict: Dict,
    tokenizer: PreTrainedTokenizer,
    add_eos_token: bool,
    is_few_shot: bool,
    build_query_str_fn: Callable,
    build_response_str_fn: Callable,
    sep: str,
    cot_task_desc_str: Optional[str] = None,
    cot_example_str: Optional[str] = None,
    problem_format_str: Optional[str] = None,
):
    predata = load_jsonl(jsonl_path)
    q_r_dict_list = []
    for idx, d in enumerate(predata):
        question = d["question"]
        if question not in q2idx_dict:
            continue
        task_idx = q2idx_dict[question]
        full_answer_list = d["answer"]
        query_str = build_query_str_fn(
            cot_task_desc=cot_task_desc_str,
            cot_examples=cot_example_str,
            problem_format_str=problem_format_str,
            problem_input=question,
            sep=sep,
            is_few_shot=is_few_shot,
        )

        for answer_output in full_answer_list:
            answer_txt = answer_output["text"]
            response_str = build_response_str_fn(answer_txt, tokenizer, add_eos_token)
            traj_dict = {
                "idx": task_idx,
                "query_str": query_str,
                "answer": answer_txt,
                "response_str": response_str,
            }
            q_r_dict_list.append(traj_dict)

    return q_r_dict_list


def build_critic_data_component(
    jsonl_path: Union[Path, str],
    q2idx_dict: Dict,
    tokenizer: PreTrainedTokenizer,
    sep: str,
    is_few_shot: bool,
    build_query_str_fn: Callable,
    cot_task_desc_str: Optional[str] = None,
    cot_example_str: Optional[str] = None,
    problem_format_str: Optional[str] = None,
):
    def get_value_index(q_str: str, answer_str: str):
        pre_state_token_length = len(tokenizer.encode(q_str))
        indices = [pre_state_token_length]
        if sep != "":
            answer_list = answer_str.split(sep)

            check_indices = [pre_state_token_length - 1]
            current_str = q_str
            for action in answer_list:
                current_str += action + sep
                if len(action) == 0:
                    print_with_rank(
                        "WARNING: possbile problems met in sft instance building. {}".format(
                            action
                        )
                    )
                    continue
                check_indices.append(len(tokenizer.encode(current_str)) - 1)
            check_indices = np.array(check_indices)
            indices = check_indices

        else:
            answer_tokens = tokenizer.encode(answer_str, add_special_tokens=False)
            for token in answer_tokens:
                indices.append(1)

            indices = np.cumsum(indices) - 1

        return indices

    predata = load_jsonl(jsonl_path)
    traj_dict_list = []
    for idx, d in enumerate(predata):
        question = d["question"]
        if question not in q2idx_dict.keys():
            continue
        task_idx = q2idx_dict[question]
        full_answer_list = d["answer"]
        query_str = build_query_str_fn(
            cot_task_desc=cot_task_desc_str,
            cot_examples=cot_example_str,
            problem_format_str=problem_format_str,
            problem_input=question,
            sep=sep,
            is_few_shot=is_few_shot,
        )
        for answer_output in full_answer_list:
            """answer_output is a dict with keys:
            "text", "reward",
            if there is not "reward" key, use "correct" key
            """
            if len(sep) > 1:
                print_with_rank("WARNING: sep is not empty, but {}".format(sep))
            answer = answer_output["text"].strip(sep)
            value_index = get_value_index(query_str, answer)
            # :-1 is value index, -1 is the reward index
            reward_list = np.zeros(len(value_index) - 1)
            if "reward" not in answer_output:
                answer_output["reward"] = 1.0 if answer_output["correct"] else -1.0

            reward_list[-1] = answer_output["reward"]
            traj_dict = {
                "idx": task_idx,
                "query_str": query_str,
                "answer": answer + sep,
                "value_index": value_index,
                "reward_list": reward_list,
            }
            traj_dict_list.append(traj_dict)
    return traj_dict_list
