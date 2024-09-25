from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
import json
import numpy as np
from tsllm.distributed.utils import print_with_rank
from tsllm.offline_rl.utils import load_jsonl
from .prompt import PROBLEM_FORMAT_STR, SEP
from .env import RLHF_TokenEnv


'''
def get_train_test_dataset(*args, **kwargs):
    train_dataset = build_dataset(dataset_name = "imdb", setting = "train", **kwargs)
    test_dataset = build_dataset(dataset_name = "imdb", setting = "test", **kwargs)
    return train_dataset, test_dataset

def build_dataset(model_name, dataset_name="imdb", setting="train", input_text_length=5):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = load_dataset(dataset_name, split=setting)
    ds = ds.rename_columns({"text": "review"})
    #ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    #input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        input_ids = tokenizer.encode(sample["review"])[: input_text_length]
        sample["question"] = tokenizer.decode(input_ids)
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds
'''


def get_train_test_dataset(*args, **kwargs):
    if "num_train_data" in kwargs.keys():
        num_train_data = kwargs.pop("num_train_data")
        if "path" in kwargs.keys():
            train_dataset = load_dataset(**kwargs, split=f"train[:{num_train_data}]")
            test_dataset = load_dataset(**kwargs, split=f"train[{num_train_data}:]")
        else:
            full_dataset = load_from_disk(**kwargs)["train"]
            train_dataset = full_dataset.select([i for i in range(num_train_data)])
            test_dataset = full_dataset.select(
                [i for i in range(num_train_data, len(full_dataset))]
            )
    else:
        train_data_pre = kwargs.pop("train_data_pre")
        train_data_post = kwargs.pop("train_data_post")
        full_dataset = load_from_disk(**kwargs)["train"]
        train_dataset = full_dataset.select(
            [i for i in range(train_data_pre, train_data_post)]
        )
        test_dataset = full_dataset.select(
            [i for i in range(train_data_pre, train_data_post)]
        )
    train_dataset = train_dataset.rename_column("prompt", "question")
    test_dataset = test_dataset.rename_column("prompt", "question")
    return train_dataset, test_dataset


def build_offline_data_component(path, q2idx_dict, tokenizer, sep):
    def get_value_index(question, answer):
        pre_state_token_length = len(tokenizer.encode(question + sep))
        index = [pre_state_token_length]
        if not sep == "":
            answer_list = answer.split(sep)
            for action in answer_list:
                action_length = len(
                    tokenizer.encode(action + sep, add_special_tokens=False)
                )
                index.append(action_length)
                if action_length == 0:
                    print_with_rank(
                        "possbile problems met in online value instance building. {}".format(
                            action
                        )
                    )
        else:
            answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
            for token in answer_tokens:
                index.append(1)

        index = np.cumsum(index) - 1
        return index

    predata = load_jsonl(path)
    traj_dict_list = []
    for idx, d in enumerate(predata):
        question = d["question"]
        if question in q2idx_dict.keys():
            task_idx = q2idx_dict[question]
            full_answer_list = d["answer"]
            # deduplication
            # note that in Dahoas/synthetic-instruct-gptj-pairwise, these exists several questions that are the same
            # but here the deduplication only happens for the answer list given one question
            # So it is still possible to have sample example when add traj
            unique_answer_list = list(
                {item["text"]: item for item in full_answer_list}.values()
            )
            answer_list = []
            for a in unique_answer_list:
                answer = a["text"]
                query_str = RLHF_TokenEnv.build_query_str(
                    cot_task_desc=None,
                    cot_examples=None,
                    problem_format_str=PROBLEM_FORMAT_STR,
                    problem_input=question,
                    sep=SEP,
                    is_few_shot=False,
                )
                value_index = get_value_index(query_str, answer)
                # :-1 is value index, -1 is the reward index
                reward_list = np.zeros(len(value_index) - 1)
                reward_list[-1] = a["reward"]
                traj_dict = {
                    "idx": task_idx,
                    "query_str": query_str,
                    "answer": answer,
                    "value_index": value_index,
                    "reward_list": reward_list,
                }
                traj_dict_list.append(traj_dict)
                answer_list.append(answer)

    return traj_dict_list


# def build_query_str(problem_input, config=None):
#     from .prompt import PROBLEM_FORMAT_STR
#     return PROBLEM_FORMAT_STR.format(problem_input)
