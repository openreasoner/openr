import json
from typing import Optional
import random
import numpy as np
import os
import torch
from dataclasses import dataclass
import jsonlines
from typing import Any, Callable, Dict, List, Optional, Union


def write_to_jsonl(data, output_file):
    cnt = 0
    with open(output_file, "w") as outfile:
        for item in data:
            outfile.write(json.dumps(item) + "\n")
            cnt += len(item["answer"])
        print("Write {} items into {}".format(cnt, output_file))


def load_jsonl(file_path):
    data_list = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())
            data_list.append(data)
    return data_list


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


def load_record_data(path: str) -> (List[Dict], int):
    """
    Load the record.jsonl data
    """
    record_data = []
    with jsonlines.open(path, "r") as reader:
        for obj in reader:
            record_data.append(obj)

    n = len(record_data[0]["output"])  # number of sequence
    flatten_data = []
    for idx, d in enumerate(record_data):
        for j, output in enumerate(d["output"]):
            flatten_data.append(
                {
                    "question": d["question"],
                    "groundtruth": d["groundtruth"],
                    "question_idx": idx,
                    "path_idx": output["path_idx"],
                    "value": output["value"],
                    "gen_answer": output["text"],
                }
            )

    return flatten_data, n
