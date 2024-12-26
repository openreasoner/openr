import json
from typing import Optional
import random
import numpy as np
import os
import torch
from dataclasses import dataclass


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

def read_txt(file_path):
    assert str(file_path).endswith(".txt")
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    return data
