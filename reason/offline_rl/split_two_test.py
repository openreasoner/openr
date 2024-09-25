import copy
import json
import random
from pathlib import Path
from argparse import ArgumentParser
from tsllm.offline_rl.utils import load_jsonl, write_to_jsonl


def sample_dicts(data, n):
    if n > len(data):
        n = len(data)
    return random.sample(data, n)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_num", type=int, default=20)
    parser.add_argument("--train_test_num", type=int, default=2)
    parser.add_argument("--train_data_prefix", type=str)

    args = parser.parse_args()

    train_data = load_jsonl(f"{args.train_data_prefix}_dedup.jsonl")


    train_data_eval = copy.deepcopy(train_data)
    for d1, d2 in zip(train_data, train_data_eval):
        picked_ans = sample_dicts(d1["answer"], args.train_num + args.train_test_num)
        if len(picked_ans) > args.train_num:
            d1["answer"] = picked_ans[: args.train_num]
            d2["answer"] = picked_ans[args.train_num :]
        else:
            d1["answer"] = picked_ans
            d2["answer"] = []

    write_to_jsonl(
        train_data, f"{args.train_data_prefix}_dedup_sample{args.train_num}.jsonl"
    )
    write_to_jsonl(
        train_data_eval,
        f"{args.train_data_prefix}_dedup_sample{args.train_num}_train_test_sample_{args.train_test_num}.jsonl",
    )
