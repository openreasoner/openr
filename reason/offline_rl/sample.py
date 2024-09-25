import json
import random
from argparse import ArgumentParser

from tsllm.offline_rl.utils import load_jsonl, write_to_jsonl


def sample_dicts(data, n):
    if n > len(data):
        n = len(data)
    return random.sample(data, n)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("-n", type=int, default=10)
    args = parser.parse_args()

    data = load_jsonl(args.input_path)
    for d in data:
        d["answer"] = sample_dicts(d["answer"], args.n)

    write_to_jsonl(data, args.output_path)
