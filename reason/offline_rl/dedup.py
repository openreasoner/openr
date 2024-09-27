import json
from argparse import ArgumentParser

from offline_rl.utils import load_jsonl, write_to_jsonl

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    data = load_jsonl(args.input_path)
    print("Load from {}".format(args.input_path))
    cnt = 0
    total_cnt = 0
    correct_cnt = 0
    for d in data:
        answer = d["answer"] if "answer" in d else d["output"]
        unique_data = []
        seen_texts = []
        for item in answer:
            if item["text"] not in seen_texts:
                seen_texts.append(item["text"])
                unique_data.append(item)
                cnt += 1
                if item["correct"]:
                    correct_cnt += 1
            total_cnt += 1
        d["answer"] = unique_data

    write_to_jsonl(data, args.output_path)

    print("{} {} {}".format(cnt, total_cnt, correct_cnt))
