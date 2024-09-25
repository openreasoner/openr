import json
from argparse import ArgumentParser
from tsllm.argparse_utils import str2bool

from tsllm.offline_rl.utils import load_jsonl, write_to_jsonl

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_paths", action="store", nargs="+")
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--deduplicate", type=str2bool, default=True)
    args = parser.parse_args()

    print(args.input_paths)
    data = []
    for input_path in args.input_paths:
        data.extend(load_jsonl(input_path))
    d_by_q = {}
    for d in data:
        q_str = d["question"]
        if q_str not in d_by_q:
            d_by_q[q_str] = {
                "question": q_str,
                "answer": d["answer"],
                "groundtruth": d.get("groundtruth", "[DUMMY]"),
            }
        else:
            d_by_q[q_str]["answer"].extend(d["answer"])

    if args.deduplicate:
        cnt, total_cnt, correct_cnt = 0, 0, 0
        data = list(d_by_q.values())
        print(len(data))
        for d in data:
            answer = d["answer"]
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
        print(cnt, total_cnt, correct_cnt)

    write_to_jsonl(data, args.output_path)
