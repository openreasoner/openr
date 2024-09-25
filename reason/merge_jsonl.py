import jsonlines
import pathlib
import argparse


def merge_jsonl_files(target_dir):
    target_dir = pathlib.Path(target_dir)

    total_data = dict()
    for jl_file in target_dir.glob(r"*.jsonl"):
        if "merged.jsonl" in str(jl_file):
            print("WARNING!!: SKIP {}".format(jl_file))
            continue
        print(jl_file)
        with jsonlines.open(jl_file, "r") as reader:
            for obj in reader:
                total_data[obj["i"]] = obj

    with jsonlines.open(target_dir / "merged.jsonl", "w") as writer:
        for i in range(len(total_data)):
            try:
                writer.write(total_data[i])
            except KeyError as e:
                print("key i not exist: ", i)
                raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=str, required=True)
    args = parser.parse_args()
    merge_jsonl_files(args.target_dir)
