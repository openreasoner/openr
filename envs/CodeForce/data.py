import os
import json
from pathlib import Path
import jsonlines
from torch.utils.data import Dataset

def get_train_test_dataset(*args, **kwargs):
    env_dir = Path(__file__).parent.parent
    test_ds = CodeForceDataset(env_dir / "CodeForce/dataset/test/")
    train_ds = CodeForceDataset(env_dir / "CodeForce/dataset/train/")
    return train_ds, test_ds

def get_test_cases(public_testcases, private_testcases):
    in_out_len = len(public_testcases["input"]) + len(private_testcases["input"])
    public_test_cases = in_out_len // 2
    inputs = public_testcases["input"] + private_testcases["input"]
    outputs = public_testcases["output"] + private_testcases["output"]
    train_in_outs, test_in_outs = {}, {}
    private_test_cases = in_out_len - public_test_cases
    if public_test_cases < 1 or private_test_cases < 1:
        # print(f"Not enough test cases: {public_test_cases}, {private_test_cases}.")
        return None, None
    train_in_outs["inputs"] = inputs[:public_test_cases]
    train_in_outs["outputs"] = outputs[:public_test_cases]
    test_in_outs["inputs"] = inputs[public_test_cases:]
    test_in_outs["outputs"] = outputs[public_test_cases:]
    return train_in_outs, test_in_outs

class CodeForceDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        remained = 0
        self.data = []
        self.all = json.load(open(data_path, "r"))
        for problem in self.all:
            problem_instance = {}
            problem_instance["index"] = problem["index"]
            problem_instance["prompt"] = (
                "\nQUESTION:\n" + problem["input"] + "\nANSWER:\n"
            )
            problem_instance["code_type"] = "standard_input"
            problem_instance["method_name"] = None
            train_in_outs, test_in_outs = get_test_cases(
                problem["public_tests"], problem["private_tests"]
            )
            if train_in_outs == None:
                continue
            remained += 1
            problem_instance["train_in_outs"] = train_in_outs
            problem_instance["test_in_outs"] = test_in_outs

            self.data.append(problem_instance)
        print(f"{remained} questions remain.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        return {"question": x["prompt"],
                "answer": x}
