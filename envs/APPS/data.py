import os
import json
from pathlib import Path
import jsonlines
from torch.utils.data import Dataset
from .apps_utils import get_test_cases


def get_train_test_dataset(*args, **kwargs):
    env_dir = Path(__file__).parent.parent
    test_ds = APPSDataset(env_dir / "APPS/dataset/test/")
    train_ds = APPSDataset(env_dir / "APPS/dataset/train/")
    return train_ds, test_ds


class APPSDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = []
        difficulty = 'introductory'
        if 'train' in data_path.name:
            problem_indices = range(4000, 4002)
        else:
            problem_indices = range(4000, 4070)
        for idx in problem_indices:
            prob_dir = f"{data_path}/{idx:04d}"
            pro_metadata_path = os.path.join(prob_dir, "metadata.json")
            public_test_case_path = os.path.join(prob_dir, "public_input_output.json")
            test_case_path = os.path.join(prob_dir, "input_output.json")
            prompt_path = os.path.join(prob_dir, "question.txt")
            starter_path = os.path.join(prob_dir, "starter_code.py")
            solutions_path = os.path.join(prob_dir, "solutions.json")
            if not os.path.exists(starter_path):
                starter_path = None

            # difficulty filtering
            if difficulty is not None:
                with open(pro_metadata_path) as f:
                    pro_metadata = json.load(f)
                    if pro_metadata['difficulty'] != difficulty:
                        continue
            # problem instance formation
            problem_instance = {}
            problem_instance['index'] = idx
            input_prompt = "\nQUESTION:\n"
            with open(prompt_path, "r") as f:
                data = f.readlines()
                data = "".join(data)
            input_prompt += data
            if starter_path != None:
                with open(starter_path, "r") as f:
                    data = f.readlines()
                    data = "".join(data)
                    data = "\n" + data  # + "\n"
                input_prompt += data
            with open(test_case_path, "r") as f:
                data = json.load(f)
            if not data.get("fn_name"):
                input_prompt += "\nUse Standard Input format"  # \n"
                problem_instance['code_type'] = "standard_input"
                problem_instance['method_name'] = None
            else:
                input_prompt += "\nUse Call-Based format"  # \n"
                problem_instance['code_type'] = "call_based"
                problem_instance['method_name'] = data.get("fn_name")
            input_prompt += "\nANSWER:\n"
            problem_instance["prompt"] = input_prompt

            # test cases for train and test
            train_in_outs, test_in_outs = get_test_cases(prob_dir, 'half')
            problem_instance["train_in_outs"] = train_in_outs
            problem_instance["test_in_outs"] = test_in_outs
            self.data.append(problem_instance)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        return {"question": x["prompt"],
                "answer": x}
