from pathlib import Path
import jsonlines
import os
import json
from torch.utils.data import Dataset


def get_train_test_dataset(*args, **kwargs):
    env_dir = Path(__file__).parent.parent
    test_ds = JsonlMathDataset(env_dir / "MATH/dataset/test500.jsonl")
    train_ds = JsonlMathDataset(env_dir / "MATH/dataset/train.jsonl")
    return train_ds, test_ds

def get_dataset(**kwargs):
    root_dir = Path(__file__).parent.parent.parent.resolve()
    print(f"path = {os.path.join(root_dir, kwargs['data_path'])}")
    return JsonlMathDataset(os.path.join(root_dir, kwargs['data_path']))


class JsonlMathDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = []
        if data_path.endswith(".jsonl"):
            with jsonlines.open(data_path, "r") as reader:
                for obj in reader:
                    self.data.append(obj)
        elif data_path.endswith(".json"):
            with open(data_path, "r") as reader:
                self.data = json.load(reader)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        return {"question": x["problem"], "answer": x["solution"]}
