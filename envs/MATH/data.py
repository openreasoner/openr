from pathlib import Path
import jsonlines
import json
import os
from torch.utils.data import Dataset


def get_train_test_dataset(*args, **kwargs):
    root_dir = Path(__file__).parent.parent.parent  # base dir
    train_data_path = kwargs['train_data_path']
    test_data_path = kwargs['test_data_path']
    test_ds = JsonlMathDataset(root_dir / test_data_path)
    train_ds = JsonlMathDataset(root_dir / train_data_path)

    return train_ds, test_ds

def get_dataset(**kwargs):
    root_dir = Path(__file__).parent.parent.parent.resolve()
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
