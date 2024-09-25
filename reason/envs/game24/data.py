from torch.utils.data import Dataset
import pandas as pd
import jsonlines
from pathlib import Path


def get_train_test_dataset(*args, **kwargs):
    env_dir = Path(__file__).parent
    train_data_path = kwargs.get(
        "train_data_path", env_dir / "train_data/train_dedup.jsonl"
    )
    test_data_path = kwargs.get(
        "test_data_path", env_dir / "train_data/test_dedup.jsonl"
    )

    train_ds = JsonlQuestionDataset(train_data_path)
    test_ds = JsonlQuestionDataset(test_data_path)
    return train_ds, test_ds


class JsonlQuestionDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = []

        with jsonlines.open(data_path, "r") as reader:
            for obj in reader:
                obj.pop("answer")
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # dummy answer, just for compatibility
        return {"question": self.data[index]["question"], "answer": "[DUMMY ANSWER]"}


# This is the dataset that
class CSVDataset(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        self.data = list(pd.read_csv(data_path)["Puzzles"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {"question": self.data[index]}
