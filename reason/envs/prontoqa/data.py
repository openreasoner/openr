from torch.utils.data import Dataset
import pandas as pd
import jsonlines
import json
import numpy as np
import re
from pathlib import Path


def get_train_test_dataset(*args, **kwargs):
    env_dir = Path(__file__).parent
    train_data_path = kwargs.get("train_data_path", env_dir / "train_data/train.jsonl")
    test_data_path = kwargs.get("test_data_path", env_dir / "train_data/test.jsonl")

    train_ds = JsonlQuestionDataset(train_data_path)
    test_ds = JsonlQuestionDataset(test_data_path)
    return train_ds, test_ds


class JsonlQuestionDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = []

        with jsonlines.open(data_path, "r") as reader:
            for obj in reader:
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            "question": self.data[index]["question"],
            "answer": self.data[index]["answer"][0]["text"],
        }


# This is the dataset that
class CSVDataset(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        self.data = list(pd.read_csv(data_path)["Puzzles"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {"question": self.data[index]}


def build_question(question, query):
    statement = query[15:-1]
    new_question = question + ' Is the statement "{}" true or false?'.format(statement)
    return new_question


def build_text(cot, answer):
    text = "\n".join(cot) + "\nThe answer is {}.".format(answer.lower())
    return text


def data_preprocess(read_path, save_dir):
    text_steps = []
    with open(read_path, "r") as reader:
        data_set = json.load(reader)
        i = 0

        for batch in data_set.values():
            for key in batch.keys():
                question = build_question(batch[key]["question"], batch[key]["query"])
                text = build_text(batch[key]["chain_of_thought"], batch[key]["answer"])
                text_steps.append(len(text.split("\n")))

                new_obj = {
                    "question": question,
                    "i": i,
                    "answer": [{"text": text, "correct": True}],
                }

                if "test_example" == key:
                    with jsonlines.open(save_dir + "test.jsonl", "a") as writer:
                        writer.write(new_obj)
                else:
                    with jsonlines.open(save_dir + "train.jsonl", "a") as writer:
                        writer.write(new_obj)

                with jsonlines.open(save_dir + "all.jsonl", "a") as writer:
                    writer.write(new_obj)
                i += 1

    print("count: ", len(text_steps))
    print("mean step num: ", np.mean(text_steps))
    print("std step num: ", np.std(text_steps))
    print("max step num: ", np.max(text_steps))
    print("min step num: ", np.min(text_steps))
