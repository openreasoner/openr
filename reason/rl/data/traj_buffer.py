from typing import List, Sequence, Dict
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from distributed.utils import print_rank_0, print_with_rank
from rl.data.node_types_new import TrajBatch, TrajInstance
import bisect
import json
import os


def collate_fn(
    pad_token_id: int,
    IGNORE_INDEX: int,
    elems: Sequence[TrajInstance],
) -> TrajBatch:
    input_ids = pad_sequence(
        [elem.input_ids for elem in elems],
        padding_value=pad_token_id,
        batch_first=True,
    )
    label = pad_sequence(
        [elem.label for elem in elems],
        padding_value=IGNORE_INDEX,
        batch_first=True,
    )
    attn_mask = input_ids.ne(pad_token_id)
    # result = torch.cat([elem.result for elem in elems])

    returns = pad_sequence(
        [elem.returns for elem in elems],
        padding_value=0.0,
        batch_first=True,
    )

    mask = pad_sequence(
        [elem.mask for elem in elems],
        padding_value=0,
        batch_first=True,
    )

    return TrajBatch(input_ids, label, attn_mask, returns, mask)


class TrajBuffer(Dataset):
    def __init__(self, max_size, pad_token_id, IGNORE_INDEX=-100) -> None:
        super().__init__()

        self.history: List[TrajInstance] = []
        self.max_size = max_size
        self.IGNORE_INDEX = IGNORE_INDEX
        self.pad_token_id = pad_token_id

    def push(self, exps: Sequence[TrajInstance]):
        self.history += exps

    def add(self, inst: TrajInstance):
        # add example
        # check whether the traj is the same with history
        # check whether the buffer is full
        result = ""
        for d in self.history:
            # need double check
            if inst.response == d.response:
                result = "repeat"
                break
        if not result == "repeat":
            if len(self.history) == self.max_size:
                self.history.pop(0)
                result = "full"
            self.history.append(inst)
        return result

    def clear(self):
        self.history = []

    def __len__(self):
        return len(self.history)

    def __getitem__(self, index):
        return self.history[index]

    def save(self, path):
        # save the databuffer
        data_dict_list = [
            {"question": data.question, "response": data.response}
            for data in self.history
        ]
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(path, "w") as file:
            for entry in data_dict_list:
                json_str = json.dumps(entry)
                file.write(json_str + "\n")

    def create_loader(self, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            self,
            batch_size,
            shuffle=shuffle,
            collate_fn=partial(collate_fn, self.pad_token_id, self.IGNORE_INDEX),
        )


class MultiTrajBuffer(Dataset):
    def __init__(
        self,
        num,
        per_problem_max_size,
        pad_token_id,
        IGNORE_INDEX=-100,
        buffer_name="SFT",
    ) -> None:
        super().__init__()
        self.num = num
        self.buffer_name = buffer_name
        self.IGNORE_INDEX = IGNORE_INDEX
        self.pad_token_id = pad_token_id
        self.history: Dict[TrajBuffer] = {
            i: TrajBuffer(per_problem_max_size, pad_token_id, IGNORE_INDEX)
            for i in range(num)
        }

    def add(self, idx: int, inst: TrajInstance):
        result = self.history[idx].add(inst)

        if result == "full":
            print_with_rank(f"The {idx} buffer is full, pop out the first traj")
        elif result == "repeat":
            print_with_rank(f"The {idx} buffer has the same example")
        else:
            pass

    def save(self, path):
        import os

        for i in range(self.num):
            self.history[i].save(
                os.path.join(path, f"Buffer_{self.buffer_name}_{i}.jsonl")
            )

    def clear_idx(self, idx: int):
        self.history[idx].clear()

    def clear_all(self):
        for i in range(self.num):
            self.history[i].clear()

    def cumsum(self, sequence):
        r, s = [], 0
        for e in range(self.num):
            l = len(sequence[e])
            r.append(l + s)
            s += l
        return r

    def __len__(self):
        return self.cumulative_sizes[-1]

    @property
    def cumulative_sizes(self):
        return self.cumsum(self.history)

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.history[dataset_idx][sample_idx]

    def create_loader(self, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            self,
            batch_size,
            shuffle=shuffle,
            collate_fn=partial(collate_fn, self.pad_token_id, self.IGNORE_INDEX),
        )


if __name__ == "__main__":
    # test
    multi_buffer = MultiTrajBuffer(num=2, per_problem_max_size=3, pad_token_id=0)
    sample = TrajInstance(
        torch.ones(2),
        torch.ones(3),
        torch.tensor([1]),
        torch.ones(3),
        torch.ones(3),
        "hello",
    )
    negative_sample = TrajInstance(
        -torch.ones(3),
        -torch.ones(3),
        torch.tensor([0]),
        -torch.ones(3),
        -torch.ones(3),
        "hello",
    )
    multi_buffer.add(idx=0, inst=sample)
    multi_buffer.add(idx=0, inst=negative_sample)
    multi_buffer.add(idx=1, inst=sample)
    multi_buffer.add(idx=1, inst=negative_sample)
    print(multi_buffer.history[0].history, multi_buffer.history[1].history)
    multi_buffer.history[0].history[0].input_ids = torch.zeros(3)
    # print(multi_buffer[0], multi_buffer[1])
    dataloader = multi_buffer.create_loader(2, False)
    for data in dataloader:
        print(data)
