from typing import List, Sequence
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from rl.data.node_types import SftBatch, SftInstance


def collate_fn(
    pad_token_id: int,
    IGNORE_INDEX: int,
    elems: Sequence[SftInstance],
) -> SftBatch:
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

    return SftBatch(input_ids, label, attn_mask, returns, mask)


class SFTBuffer(Dataset):
    def __init__(self, pad_token_id, IGNORE_INDEX=-100) -> None:
        super().__init__()

        self.history: List[SftInstance] = []
        self.IGNORE_INDEX = IGNORE_INDEX
        self.pad_token_id = pad_token_id

    def push(self, exps: Sequence[SftInstance]):
        self.history += exps

    def add(self, inst: SftInstance):
        self.history.append(inst)

    def clear(self):
        self.history = []

    def __len__(self):
        return len(self.history)

    def __getitem__(self, index):
        return self.history[index]

    def create_loader(self, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            self,
            batch_size,
            shuffle=shuffle,
            collate_fn=partial(collate_fn, self.pad_token_id, self.IGNORE_INDEX),
        )
