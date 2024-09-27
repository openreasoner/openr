from typing import List, Sequence
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from distributed.utils import print_with_rank
from rl.data.node_types import TimeStep, Trajectory, MCTSBatch
from functools import partial
from torch.nn.utils.rnn import pad_sequence


def collate_fn(
    padding_side: str,
    pad_token_id: int,
    max_action_length: int,
    max_num_actions: int,
    elems: Sequence[TimeStep],
) -> MCTSBatch:
    if padding_side == "left":
        # Left padding of already left-padded queries
        query_tensors = pad_sequence(
            [elem.query_tensor.flip(0) for elem in elems],
            padding_value=pad_token_id,
            batch_first=True,
        ).flip(1)
    elif padding_side == "right":
        query_tensors = pad_sequence(
            [elem.query_tensor for elem in elems],
            padding_value=pad_token_id,
            batch_first=True,
        )
    else:
        raise ValueError(f"Invalid padding side: {padding_side}")

    for elem in elems:
        # pad from [flexible_n_action, flexible_act_len] to [n_action, max_action_length]
        elem.legal_actions_tensor = F.pad(
            elem.legal_actions_tensor,
            (
                0,
                max_action_length - elem.legal_actions_tensor.shape[1],
                0,
                max_num_actions - elem.legal_actions_tensor.shape[0],
            ),
            mode="constant",
            value=pad_token_id,
        )

        # pad from [flexible_n_action] to [n_action]
        elem.action_probs = F.pad(
            elem.action_probs,
            (0, max_num_actions - elem.action_probs.shape[0]),
            mode="constant",
            value=0.0,
        )
    try:
        padded_response_tensor = pad_sequence(
            [elem.response_tensor for elem in elems],
            padding_value=pad_token_id,
            batch_first=True,
        )
    except Exception as e:
        print_with_rank([elem.response_tensor.shape for elem in elems])
        print_with_rank([elem.response_tensor for elem in elems])
        raise e

    return MCTSBatch(
        query_tensors,
        # Right pad the rest, to have a single horizontal query/response split
        padded_response_tensor,
        torch.stack([elem.reward for elem in elems]),
        torch.stack([elem.value for elem in elems]),
        torch.stack([elem.returns for elem in elems]),
        torch.stack([elem.legal_actions_tensor for elem in elems]),
        torch.stack([elem.action_probs for elem in elems]),
        torch.stack([elem.termiated for elem in elems]),
        torch.stack([elem.truncated for elem in elems]),
    )


class MCTSBuffer(Dataset):
    def __init__(self, padding_side, pad_token_id, max_action_length, max_num_actions):
        super().__init__()
        self.history: List[TimeStep] = []
        self.padding_side = padding_side
        self.pad_token_id = pad_token_id
        self.max_action_length = max_action_length
        self.max_num_actions = max_num_actions

    def __len__(self):
        return len(self.history)

    def push(self, exps: Sequence[TimeStep]):
        self.history += exps

    def clear(self):
        self.history = []

    def __getitem__(self, index: int) -> TimeStep:
        return self.history[index]

    def create_loader(
        self,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size,
            shuffle=shuffle,
            collate_fn=partial(
                collate_fn,
                self.padding_side,
                self.pad_token_id,
                self.max_action_length,
                self.max_num_actions,
            ),
        )
