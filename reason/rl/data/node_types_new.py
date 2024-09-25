from dataclasses import dataclass
import numpy as np
import torch
from torchtyping import TensorType
from typing import Optional, Sequence, List
from transformers import AutoTokenizer
from tsllm.distributed.utils import print_with_rank, print_rank_0


def _tokenize_fn(s, tokenizer, drop_bos: bool = False):
    input_ids = tokenizer(s, return_tensors="pt", padding=True).input_ids
    if drop_bos and torch.all(input_ids[:, 0] == tokenizer.bos_token_id):
        input_ids = input_ids[:, 1:]
    return input_ids


@dataclass
class TimeStep:
    query_tensor: TensorType["query_length"]
    response_tensor: TensorType["response_length"]
    reward: TensorType
    value: TensorType
    returns: TensorType
    legal_actions_tensor: TensorType["num_action", "max_action_length"]
    # legal_actions_attn_mask: TensorType["num_action", "max_action_length"]
    action_probs: TensorType["num_action"]
    termiated: TensorType
    truncated: TensorType

    @classmethod
    def from_string(
        cls,
        tokenizer: AutoTokenizer,
        query_str: str,
        response_str: str,
        reward: float,
        value: float,
        legal_actions: List[str],
        action_probs: TensorType["num_action"],
        terminated: bool,
        truncated: bool,
    ):
        assert (
            tokenizer.padding_side == "right"
        ), "the tokenizer's padding side should be right."
        assert tokenizer.pad_token != None, "Your tokenizer's pad_token is None"

        # here only squeeze the first batch dimension
        query_tensor = _tokenize_fn(query_str, tokenizer, False).squeeze_(0)
        total_tensor = _tokenize_fn(
            query_str + response_str, tokenizer, False
        ).squeeze_(0)
        response_tensor = total_tensor[len(query_tensor) :]

        total_legal_action_qa_tensor = _tokenize_fn(
            [query_str + action_str for action_str in legal_actions], tokenizer, False
        )
        legal_action_tensor = total_legal_action_qa_tensor[:, len(query_tensor) :]

        # yapf: disable
        return cls(query_tensor,
                   response_tensor,
                   torch.tensor(reward),
                   torch.tensor(value),
                   torch.tensor(0.),
                   legal_action_tensor,
                   torch.tensor(action_probs),
                   torch.tensor(terminated),
                   torch.tensor(truncated))
        # yapf: enable


import scipy.signal


def discount_cumsum(x, discount):
    if isinstance(x, torch.Tensor):
        x = x.numpy()
        return torch.tensor(
            scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[
                ::-1
            ].copy()
        )
    else:
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def _compute_return_fn(rews, vals, gamma, gae_lambda, last_value):
    last_gae_lam = 0
    reversed_adv = []
    for t in reversed(range(len(rews))):
        next_v = vals[t + 1] if t < len(rews) - 1 else last_value
        delta = rews[t] + gamma * next_v - vals[t]
        last_gae_lam = delta + gamma * gae_lambda * last_gae_lam
        reversed_adv.append(last_gae_lam)
    adv = torch.tensor(reversed_adv[::-1])
    ret = adv + vals
    return ret


@dataclass
class Trajectory:
    timesteps: Sequence[TimeStep]

    def compute_returns(
        self, gamma: float = 0.99, gae_lambda: float = 0.95, last_value: float = 0
    ):
        rews = torch.tensor([ts.reward for ts in self.timesteps])
        vals = torch.tensor([ts.value for ts in self.timesteps])
        ret = _compute_return_fn(rews, vals, gamma, gae_lambda, last_value)

        ## ========= trlx PPO's implementation ===========
        # lastgaelam = 0
        # advantages_reversed = []
        # for t in reversed(range(response_length)):
        #     nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
        #     delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
        #     lastgaelam = delta + self.gamma * self.lam * lastgaelam
        #     advantages_reversed.append(lastgaelam)
        # advantages = torch.stack(advantages_reversed[::-1], dim=1)
        # returns = advantages + values
        # if use_whitening:
        #     advantages = whiten(advantages)

        ## ========= OpenAI SpinningUp PPO's implementation ===========
        # rews = [ts.reward for ts in self.timesteps] + [last_value]
        # vals = [ts.value for ts in self.timesteps] + [last_value]
        # rews = np.array(rews)
        # vals = np.array(vals)

        # deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
        # adv = discount_cumsum(deltas, gamma * gae_lambda)
        # ret = discount_cumsum(rews, gamma)[:-1]

        for i in range(len(self.timesteps)):
            self.timesteps[i].returns = ret[i]


@dataclass
class MCTSBatch:
    query_tensor: TensorType["bsz", "query_length"]
    response_tensor: TensorType["bsz", "response_length"]
    reward: TensorType["bsz"]
    value: TensorType["bsz"]
    returns: TensorType["bsz"]
    legal_actions_tensor: TensorType["bsz", "num_action", "max_action_length"]
    # legal_actions_attn_mask: TensorType["bsz", "num_action", "max_action_length"]
    action_probs: TensorType["bsz", "num_action"]
    termiated: TensorType["bsz"]
    truncated: TensorType["bsz"]

    def to(self, *args, **kwargs):
        self.query_tensor = self.query_tensor.to(*args, **kwargs)
        self.response_tensor = self.response_tensor.to(*args, **kwargs)
        self.reward = self.reward.to(*args, **kwargs)
        self.value = self.value.to(*args, **kwargs)
        self.returns = self.returns.to(*args, **kwargs)
        self.legal_actions_tensor = self.legal_actions_tensor.to(*args, **kwargs)
        self.action_probs = self.action_probs.to(*args, **kwargs)
        self.termiated = self.termiated.to(*args, **kwargs)
        self.truncated = self.truncated.to(*args, **kwargs)


@dataclass
class SftInstance:
    input_ids: TensorType["seq_len"]
    label: TensorType["seq_len"]
    returns: TensorType["seq_len"]
    mask: TensorType["seq_len"]

    @classmethod
    def from_string(
        cls,
        q_str: str,
        r_str: str,
        tokenizer: AutoTokenizer,
        policy_forward_seq_value: callable,
        gamma: float,
        gae_lambda: float,
        IGNORE_IDX=-100,
    ):
        q_ids = _tokenize_fn(q_str, tokenizer, False).squeeze(0)
        r_ids = _tokenize_fn(r_str, tokenizer, True).squeeze(0)
        total_ids = _tokenize_fn(q_str + r_str, tokenizer, False).squeeze(0)
        assert len(total_ids) == len(q_ids) + len(r_ids)

        label = total_ids.clone()
        label[: len(q_ids)] = IGNORE_IDX
        input_ids = total_ids

        answer_steps = r_str.split("\n")
        vs = []
        mask = torch.zeros_like(input_ids)
        current_str = q_str

        value_seq = policy_forward_seq_value(q_str + r_str).squeeze(0)
        for i, a in enumerate(answer_steps):
            if len(a) == 0:
                if i != len(answer_steps) - 1:
                    print_with_rank(
                        "possible problems met in sft instance building. {}".format(
                            answer_steps
                        )
                    )
                continue
            current_str += a + "\n"
            current_ids = _tokenize_fn(current_str, tokenizer, False).squeeze(0)
            # current_value = policy_forward_value_fn(current_str).item()
            current_value = value_seq[len(current_ids) - 1]
            mask[len(current_ids) - 1] = 1
            vs.append(current_value)
        vs = torch.tensor(vs)
        rews = torch.zeros_like(vs)
        rews[-1] = 1.0  # sft instance is corrent.
        rets = _compute_return_fn(
            rews=rews, vals=vs, gamma=gamma, gae_lambda=gae_lambda, last_value=0
        )  # sft instances always terminate.
        returns = torch.zeros_like(input_ids)
        nonzero_indices = mask.nonzero()
        assert len(nonzero_indices) == len(vs), (
            len(nonzero_indices),
            len(vs),
            nonzero_indices,
            vs,
        )
        for i, idx in enumerate(nonzero_indices):
            returns[idx.item()] = rets[i]
        return cls(input_ids, label, returns, mask)


@dataclass
class SftBatch:
    input_ids: TensorType["bsz", "seq_len"]
    label: TensorType["bsz", "seq_len"]
    attn_mask: TensorType["bsz", "seq_len"]
    returns: TensorType["bsz", "seq_len"]
    mask: TensorType["bsz", "seq_len"]  # value_mask

    def to(self, *args, **kwargs):
        self.input_ids = self.input_ids.to(*args, **kwargs)
        self.label = self.label.to(*args, **kwargs)
        self.attn_mask = self.attn_mask.to(*args, **kwargs)
        self.returns = self.returns.to(*args, **kwargs)
        self.mask = self.mask.to(*args, **kwargs)


@dataclass
class TrajInstance:
    input_ids: TensorType["seq_len"]
    label: TensorType["seq_len"]
    returns: TensorType["seq_len"]
    mask: TensorType["seq_len"]
    question: str
    response: str

    @classmethod
    def from_string(
        cls,
        q_str: str,
        r_str: str,
        value_index: np.array,
        reward_list: np.array,
        tokenizer: AutoTokenizer,
        policy_forward_seq_value: callable,
        gamma: float,
        gae_lambda: float,
        cal_value: bool,
        use_gae=True,
        IGNORE_IDX=-100,
    ):
        q_ids = _tokenize_fn(q_str, tokenizer, False).squeeze(0)
        r_ids = _tokenize_fn(r_str, tokenizer, True).squeeze(0)
        total_ids = _tokenize_fn(q_str + r_str, tokenizer, False).squeeze(0)

        # Check whether q_str + r_str creates some new tokens
        # assert len(total_ids) == len(q_ids) + len(r_ids)

        label = total_ids.clone()
        label[: len(q_ids)] = IGNORE_IDX
        input_ids = total_ids

        if cal_value:
            value_index = torch.tensor(value_index).int()
            reward_list = torch.tensor(reward_list).float()
            mask = torch.zeros_like(input_ids)

            # Check value index dimension is correct
            assert value_index[0] == len(q_ids) - 1
            assert value_index[-1] == len(mask) - 1
            # assert r_str.split("\n")[-1] == ""
            # answer_steps = r_str.split("\n")[:-1]  # The last one is null or eos
            vs = []
            # current_str = q_str

            # TODO: Organize code and check logic
            if use_gae:
                value_seq = policy_forward_seq_value(q_str + r_str).squeeze(0)

                # mask and reward for the final answer
                # Value index
                mask[value_index[:-1]] = 1
                # Final reward index
                mask[value_index[-1]] = 2

                vs = value_seq[value_index[:-1]]
                rews = reward_list

                rets = _compute_return_fn(
                    rews=rews, vals=vs, gamma=gamma, gae_lambda=gae_lambda, last_value=0
                )  # sft instances always terminate.

                # add the final reward
                rets = torch.cat([rets, reward_list[-1:]])

                returns = torch.zeros(len(input_ids))
                nonzero_indices = mask.nonzero()
                assert len(nonzero_indices) == len(rets) == len(vs) + 1

                for i, idx in enumerate(nonzero_indices):
                    returns[idx.item()] = rets[i]
                # result = torch.tensor([result])
            else:
                # conduct MC return calculation
                # Value index
                mask[value_index[:-1]] = 1
                # Final reward index
                mask[value_index[-1]] = 2
                assert value_index[-1] == len(mask) - 1

                rets = discount_cumsum(reward_list, gamma)

                # append the final reward
                rets = torch.cat([rets, reward_list[-1:]])

                returns = torch.zeros(len(input_ids))
                nonzero_indices = mask.nonzero()
                assert len(nonzero_indices) == len(rets)

                for i, idx in enumerate(nonzero_indices):
                    returns[idx.item()] = rets[i]
        else:
            # add padding tensor if not calculate value
            # result = torch.tensor([result])
            returns = torch.zeros(len(input_ids))
            mask = torch.zeros_like(input_ids)
        return cls(input_ids, label, returns, mask, q_str, r_str)


@dataclass
class TrajBatch:
    input_ids: TensorType["bsz", "seq_len"]
    label: TensorType["bsz", "seq_len"]
    attn_mask: TensorType["bsz", "seq_len"]
    returns: TensorType["bsz", "seq_len"]
    mask: TensorType["bsz", "seq_len"]  # value_mask

    def to(self, *args, **kwargs):
        self.input_ids = self.input_ids.to(*args, **kwargs)
        self.label = self.label.to(*args, **kwargs)
        self.attn_mask = self.attn_mask.to(*args, **kwargs)
        self.returns = self.returns.to(*args, **kwargs)
        self.mask = self.mask.to(*args, **kwargs)
