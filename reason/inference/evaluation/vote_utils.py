from collections import Counter, defaultdict
from typing import List

MAJORITY_VOTE = "majority_vote"
ORM_VOTE = "orm_vote"
ORM_MAX = "orm_max"


def _agg_majority_vote(x_list: List[str], unused_v_list: List[float]):
    counts = Counter(x_list)
    most_common = max(counts, key=counts.get)
    return most_common


def _agg_orm_vote(x_list: List[str], v_list: List[float]):
    assert len(x_list) == len(v_list)
    x_dict = defaultdict(lambda: 0.0)
    for x, v in zip(x_list, v_list):
        x_dict[x] += v

    highest_x = max(x_dict, key=x_dict.get)
    return highest_x


def _agg_orm_max(x_list: List[str], v_list: List[float]):
    text_max = x_list[v_list.index(max(v_list))]
    return text_max


AGG_FN_MAP = {
    MAJORITY_VOTE: _agg_majority_vote,
    ORM_VOTE: _agg_orm_vote,
    ORM_MAX: _agg_orm_max,
}
