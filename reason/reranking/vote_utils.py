from collections import Counter, defaultdict
from typing import List

MAJORITY_VOTE = "majority_vote"
ORM_VOTE = "orm_vote"
ORM_MAX = "orm_max"
PRM_MIN_MAX = "prm_min_max"
PRM_MIN_VOTE = "prm_min_vote"
PRM_LAST_MAX = "prm_last_max"
PRM_LAST_VOTE = "prm_last_vote"
PRM_AVG_MAX = "prm_avg_max"
PRM_AVG_VOTE = "prm_avg_vote"


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


def _agg_prm_min_max(x_list: List[str], v_list: List[List[float]]):
    v_list = [min(v) if v else -1.0 for v in v_list]
    text_max = x_list[v_list.index(max(v_list))]
    return text_max


def _agg_prm_last_max(x_list: List[str], v_list: List[List[float]]):
    v_list = [v[-1] if v else -1.0 for v in v_list]
    text_max = x_list[v_list.index(max(v_list))]
    return text_max


def _agg_prm_min_vote(x_list: List[str], v_list: List[List[float]]):
    v_list = [min(v) if v else -1.0 for v in v_list]
    return _agg_orm_vote(x_list, v_list)


def _agg_prm_last_vote(x_list: List[str], v_list: List[List[float]]):
    v_list = [v[-1] if v else -1.0 for v in v_list]
    return _agg_orm_vote(x_list, v_list)


def _agg_prm_avg_max(x_list: List[str], v_list: List[List[float]]):
    v_list = [(sum(v)/len(v)) if v else -1.0 for v in v_list]
    text_max = x_list[v_list.index(max(v_list))]
    return text_max


def _agg_prm_avg_vote(x_list: List[str], v_list: List[List[float]]):
    v_list = [(sum(v) / len(v)) if v else -1.0 for v in v_list]
    return _agg_orm_vote(x_list, v_list)


AGG_FN_MAP = {
    MAJORITY_VOTE: _agg_majority_vote,
    # ORM_VOTE: _agg_orm_vote,
    # ORM_MAX: _agg_orm_max,
    PRM_MIN_MAX: _agg_prm_min_max,
    PRM_MIN_VOTE: _agg_prm_min_vote,
    PRM_LAST_MAX: _agg_prm_last_max,
    PRM_LAST_VOTE: _agg_prm_last_vote,
    PRM_AVG_MAX: _agg_prm_avg_max,
    PRM_AVG_VOTE: _agg_prm_avg_vote,
}
