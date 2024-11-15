import json
from pathlib import Path
from typing import Iterable

from src.data_types import (
    ConvertedItem,
    MathAPSItem,
    MathAPSItemV2Tree,
    MathShepherdItem,
    PRM800KItem,
)


def read_prm800k_ds(ds_path: Path) -> Iterable[PRM800KItem]:
    with open(ds_path, "r", encoding="utf-8") as fd:
        dict_lst = map(json.loads, fd.readlines())

    return map(PRM800KItem.from_dict, dict_lst)


def read_math_aps_ds(ds_path: Path) -> Iterable[MathAPSItem]:
    with open(ds_path, "r", encoding="utf-8") as fd:
        dict_lst = map(json.loads, fd.readlines())

    return map(MathAPSItem.from_dict, dict_lst)


def read_math_shepherd_ds(ds_path: Path) -> Iterable[MathShepherdItem]:
    with open(ds_path, "r", encoding="utf-8") as fd:
        dict_lst = map(json.loads, fd.readlines())

    return map(MathShepherdItem.from_dict, dict_lst)


def read_math_aps_v2_tree_ds(ds_path: Path) -> Iterable[MathAPSItemV2Tree]:
    with open(ds_path, "r", encoding="utf-8") as fd:
        dict_lst = map(json.loads, fd.readlines())

    return map(MathAPSItemV2Tree.from_dict, dict_lst)


def dump_converted_ds(save_path: Path, items: list[ConvertedItem]) -> None:
    with open(save_path, "w", encoding="utf-8") as fd:
        json.dump([it.to_dict() for it in items], fd)
