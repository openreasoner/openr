import functools as ft
import math
import multiprocessing as mp
import re
from pathlib import Path
from typing import Iterable

from src.data_types import ConvertedItem
from src.data_types.math_aps import (MathAPSItem, MathAPSItemV2Tree,
                                     ReasoningNode, ReasoningStep)
from src.preprocessors.base import PreprocessorBase
from src.preprocessors.utils import (dump_converted_ds, read_math_aps_ds,
                                     read_math_aps_v2_tree_ds)


class MathAPSPreprocessor(PreprocessorBase):
    def __init__(
        self,
        ds_path: str | Path,
        step_tag: str,
        suffix: str = "new",
        **kwargs,
    ) -> None:
        del kwargs  # for compatibility with PRM800KPreprocessor
        super().__init__(ds_path, step_tag, suffix)

    def _read_ds(self) -> None:
        self.original_items = read_math_aps_ds(self.ds_path)

    def convert(self) -> None:
        self._read_ds()
        assert self.original_items is not None

        convert_fn = ft.partial(convert_math_aps_item, step_tag=self.step_tag)
        with mp.Pool(mp.cpu_count()) as p:
            unflattened_items = p.map(convert_fn, self.original_items)

        self.converted_items = list(
            ft.reduce(
                lambda res, items: res + items,  # type: ignore
                unflattened_items,
                [],
            )
        )

    def dump(self) -> None:
        assert self.converted_items is not None
        dump_converted_ds(self.output_path, list(self.converted_items))


def convert_math_aps_item(
    original_item: MathAPSItem,
    step_tag: str,
) -> list[ConvertedItem]:
    question = original_item.question

    def extract_item(reasoning_step: ReasoningStep) -> ConvertedItem:
        return ConvertedItem(
            question=question,
            process=f"{reasoning_step.solution_prefix} {step_tag}",
            label=[label_from_mc_value(reasoning_step.mc_value, 0.5)],
        )

    def filter_item(reasoning_step: ReasoningStep) -> bool:
        return not (
            completion_too_short(reasoning_step.solution_prefix)
            or contain_non_ascii_text(reasoning_step.solution_prefix)
        )

    extracted_items = map(
        extract_item, filter(filter_item, original_item.reasoning_steps)
    )
    distinct_items = set(extracted_items)

    return list(distinct_items)


class MathAPSV2TreePreprocessor(PreprocessorBase):
    def __init__(
        self, ds_path: str | Path, step_tag: str, suffix: str = "new", **kwargs
    ) -> None:
        del kwargs  # for compatibility with PRM800KPreprocessor
        super().__init__(ds_path, step_tag, suffix)

    def _read_ds(self) -> None:
        self.original_items = read_math_aps_v2_tree_ds(self.ds_path)

    def convert(self) -> None:
        self._read_ds()
        assert self.original_items is not None

        convert_fn = ft.partial(convert_math_aps_v2_tree_item, step_tag=self.step_tag)
        with mp.Pool(mp.cpu_count()) as p:
            unflattened_items = p.map(convert_fn, self.original_items)

        self.converted_items = list(
            ft.reduce(
                lambda res, items: res + items,  # type: ignore
                unflattened_items,
                [],
            )
        )

    def dump(self) -> None:
        assert self.converted_items is not None

        dump_converted_ds(self.output_path, list(self.converted_items))


def convert_math_aps_v2_tree_item(
    original_item: MathAPSItemV2Tree,
    step_tag: str,
) -> list[ConvertedItem]:
    question = original_item.question
    rollouts = recover_rollouts_from_tree_node(original_item.reasoning_steps, step_tag)

    def filter_rollout(rollout: tuple[str, list[str]]) -> bool:
        return not contain_non_ascii_text(
            rollout[0]
        )  # CAUTION: step_tag maybe non-ascii

    def extract_item(rollout: tuple[str, list[str]]) -> ConvertedItem:
        process, labels = rollout
        return ConvertedItem(question, process, labels)

    extracted_items = map(extract_item, filter(filter_rollout, rollouts))
    distinct_items = set(extracted_items)

    return list(distinct_items)


def recover_rollouts_from_tree_node(
    node: ReasoningNode, step_tag: str
) -> list[tuple[str, list[str]]]:
    label = label_from_mc_value(node.mc_value, 0.5)
    if len(node.children) == 0:  # leaf
        return [(f"{node.text} {step_tag}", [label])]
    else:
        cur_step = node.text
        res = []

        for child in node.children:
            for future_rollout in recover_rollouts_from_tree_node(child, step_tag):
                future_steps, future_vals = future_rollout
                future_vals.insert(0, label)
                res.append((f"{cur_step} {step_tag} {future_steps}", future_vals))

        return res


def label_from_mc_value(mc_value: float, threshold: float = 0.5) -> str:
    """If `mc_value` > `threshold` returns `+` otherwise `-`."""
    return "+" if mc_value > threshold else "-"


# Sifters for cleaning Math-APS data


def character_counts(string: str) -> dict[str, int]:
    chars = sorted(list(set(string)))
    return {c: string.count(c) for c in chars}


def entropy_of_counts(counts: Iterable[int]) -> float:
    total_counts = sum(counts)
    dist = map(lambda x: x / total_counts, counts)
    entropy = sum(map(lambda x: -x * math.log(x), dist))

    return entropy


def entropy_of_string(string: str) -> float:
    return entropy_of_counts(character_counts(string).values())


def entropy_too_large(string: str, entropy_thres: float = 3.5) -> bool:
    entropy = entropy_of_string(string)
    return entropy > entropy_thres


def contain_non_ascii_text(string: str) -> bool:
    try:
        string.encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def completion_too_short(string: str, word_count_thres: int = 5) -> bool:
    return len(string.split(" ")) <= word_count_thres


def contains_chinese(string: str) -> bool:
    # Chinese Unicode \u4e00-\u9fff
    chinese_characters = re.compile(r"[\u4e00-\u9fff]")
    return bool(chinese_characters.search(string))
