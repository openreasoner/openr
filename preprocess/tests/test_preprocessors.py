from pathlib import Path

import pytest
from src.data_types.math_aps import MathAPSItemV2Tree, ReasoningNode
from src.preprocessors.math_aps import (
    MathAPSPreprocessor,
    MathAPSV2TreePreprocessor,
    convert_math_aps_v2_tree_item,
    recover_rollouts_from_tree_node,
)
from src.preprocessors.math_shepherd import MathShepherdPreprocessor
from src.preprocessors.prm800k import PRM800KPreprocessor
from src.preprocessors.utils import read_math_aps_v2_tree_ds
from tests.test_data_types import (
    example_math_aps_path,
    example_math_aps_v2_tree_path,
    example_math_shepherd_path,
    example_prm800k_path,
)

STEP_TAG = "च"


def test_prm800k_preprocessor(example_prm800k_path: Path) -> None:
    runner = PRM800KPreprocessor(example_prm800k_path, STEP_TAG)
    runner.convert()

    assert runner.converted_items is not None
    assert len(runner.converted_items) > 0


def test_math_aps_preprocessor(example_math_aps_path: Path) -> None:
    runner = MathAPSPreprocessor(example_math_aps_path, STEP_TAG)
    runner.convert()

    assert runner.converted_items is not None
    assert len(runner.converted_items) > 0


def test_math_shepherd_preprocessor(example_math_shepherd_path: Path) -> None:
    runner = MathShepherdPreprocessor(example_math_shepherd_path, STEP_TAG)
    runner.convert()

    assert runner.converted_items is not None
    assert len(runner.converted_items) > 0


@pytest.fixture
def example_reasoning_node() -> ReasoningNode:
    data = {
        "text": "s0",
        "mc_value": 0.0,
        "children": [
            {
                "text": "s1-0",
                "mc_value": 0.1,
                "children": [
                    {"text": "s2-0", "mc_value": 0.92, "children": []},
                    {
                        "text": "s2-1",
                        "mc_value": 0.21,
                        "children": [{"text": "s3-0", "mc_value": 0.3, "children": []}],
                    },
                ],
            },
            {
                "text": "s1-1",
                "mc_value": 0.99,
                "children": [{"text": "s2-0", "mc_value": 0.9, "children": []}],
            },
        ],
    }
    return ReasoningNode.from_dict(data)


def test_recover_rollouts_from_tree_node(example_reasoning_node: ReasoningNode) -> None:
    rollouts = recover_rollouts_from_tree_node(example_reasoning_node, STEP_TAG)

    assert len(rollouts) == 3

    reasonings, labels = zip(*rollouts)
    assert set(reasonings) == {
        f"s0 {STEP_TAG} s1-0 {STEP_TAG} s2-0 {STEP_TAG}",
        f"s0 {STEP_TAG} s1-0 {STEP_TAG} s2-1 {STEP_TAG} s3-0 {STEP_TAG}",
        f"s0 {STEP_TAG} s1-1 {STEP_TAG} s2-0 {STEP_TAG}",
    }
    assert list(labels) == [["-", "-", "+"], ["-", "-", "-", "-"], ["-", "+", "+"]]


@pytest.fixture
def example_math_aps_v2_tree_ds(
    example_math_aps_v2_tree_path: Path,
) -> list[MathAPSItemV2Tree]:
    return list(read_math_aps_v2_tree_ds(example_math_aps_v2_tree_path))


STEP_TAG = "<|STEP|>"  # change to ascii step-tag to pass non-ascii filter


def test_convert_math_aps_v2_tree_item(
    example_math_aps_v2_tree_ds: list[MathAPSItemV2Tree],
) -> None:
    for i, item in enumerate(example_math_aps_v2_tree_ds):
        assert (
            len(convert_math_aps_v2_tree_item(item, STEP_TAG)) > 0
        ), f"Item {i} find none"


def test_math_aps_v2_tree_preprocessor(example_math_aps_v2_tree_path: Path) -> None:
    runner = MathAPSV2TreePreprocessor(example_math_aps_v2_tree_path, STEP_TAG)
    runner.convert()

    assert runner.converted_items is not None
    assert len(runner.converted_items) > 0
