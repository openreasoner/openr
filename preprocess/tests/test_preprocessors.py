from pathlib import Path

from src.preprocessors.math_aps import MathAPSPreprocessor
from src.preprocessors.prm800k import PRM800KPreprocessor
from tests.test_data_types import (
    example_math_aps_path,
    example_math_shepherd_path,
    example_prm800k_path,
)

STEP_TAG = 'च'


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
