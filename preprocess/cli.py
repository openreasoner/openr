import argparse
from pathlib import Path
from typing import NamedTuple

from src.preprocessors.base import PreprocessorBase
from src.preprocessors.math_aps import (MathAPSPreprocessor,
                                        MathAPSTreePreprocessor)
from src.preprocessors.math_shepherd import MathShepherdPreprocessor
from src.preprocessors.prm800k import PRM800KPreprocessor


class Args(NamedTuple):
    """
    Stores cli arguments.

    Attributes:
      dataset_type: One of prm800k, math-aps, math-aps-tree, and math-shepherd
      file: Path to the original dataset
      step_tag: Step tag to be appended to each step
      suffix: Suffix to be appended to `file` for the output path
      add_step_prefix: Prepend 'Step i: ' to each step for PRM800K
      neutral_is_bad: Treat `neutral` as negative labels for PRM800K
    """

    dataset_type: str
    file: Path
    step_tag: str
    suffix: str = "new"
    add_step_prefix: bool | None = None
    neutral_is_bad: bool | None = None


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_type",
        type=str,
        help="Which dataset FILE is (prm800k, math-aps, math-aps-tree, or math-shepherd)",
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to the original dataset file",
    )
    parser.add_argument(
        "-t",
        "--step-tag",
        type=str,
        help=r"Step tag. Default: \n\n\n\n\n",
        default="\n\n\n\n\n",
    )
    parser.add_argument(
        "-s",
        "--suffix",
        type=str,
        help="Suffix appended to FILE for the output path. Default: new",
        default="new",
    )
    parser.add_argument(
        "--add-step-prefix",
        action="store_true",
        help='Prepend "Step i: " to each step. PRM800K only',
    )
    parser.add_argument(
        "--neutral-is-bad",
        action="store_true",
        help="Treat `neutral` as negative labels for PRM800K. PRM800K only",
    )

    return Args(**vars(parser.parse_args()))


def main(args: Args) -> None:
    runner_args = {
        "ds_path": args.file,
        "step_tag": args.step_tag,
        "suffix": args.suffix,
        "add_step_prefix": args.add_step_prefix,
        "neutral_is_bad": args.neutral_is_bad,
    }

    runner: PreprocessorBase
    runner = {
        "prm800k": PRM800KPreprocessor,
        "math-aps": MathAPSPreprocessor,
        "math-aps-tree": MathAPSTreePreprocessor,
        "math-shepherd": MathShepherdPreprocessor,
    }[args.dataset_type](**runner_args)
    runner.convert()
    runner.dump()


if __name__ == "__main__":
    main(parse_args())
