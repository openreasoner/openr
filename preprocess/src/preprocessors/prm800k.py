import functools as ft
import multiprocessing as mp
from pathlib import Path
from typing import Iterable

from src.data_types import ConvertedItem
from src.data_types.prm800k import Completion, PRM800KItem, Step
from src.preprocessors.base import PreprocessorBase
from src.preprocessors.utils import dump_converted_ds, read_prm800k_ds


class PRM800KPreprocessor(PreprocessorBase):
    def __init__(
        self,
        ds_path: str | Path,
        step_tag: str,
        suffix: str = "new",
        add_step_prefix: bool | None = None,
        neutral_is_bad: bool | None = None,
    ) -> None:
        super().__init__(ds_path, step_tag, suffix)

        self.add_step_prefix: bool = (
            add_step_prefix if add_step_prefix is not None else False
        )
        self.neutral_is_bad: bool = (
            neutral_is_bad if neutral_is_bad is not None else False
        )

    def _read_ds(self) -> None:
        self.original_items = read_prm800k_ds(self.ds_path)

    def convert(self) -> None:
        self._read_ds()
        assert self.original_items is not None

        convert_fn = ft.partial(
            convert_prm800k_item,
            step_tag=self.step_tag,
            add_step_prefix=self.add_step_prefix,
            neutral_is_good=not self.neutral_is_bad,
        )
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


def convert_prm800k_item(
    original_item: PRM800KItem,
    step_tag: str,
    add_step_prefix: bool = False,
    neutral_is_good: bool = True,
) -> list[ConvertedItem]:
    """Convert the original dataset item to desired format.

    PRM800K may have multiple completions for a single reasoning step. For
    example, a single item in the original dataset has the following steps:
    - Step 1: [choice 1]
    - Step 2: [choice 1, choice 2, choice 3]
    - Step 3: [choice 1, choice 2]
    This item will produce 1x3x2=6 converted items, each contains one possible
    reasoning process that is a combination of these steps:
    - Item 1: [step1-choice1, step2-choice1, step3-choice1]
    - Item 2: [step1-choice1, step2-choice1, step3-choice2]
    - Item 3: [step1-choice1, step2-choice2, step3-choice1], ...

    Args:
      original_item: An `PRM800KItem` instance obtained by parsing a single line
        of the original dataset JSON file.
      neutral_is_good: Cast `neutral` label to `good` if True, `bad` otherwise.
        PRM800K has three labels, while we only need binary classification.

    Returns:
      A list of `ConvertedItem` that contain all combinations of reasoning steps.
    """
    # Some `Step` of the original item may contain all-null `Completion`.
    filtered_steps = filter(
        lambda x: x.completions is not None, original_item.label.steps
    )

    processes_and_labels_for_each_step = extract_processes_and_labels_for_steps(
        filtered_steps, step_tag, add_step_prefix, neutral_is_good
    )  # [([str], [str])]

    saved_processes: list[str] = [""]
    updated_processes: list[str] = []
    saved_labels: list[list[str]] = [[]]
    updated_labels: list[list[str]] = []

    for completions, labels in processes_and_labels_for_each_step:
        for step_cache, label_cache in zip(saved_processes, saved_labels):
            for completion, label in zip(completions, labels):
                updated_processes.append(f"{step_cache} {completion}".strip(" "))
                updated_labels.append(label_cache + [label])

            saved_processes, saved_labels = updated_processes, updated_labels
            updated_processes, updated_labels = [], []

    return list(
        map(
            lambda proc_lbl: ConvertedItem(original_item.question.problem, *proc_lbl),
            zip(saved_processes, saved_labels),
        )
    )


def extract_processes_and_labels_for_steps(
    steps: Iterable[Step],
    step_tag: str,
    add_step_prefix: bool = False,
    neutral_is_good: bool = True,
) -> Iterable[tuple[list[str], list[str]]]:
    """
    For each reason step, return all its contents and labels as a tuple of two
    lists.

    Args:
      steps: A list of `Step` instances.
      neutral_is_good: Cast `neutral` label to `good` if True, otherwise `bad`.

    Returns:
      Tuples of two lists, one contains all completions of this step and the other
      contains corresponding labels. Each tuple corresponds to a step.
    """
    steps_with_ind = enumerate(steps)  # [(ind, Step)] = [(ind, [(text, rating)])]

    def deal_with_one_step(ind: int, step: Step) -> tuple[list[str], list[str]]:
        assert step.completions is not None
        lst_of_proc_lbl_tuple = map(
            lambda completion: get_process_and_label(
                ind, completion, step_tag, add_step_prefix, neutral_is_good
            ),
            step.completions,
        )  # [(str, str)]
        tuple_of_proc_lbl_lst: tuple[list[str], list[str]] = ft.reduce(
            lambda res, tup: (res[0] + [tup[0]], res[1] + [tup[1]]),  # type: ignore
            lst_of_proc_lbl_tuple,
            ([], []),
        )  # ([str], [str])
        return tuple_of_proc_lbl_lst

    steps_of_proc_and_lbl = map(
        lambda x: deal_with_one_step(*x), steps_with_ind
    )  # [([str], [str])]
    return steps_of_proc_and_lbl


def get_process_and_label(
    ind: int,
    completion: Completion,
    step_tag: str,
    add_step_prefix: bool = False,
    neutral_is_good: bool = True,
) -> tuple[str, str]:
    """Extract current reason step of index `ind` and corresponding label."""
    prefix = f"Step {ind + 1}: " if add_step_prefix else ""
    process = f"{prefix}{completion.text} {step_tag}"
    rating = (
        completion.rating if completion.rating is not None else 0
    )  # treat null rating as `neutral`

    if rating == 0:
        label = "+" if neutral_is_good else "-"
    else:
        label = "+" if rating > 0 else "-"

    return process, label
