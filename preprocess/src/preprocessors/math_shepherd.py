import functools as ft
import multiprocessing as mp
import re
import sys
from pathlib import Path

from src.data_types import ConvertedItem
from src.data_types.math_shepherd import MathShepherdItem
from src.preprocessors.base import PreprocessorBase
from src.preprocessors.utils import dump_converted_ds, read_math_shepherd_ds


class MathShepherdPreprocessor(PreprocessorBase):

  def __init__(
      self,
      ds_path: str | Path,
      step_tag: str,
      suffix: str = 'new',
      verbose: bool = False,
  ) -> None:
    super().__init__(ds_path, step_tag, suffix)

    self.verbose = verbose

  def _read_ds(self) -> None:
    original_items = read_math_shepherd_ds(self.ds_path)
    # only look to items that contain only one 'Step 1: '
    items = filter(lambda x: x.input.count('Step 1: ') == 1, original_items)
    self.original_items = list(items)

  def convert(self) -> None:
    self._read_ds()
    assert self.original_items is not None

    convert_fn = ft.partial(convert_math_shepherd_item, step_tag=self.step_tag)
    with mp.Pool(mp.cpu_count()) as p:
      res = p.map(convert_fn, self.original_items)

    self.converted_items = list(filter(lambda x: len(x.label) > 0, res))

  def dump(self) -> None:
    assert self.converted_items is not None
    dump_converted_ds(self.output_path, list(self.converted_items))


def convert_math_shepherd_item(
    original_item: MathShepherdItem,
    step_tag: str,
    verbose: bool = False,
) -> ConvertedItem:
  # no preceding 'Step i: ' and trailing \n
  steps_with_tags: list[str]
  steps_with_labels: list[str]

  question, steps_with_tags = split_question_and_rationale(original_item.input)
  _, steps_with_labels = split_question_and_rationale(original_item.label)

  steps: list[str] = remove_step_tag_of_math_shepherd(steps_with_tags)
  labels: list[str] = extract_labels_of_math_shepherd(steps_with_labels)

  if '' in labels:  # some step isn't labelled nor have step tag
    if verbose:
      print(
          '\x1b[1;31mWARNING\x1b[0m Lack of step tag detected! '
          'Will be removed.\n'
          f'{original_item}',
          file=sys.stderr,
      )
    return ConvertedItem('', '', [])

  assert len(steps) == len(labels), '# of extracted labels != # of steps'

  rationale = f' {step_tag} '.join(steps + ['']).strip(' ')

  assert rationale.endswith(step_tag), 'Rationale not ends with step_tag!'

  return ConvertedItem(question, rationale, labels)


def split_question_and_rationale(string: str) -> tuple[str, list[str]]:
  """
  Split question and reasoning steps.

  First split question and rationale by 'Step 1: '. Then split reasoning steps
  by 'Step i: '.
  """
  question, rationale = string.split('Step 1: ')
  separator = re.compile(r'Step \d*:')
  # cautious about this strip when step_tag is whitespace
  steps = list(
      map(lambda x: x.strip(),
          separator.sub('च', rationale).split('च')))

  return question.strip(), steps


def remove_step_tag_of_math_shepherd(steps_with_tags: list[str]) -> list[str]:

  def remove_trailing_tag(step: str, tag: str) -> str:
    return step[:step.find(tag)].strip()

  return list(map(lambda x: remove_trailing_tag(x, 'ки'), steps_with_tags))


def extract_labels_of_math_shepherd(steps_with_labels: list[str]) -> list[str]:

  def extract_label(step: str) -> str:
    label = step.strip()[-1]
    if label in ['+', '-']:
      return label
    else:
      return ''

  return list(map(extract_label, steps_with_labels))
