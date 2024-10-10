from abc import ABC, abstractmethod
from pathlib import Path

from src.data_types.base import OriginalItemBase
from src.data_types.converted import ConvertedItem


class PreprocessorBase(ABC):
  """
  Base class for preprocessors.

  Attributes:
    ds_path: Path to the original dataset file. Type: Path
    output_path: Path to the preprocessed dataset file (a JSON file). Type: Path
    step_tag: Step tag for the new dataset. Type: str
    original_items: Items of the original data. Optional, init by _read_ds()
    converted_items: Items of processed data. Optional, init by convert()

  Methods:
    convert() -> None: Read the original dataset from `ds_path` and dump the
      processed one to `output_path`
  """

  def __init__(self,
               ds_path: str | Path,
               step_tag: str,
               suffix: str = 'new') -> None:
    self.ds_path: Path = Path(ds_path).resolve()
    self.output_path: Path = self.ds_path.with_suffix(f'.{suffix}.json')
    self.step_tag: str = step_tag
    self.original_items: list[OriginalItemBase] | None = None
    self.converted_items: list[ConvertedItem] | None = None

  @abstractmethod
  def _read_ds(self) -> None:
    ...

  @abstractmethod
  def convert(self) -> None:
    ...
