from abc import ABC, abstractmethod
from pathlib import Path


class PreprocessorBase(ABC):
  """
  Base class for preprocessors.

  Attributes:
    ds_path: Path to the original dataset file. Type: Path
    output_path: Path to the preprocessed dataset file (a JSON file). Type: Path
    step_tag: Step tag for the new dataset. Type: str

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

  @abstractmethod
  def convert(self) -> None:
    ...
