from dataclasses import dataclass
from typing import Any

from src.data_types.base import OriginalItemBase
from src.data_types.utils import from_str


@dataclass
class MathShepherdItem(OriginalItemBase):
    input: str
    label: str
    task: str

    @staticmethod
    def from_dict(obj: Any) -> "MathShepherdItem":
        assert isinstance(obj, dict)

        input = from_str(obj.get("input"))
        label = from_str(obj.get("label"))
        task = from_str(obj.get("task"))

        return MathShepherdItem(input, label, task)

    def to_dict(self) -> dict:
        return dict(
            input=from_str(self.input),
            label=from_str(self.label),
            task=from_str(self.task),
        )
