from dataclasses import dataclass
from typing import Any

from src.data_types.utils import from_list, from_str


@dataclass
class ConvertedItem:
    question: str
    process: str
    label: list[str]

    @staticmethod
    def from_dict(obj: Any) -> "ConvertedItem":
        assert isinstance(obj, dict)

        question = from_str(obj.get("question"))
        process = from_str(obj.get("process"))
        label = from_list(from_str, obj.get("label"))

        return ConvertedItem(question, process, label)

    def to_dict(self) -> dict:
        return dict(
            question=from_str(self.question),
            process=from_str(self.process),
            label=from_list(from_str, self.label),
        )

    def __hash__(self) -> int:
        labels = " ".join(self.label)
        return hash(f"{self.question} {self.process} {labels}")
