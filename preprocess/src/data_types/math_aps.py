from dataclasses import dataclass
from typing import Any

from src.data_types.base import OriginalItemBase
from src.data_types.utils import (from_float, from_int, from_list, from_str,
                                  to_dict, to_float)


@dataclass
class ReasoningStep:
    solution_prefix: str
    mc_value: float

    @staticmethod
    def from_dict(obj: Any) -> "ReasoningStep":
        assert isinstance(obj, dict)

        solution_prefix = from_str(obj.get("solution_prefix"))
        mc_value = from_float(obj.get("mc_value"))

        return ReasoningStep(solution_prefix, mc_value)

    def to_dict(self) -> dict:
        return dict(
            solution_prefix=from_str(self.solution_prefix),
            mc_value=from_float(self.mc_value),
        )


@dataclass
class MathAPSItem(OriginalItemBase):
    question_id: int
    question: str
    final_answer: str
    reasoning_steps: list[ReasoningStep]

    @staticmethod
    def from_dict(obj: Any) -> "MathAPSItem":
        assert isinstance(obj, dict)

        question_id = from_int(obj.get("question_id"))
        question = from_str(obj.get("question"))
        final_answer = from_str(obj.get("final_answer"))
        reasoning_steps = from_list(ReasoningStep.from_dict, obj.get("reasoning_steps"))

        return MathAPSItem(question_id, question, final_answer, reasoning_steps)

    def to_dict(self) -> dict:
        return dict(
            question_id=from_int(self.question_id),
            question=from_str(self.question),
            final_answer=from_str(str(self.final_answer)),
            reasoning_steps=from_list(
                lambda x: to_dict(ReasoningStep, x), self.reasoning_steps
            ),
        )


@dataclass
class ReasoningNode:
    text: str
    mc_value: float
    children: list["ReasoningNode"]

    @staticmethod
    def from_dict(obj: Any) -> "ReasoningNode":
        assert isinstance(obj, dict)

        text = from_str(obj.get("text"))
        mc_value = from_float(obj.get("mc_value"))
        children = from_list(ReasoningNode.from_dict, obj.get("children"))

        return ReasoningNode(text, mc_value, children)

    def to_dict(self) -> dict:
        return dict(
            text=from_str(self.text),
            mc_value=to_float(self.mc_value),
            children=from_list(lambda x: to_dict(ReasoningNode, x), self.children),
        )


@dataclass
class MathAPSItemTree(OriginalItemBase):
    question_id: int
    question: str
    final_answer: str
    reasoning_steps: ReasoningNode

    @staticmethod
    def from_dict(obj: Any) -> "MathAPSItemTree":
        assert isinstance(obj, dict)

        question_id = from_int(obj.get("question_id"))
        question = from_str(obj.get("question"))
        final_answer = from_str(obj.get("final_answer"))
        reasoning_steps = ReasoningNode.from_dict(obj.get("reasoning_steps"))

        return MathAPSItemTree(question_id, question, final_answer, reasoning_steps)

    def to_dict(self) -> dict:
        return dict(
            question_id=from_int(self.question_id),
            question=from_str(self.question),
            final_answer=from_str(self.final_answer),
            reasoning_steps=to_dict(ReasoningNode, self.reasoning_steps),
        )
