from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from src.data_types.base import OriginalItemBase
from src.data_types.utils import (
    from_bool,
    from_datetime,
    from_float,
    from_int,
    from_list,
    from_optional,
    from_str,
    to_dict,
)


@dataclass
class Completion:
  text: str
  rating: int | None = None
  flagged: bool | None = None
  source: str | None = None

  @staticmethod
  def from_dict(obj: Any) -> 'Completion':
    assert isinstance(obj, dict)

    text = from_str(obj.get('text'))
    rating = from_optional(from_int, obj.get('rating'))
    flagged = from_optional(from_bool, obj.get('flagged'))
    source = from_optional(from_str, obj.get('source'))

    return Completion(text, rating, flagged, source)

  def to_dict(self) -> dict:
    res = dict(
        text=from_str(self.text),
        rating=from_optional(from_int, self.rating),
        flagged=from_optional(from_bool, self.flagged),
    )
    if self.source is not None:
      res['source'] = from_optional(from_str, self.source)

    return res


@dataclass
class Step:
  completions: list[Completion] | None = None
  human_completion: Completion | None = None
  chosen_completion: int | None = None

  @staticmethod
  def from_dict(obj: Any) -> 'Step':
    assert isinstance(obj, dict)

    completions = from_optional(lambda x: from_list(Completion.from_dict, x),
                                obj.get('completions'))
    human_completion = from_optional(Completion.from_dict,
                                     obj.get('human_completion'))
    chosen_completion = from_optional(from_int, obj.get('chosen_completion'))

    return Step(completions, human_completion, chosen_completion)

  def to_dict(self) -> dict:
    return dict(
        completions=from_optional(
            lambda x: from_list(lambda x: to_dict(Completion, x), x),
            self.completions),
        human_completion=from_optional(lambda x: to_dict(Completion, x),
                                       self.human_completion),
        chosen_completion=from_optional(from_int, self.chosen_completion),
    )


@dataclass
class Label:
  steps: list[Step]
  total_time: int
  finish_reason: str

  @staticmethod
  def from_dict(obj: Any) -> 'Label':
    assert isinstance(obj, dict)

    steps = from_list(Step.from_dict, obj.get('steps'))
    total_time = from_int(obj.get('total_time'))
    finish_reason = from_str(obj.get('finish_reason'))

    return Label(steps, total_time, finish_reason)

  def to_dict(self) -> dict:
    return dict(
        steps=from_list(lambda x: to_dict(Step, x), self.steps),
        total_time=from_int(self.total_time),
        finish_reason=from_str(self.finish_reason),
    )


@dataclass
class Question:
  problem: str
  ground_truth_answer: str | None = None
  ground_truth_solution: str | None = None
  pre_generated_steps: list[str] | None = None
  pre_generated_answer: str | None = None
  pre_generated_verifier_score: float | None = None

  @staticmethod
  def from_dict(obj: Any) -> 'Question':
    assert isinstance(obj, dict)

    problem = from_str(obj.get('problem'))
    ground_truth_answer = from_optional(from_str,
                                        obj.get('ground_truth_answer'))
    ground_truth_solution = from_optional(from_str,
                                          obj.get('ground_truth_solution'))
    pre_generated_steps = from_optional(lambda x: from_list(from_str, x),
                                        obj.get('pre_generated_steps'))
    pre_generated_answer = from_optional(from_str,
                                         obj.get('pre_generated_answer'))
    pre_generated_verifier_score = from_optional(
        from_float, obj.get('pre_generated_verifier_score'))

    return Question(problem, ground_truth_answer, ground_truth_solution,
                    pre_generated_steps, pre_generated_answer,
                    pre_generated_verifier_score)

  def to_dict(self) -> dict:
    res: dict
    res = dict(
        problem=from_str(self.problem),
        ground_truth_answer=from_optional(from_str, self.ground_truth_answer),
    )
    if self.ground_truth_solution is not None:
      res['ground_truth_solution'] = from_optional(from_str,
                                                   self.ground_truth_solution)
    if self.pre_generated_steps is not None:
      res['pre_generated_steps'] = from_optional(
          lambda x: from_list(from_str, x), self.pre_generated_steps)
    if self.pre_generated_answer is not None:
      res['pre_generated_answer'] = from_optional(from_str,
                                                  self.pre_generated_answer)
    if self.pre_generated_verifier_score is not None:
      res['pre_generated_verifier_score'] = from_optional(
          from_float, self.pre_generated_verifier_score)

    return res


@dataclass
class PRM800KItem(OriginalItemBase):
  labeler: UUID
  timestamp: datetime
  is_quality_control_question: bool
  is_initial_screening_question: bool
  question: Question
  label: Label
  generation: int | None = None

  @staticmethod
  def from_dict(obj: Any) -> 'PRM800KItem':
    assert isinstance(obj, dict)

    labeler = UUID(obj.get('labeler'))
    timestamp = from_datetime(obj.get('timestamp'))
    is_quality_control_question = from_bool(
        obj.get('is_quality_control_question'))
    is_initial_screening_question = from_bool(
        obj.get('is_initial_screening_question'))
    question = Question.from_dict(obj.get('question'))
    label = Label.from_dict(obj.get('label'))
    generation = from_optional(from_int, obj.get('generation'))

    return PRM800KItem(labeler, timestamp, is_quality_control_question,
                       is_initial_screening_question, question, label,
                       generation)

  def to_dict(self) -> dict:
    return dict(
        labeler=f'{self.labeler}',
        timestamp=self.timestamp.isoformat(),
        is_quality_control_question=from_bool(self.is_quality_control_question),
        is_initial_screening_question=from_bool(
            self.is_initial_screening_question),
        question=to_dict(Question, self.question),
        label=to_dict(Label, self.label),
        generation=from_optional(from_int, self.generation),
    )
