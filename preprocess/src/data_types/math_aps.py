from dataclasses import dataclass
from typing import Any

from src.data_types.base import OriginalItemBase
from src.data_types.utils import from_float, from_list, from_str, to_dict, to_float


@dataclass
class State:
    rollout: str
    state: str
    mcs: float

    @staticmethod
    def from_dict(obj: Any) -> "State":
        assert isinstance(obj, dict)

        rollout = from_str(obj.get("rollout"))
        state = from_str(obj.get("state"))
        mcs = from_float(obj.get("mcs"))

        return State(rollout, state, mcs)

    def to_dict(self) -> dict:
        return dict(
            rollout=from_str(self.rollout),
            state=from_str(self.state),
            mcs=to_float(self.mcs),
        )


@dataclass
class MathAPSItem(OriginalItemBase):
    q: str
    states: list[State]

    @staticmethod
    def from_dict(obj: Any) -> "MathAPSItem":
        assert isinstance(obj, dict)

        q = from_str(obj.get("q"))
        states = from_list(State.from_dict, obj.get("states"))

        return MathAPSItem(q, states)

    def to_dict(self) -> dict:
        return dict(
            q=from_str(self.q),
            states=from_list(lambda x: to_dict(State, x), self.states),
        )
