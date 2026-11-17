"""Public observation and action containers; robot drivers stay with the caller."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Observation:
    """State/vision keys and action units follow the selected server's robot contract.

    Color images are RGB uint8 arrays. Camera names, joint order and units must
    match the endpoint; this container does not perform robot calibration.
    """

    state: Mapping[str, Any]
    vision: Mapping[str, Any]
    instruction: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self):
        """Return the wire observation without changing numeric arrays."""
        return {
            "state": dict(self.state),
            "vision": dict(self.vision),
            "instruction": self.instruction,
            "additional_info": dict(self.metadata),
        }


@dataclass(frozen=True)
class ActionBatch:
    """Actions to execute, a wait response, or a stopped policy."""

    actions: tuple[dict[str, Any], ...]
    waiting: bool = False

    @property
    def done(self):
        """The policy stopped; not a claim that the physical task succeeded."""
        return not self.waiting and not self.actions

    @classmethod
    def from_wire(cls, value):
        """Validate the result before exposing any action to the caller."""
        if not isinstance(value, dict):
            raise ValueError("action result must contain status and actions")
        status, rows = value.get("status"), value.get("actions")
        if status not in ("ready", "waiting", "done"):
            raise ValueError(f"unknown action status: {status!r}")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("actions must be a list of action dictionaries")
        if bool(rows) != (status == "ready"):
            raise ValueError(f"inconsistent action list for status {status!r}")
        return cls(tuple(rows), waiting=status == "waiting")
