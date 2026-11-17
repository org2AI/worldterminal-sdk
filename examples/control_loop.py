"""Use your own real robot/simulator adapter; no benchmark imports."""

import time
from contextlib import suppress
from typing import Protocol

from worldterminal_sdk import Client, Observation


class RobotAdapter(Protocol):
    """Device operations implemented by the application."""

    def observe(self, instruction: str) -> Observation:
        """Capture the current state and camera observations."""
        ...

    def execute(self, action: dict) -> None:
        """Execute one action using the configured robot contract."""
        ...

    def stopped(self) -> bool:
        """Report whether the device has stopped its episode."""
        ...

    def outcome(self) -> dict:
        """Return the measured outcome without inferring success from policy termination."""
        ...


def run(robot: RobotAdapter, instruction: str, *, url=None, max_actions=1000, timeout=600):
    """Drive a caller-provided adapter with bounded actions and waiting."""
    deadline = time.monotonic() + timeout
    actions = 0
    with Client(url) as client:
        client.reset()
        try:
            while not robot.stopped():
                if time.monotonic() >= deadline:
                    raise TimeoutError("control loop deadline exceeded")
                result = client.step(robot.observe(instruction))
                while result.waiting:
                    if robot.stopped() or time.monotonic() >= deadline:
                        raise TimeoutError("control loop stopped while waiting")
                    time.sleep(0.01)
                    result = client.poll()
                if result.done:
                    break
                for action in result.actions:
                    if robot.stopped():
                        break
                    if actions >= max_actions:
                        raise RuntimeError("action budget exceeded")
                    robot.execute(action)
                    actions += 1
        except BaseException:
            with suppress(Exception):
                client.end({"success": False, "aborted": True})
            raise
        else:
            outcome = robot.outcome()
            client.end(outcome)
            return outcome
