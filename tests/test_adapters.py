from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest
from worldterminal_sdk import ActionBatch

ROOT = Path(__file__).resolve().parents[1]


def load_adapter(name, path):
    spec = spec_from_file_location(name, ROOT / path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Session:
    def __init__(self, rows):
        self.rows = iter(rows)
        self.observations = []
        self.verdict = None

    def reset(self):
        pass

    def step(self, observation):
        self.observations.append(observation)
        return self.poll()

    def poll(self):
        return ActionBatch.from_wire(next(self.rows))

    def end(self, result):
        self.verdict = result


class LiberoEnv:
    def __init__(self):
        self.actions = []
        self.raw = {key: np.zeros(size) for key, size in (
            ("robot0_eef_pos", 3), ("robot0_eef_quat", 4), ("robot0_joint_pos", 7),
            ("robot0_gripper_qpos", 2)
        )}
        for name in ("agentview", "robot0_eye_in_hand"):
            self.raw[name + "_image"] = np.zeros((128, 128, 3), np.uint8)

    def reset(self):
        pass

    def set_init_state(self, initial_state):
        return self.raw

    def step(self, action):
        self.actions.append(action)
        return self.raw, 0, self.check_success(), {}

    def check_success(self):
        return len(self.actions) == 2


def test_libero_wait_and_execute_until_environment_success():
    runner = load_adapter("libero_adapter", "adapters/libero/evaluate.py")
    row = {"action": [0.25] * 7}
    session = Session([{"status": "waiting", "actions": []},
                       {"status": "ready", "actions": [row, row, row]}])
    env = LiberoEnv()
    assert runner.run_episode(env, None, "pick up cup", session, 10, 10) == {"success": True, "steps": 2}
    assert len(session.observations) == 1
    assert session.observations[0].instruction == "pick up cup"
    assert len(env.actions) == 2


def test_libero_rejects_invalid_action_before_execution():
    runner = load_adapter("libero_adapter", "adapters/libero/evaluate.py")
    env = LiberoEnv()
    session = Session([{"status": "ready", "actions": [{"action": [0] * 6}]}])
    with pytest.raises(ValueError, match="seven"):
        runner.run_episode(env, None, "pick up cup", session, 10, 10)
    assert not env.actions and session.verdict["aborted"]


def test_robotwin_polls_until_action_chunk_and_forwards_end(monkeypatch):
    pytest.importorskip("XPolicyLab.model_template")
    adapter = load_adapter("worldterminal_model", "adapters/xpolicylab/WorldTerminal/model.py")
    row = {"left_arm_joint_state": [0.5] * 6}
    session = Session([{"status": "waiting", "actions": []},
                       {"status": "ready", "actions": [row]}, {"status": "done", "actions": []}])
    monkeypatch.setattr(adapter, "Client", lambda *args, **kwargs: session)
    model = adapter.Model({"bench_name": "RoboTwin"})
    model.reset()
    assert model.get_action() == [row]
    assert model.get_action() == []
    model.on_trial_end({"success": False})
    assert session.verdict == {"success": False}
