"""RoboTwin terminal holds use measurements while the official loop judges the episode."""

import copy
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest
from test_adapters import load_adapter
from test_robotwin import fake_task
from test_robotwin_evaluator import loaded_task

from worldterminal_sdk import ActionBatch, robotwin_evaluator
from worldterminal_sdk.robotwin import measured_hold, observation


class Client:
    def __init__(self, batches):
        self.batches = iter(batches)
        self.calls = []
        self.done = False

    def reset(self):
        self.calls.append(("reset",))
        self.done = False
        return "reset-result"

    def update_observation(self, value):
        assert not self.done, "ended worker received another observation"
        self.calls.append(("update", value))
        return "update-result"

    def poll(self):
        assert not self.done, "ended worker was polled again"
        self.calls.append(("poll",))
        batch = next(self.batches)
        self.done = batch.done
        return batch

    def get_action(self):
        self.calls.append(("get_action",))
        return {"status": "done", "actions": []}

    def end(self, result):
        self.calls.append(("end", result))


@pytest.fixture
def model_factory(monkeypatch):
    pytest.importorskip("XPolicyLab.model_template")
    adapter = load_adapter(
        "terminal_hold_model", "adapters/xpolicylab/WorldTerminal/model.py"
    )

    def make(batches, *, robotwin=True):
        client = Client(batches)
        monkeypatch.setattr(adapter, "Client", lambda *args, **kwargs: client)
        return adapter.Model(
            {"bench_name": "RoboTwin" if robotwin else "RoboDojo"}
        ), client

    return make


def measured_observation():
    return observation(
        fake_task(), instruction="test", identity={"embodiment": "aloha-agilex"}
    )


def test_terminal_holds_refresh_locally_without_polling_or_sending_to_ended_worker(
    model_factory,
):
    model, client = model_factory([ActionBatch((), waiting=True), ActionBatch(())])
    model.reset()
    first = measured_observation()
    model.update_obs(first)
    held = model.get_action()
    assert held == [measured_hold(first)]
    held[0]["left_arm_joint_state"][0] = 999
    first["state"]["left_arm_joint_state"][0] = 999
    assert model.get_action()[0]["left_arm_joint_state"][0] == 0.1
    latest = copy.deepcopy(first)
    latest["state"]["left_arm_joint_state"] = [0.3] * 6
    latest["state"]["left_ee_joint_state"] = [0.4]
    assert model.update_obs(latest) is None
    assert model.get_action() == [measured_hold(latest)]
    assert [call[0] for call in client.calls] == ["reset", "update", "poll", "poll"]


def test_reset_discards_terminal_state_and_forwards_the_next_episode(model_factory):
    ready = measured_hold(measured_observation())
    model, client = model_factory([ActionBatch(()), ActionBatch((ready,))])
    model.reset()
    model.update_obs(measured_observation())
    model.get_action()
    model.reset()
    assert model.update_obs(measured_observation()) == "update-result"
    assert model.get_action() == [ready]
    assert [call[0] for call in client.calls] == [
        "reset",
        "update",
        "poll",
        "reset",
        "update",
        "poll",
    ]


def test_missing_state_fails_closed_instead_of_reusing_an_earlier_hold(model_factory):
    model, client = model_factory([ActionBatch(())])
    model.reset()
    model.update_obs(measured_observation())
    model.get_action()
    with pytest.raises(ValueError, match="measured"):
        model.update_obs({"state": {}})
    with pytest.raises(ValueError, match="fresh measured"):
        model.get_action()
    assert [call[0] for call in client.calls] == ["reset", "update", "poll"]


def test_done_before_any_observation_has_no_invented_action(model_factory):
    model, client = model_factory([ActionBatch(())])
    model.reset()
    with pytest.raises(ValueError, match="fresh measured"):
        model.get_action()
    assert [call[0] for call in client.calls] == ["reset", "poll"]


@pytest.mark.parametrize("change", ("missing", "short", "nan", "opening"))
def test_hold_validation_rejects_incomplete_or_invalid_measurements(change):
    value = measured_observation()
    state = value["state"]
    if change == "missing":
        del state["right_ee_joint_state"]
    elif change == "short":
        state["left_arm_joint_state"] = [0] * 5
    elif change == "nan":
        state["left_arm_joint_state"][0] = float("nan")
    else:
        state["left_ee_joint_state"] = [1.1]
    with pytest.raises(ValueError, match="measured"):
        measured_hold(value)


def test_robodojo_client_call_sequence_and_raw_done_response_are_unchanged(
    model_factory,
):
    model, client = model_factory([], robotwin=False)
    value = {"state": {}}
    assert model.reset() == "reset-result"
    assert model.update_obs(value) == "update-result"
    assert model.get_action() == {"status": "done", "actions": []}
    model.on_trial_end({"success": False})
    assert client.calls == [
        ("reset",),
        ("update", value),
        ("get_action",),
        ("end", {"success": False}),
    ]


@pytest.fixture
def official_module(tmp_path, monkeypatch):
    path = os.environ.get("ROBOTWIN_OFFICIAL_EVALUATOR")
    if not path:
        pytest.skip("Set ROBOTWIN_OFFICIAL_EVALUATOR to the unchanged pinned source")
    envs = ModuleType("envs")
    envs.CONFIGS_PATH = str(tmp_path)
    actors = ModuleType("envs.utils.create_actor")
    actors.UnStableError = type("UnStableError", (RuntimeError,), {})
    instructions = ModuleType("generate_episode_instructions")
    instructions.generate_episode_descriptions = lambda *args, **kwargs: None
    for name, module in [
        ("envs", envs),
        ("envs.utils.create_actor", actors),
        ("generate_episode_instructions", instructions),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    previous_path = sys.path[:]
    module = robotwin_evaluator.load_evaluator(Path(path))
    monkeypatch.setattr(
        module, "robotwin_obs_to_xpolicylab", robotwin_evaluator.measured_observation
    )
    try:
        yield module
    finally:
        sys.path[:] = previous_path


@pytest.mark.parametrize("success_after", (None, 1))
def test_unmodified_official_loop_counts_terminal_hold_failure_without_seed_retry(
    tmp_path, model_factory, official_module, success_after
):
    task, raw = loaded_task(tmp_path)
    task.step_lim, task.render_freq, task.eval_video_path, task.plan_success = (
        3,
        0,
        None,
        True,
    )
    task.get_obs = lambda: raw
    task.get_instruction = lambda: task.instruction
    task.set_instruction = lambda instruction: setattr(task, "instruction", instruction)
    setups, actions = [], []

    def setup_demo(*, now_ep_num, seed, **kwargs):
        setups.append(seed)
        task.take_action_cnt, task.eval_success = 0, False

    task.setup_demo = setup_demo
    task.play_once = lambda: {"info": {}}
    task.check_success = lambda: True  # The fake expert phase accepts this seed.
    task.close_env = lambda **kwargs: None
    for side in ("left", "right"):
        setattr(
            task.robot,
            f"get_{side}_arm_real_jointState",
            lambda: [0.1 + task.take_action_cnt * 0.01] * 6 + [0.9],
        )
        getattr(task.robot, f"{side}_entity").get_qpos = lambda: [
            0.25 + task.take_action_cnt * 0.01
        ]

    def take_action(row, action_type):
        assert action_type == "qpos"
        if success_after is None:
            assert row[:6] == pytest.approx([0.1 + task.take_action_cnt * 0.01] * 6)
            assert row[6] == pytest.approx(0.25 + task.take_action_cnt * 0.01)
        actions.append(row.copy())
        task.take_action_cnt += 1
        task.eval_success = task.take_action_cnt == success_after

    task.take_action = take_action
    ready = measured_hold(measured_observation())
    batches = (
        [ActionBatch(())] if success_after is None else [ActionBatch((ready, ready))]
    )
    model, client = model_factory(batches)

    class OfficialClient:
        _robotwin_protocol = "ws"

        def call(self, *, func_name, obs=None):
            if func_name == "trial_end":
                return model.on_trial_end(obs)
            if func_name in ("prepare_case", "update_obs"):
                return getattr(model, func_name)(obs)
            return getattr(model, func_name)()

    args = {
        "task_name": "test_task",
        "policy_name": "WorldTerminal",
        "task_config": "test",
        "ckpt_setting": "test",
        "clear_cache_freq": 1,
        "render_freq": 0,
    }
    source = Path(os.environ["ROBOTWIN_OFFICIAL_EVALUATOR"])
    before = source.read_bytes()
    next_seed, successes = official_module.eval_remote_policy(
        "test_task",
        task,
        args,
        OfficialClient(),
        100000,
        usr_args={"action_type": "joint"},
        test_num=1,
        instruction_type=None,
    )
    assert setups == [100000, 100000] and next_seed == 100001
    assert task.test_num == 1 and successes == (0 if success_after is None else 1)
    assert len(actions) == (3 if success_after is None else 1)
    assert client.calls[-1] == (
        "end",
        {
            "task_name": "test_task",
            "seed": 100000,
            "success": success_after is not None,
        },
    )
    assert [call[0] for call in client.calls] == ["reset", "update", "poll", "end"]
    assert source.read_bytes() == before
