"""Exercise the explicit launcher without simulator execution or network calls."""
import hashlib
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from test_robotwin import fake_task

from worldterminal_sdk import robotwin_evaluator as adapter


def loaded_task(tmp_path):
    task = fake_task()
    task.robot.is_dual_arm = True
    for index, side in enumerate(('left', 'right')):
        path = tmp_path / 'aloha-agilex' / f'{side}.urdf'
        path.parent.mkdir(exist_ok=True)
        path.write_text(f'<robot name="actual-{side}"/>')
        fields = {'urdf_path': path, 'entity_origion_pose': SimpleNamespace(
            p=[index, 2., 3.], q=[1., 0., 0., 0.]),
            'arm_joints_name': [f'{side}{i}' for i in range(6)], 'gripper_bias': .12,
            'delta_matrix': np.eye(3), 'global_trans_matrix': np.diag([1., -1., -1.])}
        for name, value in fields.items():
            setattr(task.robot, f'{side}_{name}', value)
    raw = task.get_obs()
    for camera in raw['observation'].values():
        camera['extrinsic_cv'] = np.eye(4, dtype=np.float32)[:3]
        camera['cam2world_gl'] = np.diag([1., -1., -1., 1.])
    task.get_obs = lambda: pytest.fail('The evaluator already captured this observation')
    return task, raw


def test_named_converter_uses_raw_rgb_and_actual_loaded_identity(tmp_path):
    task, raw = loaded_task(tmp_path)
    result = adapter.measured_observation(raw, task_env=task, instruction='actual instruction',
                                          env_idx=7, frequency=20)
    assert result['instruction'] == 'actual instruction'
    assert result['env_idx'] == 7
    assert result['additional_info']['frequency'] == 20
    assert result['state']['left_arm_joint_state'] == [.1] * 6
    assert result['state']['left_ee_joint_state'] == [.25]
    identity = result['state']['robot_identity']
    assert identity['embodiment'] == 'aloha-agilex'
    assert identity['sides']['right']['root_pose'] == [1., 2., 3., 1., 0., 0., 0.]
    for side in ('left', 'right'):
        path = getattr(task.robot, f'{side}_urdf_path')
        assert identity['sides'][side]['urdf_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert np.array_equal(result['vision']['cam_head']['color'],
                          raw['observation']['head_camera']['rgb'])
    with pytest.raises(ValueError, match='actual task_env'):
        adapter.measured_observation(raw)


def test_unknown_official_bytes_rejected_before_import(tmp_path):
    path = tmp_path / 'eval_policy_xpolicylab.py'
    path.write_text('raise AssertionError("must not execute")')
    with pytest.raises(ValueError, match='unchanged RoboTwin'):
        adapter.load_evaluator(path)


@pytest.mark.skipif(not os.environ.get('ROBOTWIN_OFFICIAL_EVALUATOR'),
                    reason='Set ROBOTWIN_OFFICIAL_EVALUATOR to the unchanged pinned source')
def test_real_official_parser_boundary_preserves_source_and_refuses_batch(tmp_path, monkeypatch):
    source = Path(os.environ['ROBOTWIN_OFFICIAL_EVALUATOR'])
    original = source.read_bytes()
    path = tmp_path / 'RoboTwin' / 'scripts' / source.name
    path.parent.mkdir(parents=True)
    path.write_bytes(original)
    envs = ModuleType('envs')
    envs.CONFIGS_PATH = str(tmp_path)
    actor = ModuleType('envs.utils.create_actor')
    actor.UnStableError = RuntimeError
    instructions = ModuleType('generate_episode_instructions')
    instructions.generate_episode_descriptions = lambda *args, **kwargs: None
    for name, module in [('envs', envs), ('envs.utils.create_actor', actor),
                         ('generate_episode_instructions', instructions)]:
        monkeypatch.setitem(sys.modules, name, module)
    previous_path = sys.path[:]
    try:
        module = adapter.load_evaluator(path)
        monkeypatch.setattr(adapter, 'load_evaluator', lambda _: module)
        task, raw = loaded_task(tmp_path)
        calls = []
        renderer = ModuleType('test_render')
        renderer.Sapien_TEST = lambda: calls.append('preflight')
        monkeypatch.setitem(sys.modules, 'test_render', renderer)

        def main(parsed):
            calls.append(('main', parsed['seed'], parsed['task_name']))
            assert parsed['test_num'] == 1
            result = module.robotwin_obs_to_xpolicylab(raw, task_env=task,
                                                     instruction='real parser instruction')
            assert result['state']['robot_identity']['sides']['left']['joint_names'] == [
                f'left{i}' for i in range(6)]
            assert result['instruction'] == 'real parser instruction'
            return 'terminal result'

        monkeypatch.setattr(module, 'main', main)
        official_args = ['--bench_name', 'robotwin', '--task_name', 'click_bell',
                         '--policy_name', 'WorldTerminal', '--port', '8080', '--seed', '0',
                         '--task_config', 'demo_randomized', '--test_num', '1']
        previous_argv = sys.argv
        assert adapter.run_evaluator(path, official_args) == 'terminal result'
        assert calls == ['preflight', ('main', 0, 'click_bell')]
        assert sys.argv is previous_argv
        assert path.read_bytes() == source.read_bytes() == original
        with pytest.raises(ValueError, match='sequential eval only'):
            adapter.run_evaluator(path, [*official_args, '--eval_batch', 'true'])
        assert calls[-1] == 'preflight'
        assert path.read_bytes() == original
    finally:
        sys.path[:] = previous_path
