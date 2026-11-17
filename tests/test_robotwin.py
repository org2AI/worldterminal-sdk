from types import SimpleNamespace

import numpy as np
import pytest

from worldterminal_sdk import ActionBatch, robotwin_aloha
from worldterminal_sdk.robotwin import action_vector, camera_to_world_gl, observation, run_episode


def fake_task():
    task = SimpleNamespace(take_action_cnt=0, step_lim=2, eval_success=False)
    robot = SimpleNamespace()
    for side in ('left', 'right'):
        finger = object()
        entity = SimpleNamespace(get_qpos=lambda: [.25], get_active_joints=lambda f=finger: [f])
        setattr(robot, side + '_entity', entity)
        setattr(robot, side + '_gripper', [[finger]])
        setattr(robot, side + '_gripper_scale', [0, 1])
        setattr(robot, f'get_{side}_arm_real_jointState', lambda: [0.1] * 6 + [0.9])
        setattr(robot, f'get_{side}_arm_jointState', lambda: [0.8] * 6 + [0.9])
        setattr(robot, f'get_{side}_gripper_val', lambda: .9)
        for kind in ('tcp', 'ee'):
            setattr(robot, f'get_{side}_{kind}_pose', lambda: [0, 0, 1, 1, 0, 0, 0])
    task.robot = robot
    cameras = {name: dict(rgb=np.zeros((2, 3, 3), np.uint8), intrinsic_cv=np.eye(3),
                         extrinsic_cv=np.eye(4)) for name in
               ('head_camera', 'left_camera', 'right_camera')}
    task.get_obs = lambda: {'observation': cameras, 'joint_action': {'vector': [999] * 14}}
    return task


def test_measurement_and_cv_conversion_do_not_echo_command():
    obs = observation(fake_task(), instruction='move cup', identity={'embodiment': 'aloha-agilex'})
    assert obs['state']['left_arm_joint_state'] == [.1] * 6
    assert obs['state']['left_ee_joint_state'] == [.25]
    assert obs['state']['left_gripper_command'] == .9
    assert obs['state']['qpos_target14'] == ([.8] * 6 + [.9]) * 2
    assert obs['vision']['cam_head']['extrinsic_matrix'] == np.diag([1, -1, -1, 1]).tolist()
    cv = np.eye(4)
    cv[:3, 3] = [1, 2, 3]
    assert np.allclose(camera_to_world_gl(cv), np.linalg.inv(cv) @ np.diag([1, -1, -1, 1]))
    assert robotwin_aloha()['id'] == 'robotwin/aloha-agilex/joint-v1'


def test_wait_consumes_no_native_step_and_official_verdict_stands(monkeypatch):
    import worldterminal_sdk.robotwin as adapter
    task = fake_task()
    monkeypatch.setattr(adapter, 'robot_identity', lambda _: {'embodiment': 'aloha-agilex'})
    calls = []
    def take_action(row, action_type):
        calls.append(row)
        assert action_type == 'qpos'
        task.take_action_cnt += 1
        task.eval_success = task.take_action_cnt == 2
    task.take_action = take_action
    row = {f'{side}_{kind}_joint_state': [value] * size for side in ('left', 'right')
           for kind, value, size in [('arm', .5, 6), ('ee', .2, 1)]}
    replies = iter([ActionBatch((), waiting=True), ActionBatch((row, row, row))])
    results = []
    client = SimpleNamespace(reset=lambda: None, step=lambda _: next(replies),
                             poll=lambda: next(replies), end=results.append)
    assert run_episode(task, client, instruction='move cup') == {'success': True, 'step': 2, 'step_limit': 2}
    assert len(calls) == 2 and results[-1]['success'] is True
    row['left_ee_joint_state'] = [1.1]
    with pytest.raises(ValueError):
        action_vector(row)


def test_sapien_native_three_by_four_extrinsic_and_gl_matrix_are_equivalent():
    task = fake_task()
    raw = task.get_obs()
    for camera in raw['observation'].values():
        camera['extrinsic_cv'] = np.eye(4, dtype=np.float32)[:3]
        camera['cam2world_gl'] = np.diag([1., -1., -1., 1.]).astype(np.float32)
    task.get_obs = lambda: raw
    measured = observation(task, instruction='move cup', identity={'embodiment': 'aloha-agilex'})
    assert measured['vision']['cam_head']['extrinsic_matrix'] == np.diag([1., -1., -1., 1.]).tolist()
    camera = raw['observation']['head_camera']
    camera['cam2world_gl'][0, 3] = 1.
    with pytest.raises(ValueError, match='calibration disagree'):
        observation(task, instruction='move cup', identity={'embodiment': 'aloha-agilex'})
