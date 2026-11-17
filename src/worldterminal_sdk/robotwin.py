"""RoboTwin observations from measured simulator state, independent of WT runtime code."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

CAMERAS = {'cam_head': 'head_camera', 'cam_left_wrist': 'left_camera',
           'cam_right_wrist': 'right_camera'}


def robot_identity(robot):
    sides, labels = {}, []
    for side in ('left', 'right'):
        path = Path(getattr(robot, f'{side}_urdf_path'))
        labels.append(next((p.name for p in path.parents
                            if p.name in ('aloha-agilex', 'ARX-X5')), None))
        pose = getattr(robot, f'{side}_entity_origion_pose')
        sides[side] = dict(
            urdf_path=str(path), urdf_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            joint_names=list(getattr(robot, f'{side}_arm_joints_name')),
            gripper_scale=list(getattr(robot, f'{side}_gripper_scale')),
            gripper_bias_m=float(getattr(robot, f'{side}_gripper_bias')),
            root_pose=[float(v) for v in (*pose.p, *pose.q)],
            delta_matrix=np.asarray(getattr(robot, f'{side}_delta_matrix')).tolist(),
            global_trans_matrix=np.asarray(getattr(robot, f'{side}_global_trans_matrix')).tolist())
    return dict(embodiment=labels[0] if labels[0] == labels[1] else labels,
                dual_arm=bool(robot.is_dual_arm), sides=sides)


def measured_opening(robot, side):
    entity = getattr(robot, f'{side}_entity')
    joint = getattr(robot, f'{side}_gripper')[0][0]
    low, high = map(float, getattr(robot, f'{side}_gripper_scale'))
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError('RoboTwin gripper scale must increase')
    value = entity.get_qpos()[entity.get_active_joints().index(joint)]
    return float(np.clip((value - low) / (high - low), 0, 1))


def camera_to_world_gl(world_to_camera_cv):
    matrix = np.asarray(world_to_camera_cv, dtype=float)
    if matrix.shape == (3, 4):
        matrix = np.vstack((matrix, [0., 0., 0., 1.]))
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('camera calibration requires a finite 3x4 or 4x4 CV extrinsic')
    return (np.linalg.inv(matrix) @ np.diag([1., -1., -1., 1.])).tolist()


def observation(task, *, instruction, identity=None, camera_names=None):
    """Capture current RGB/calibration and actual joints/finger positions.

    The official joint_action field contains drive targets and is deliberately not used
    for measured state. No objects, checker details or task-internal geometry are sent.
    """
    raw = task.get_obs()
    robot = task.robot
    vision, views = {}, {}
    for target, source in (camera_names or CAMERAS).items():
        camera = raw['observation'][source]
        color = np.asarray(camera['rgb'])[..., :3].astype(np.uint8)
        intrinsic = np.asarray(camera['intrinsic_cv'], float).tolist()
        converted = camera_to_world_gl(camera['extrinsic_cv'])
        if 'cam2world_gl' in camera:
            native = np.asarray(camera['cam2world_gl'], float)
            if native.shape != (4, 4) or not np.isfinite(native).all() or not np.allclose(
                    native, converted, atol=1e-5, rtol=1e-5):
                raise ValueError('RoboTwin CV and native GL camera calibration disagree')
            # Retain the simulator's original precision after verifying the conversion;
            # the direct host reads this exact model matrix as well.
            extrinsic = native.tolist()
        else:
            extrinsic = converted
        vision[target] = dict(color=color, shape=list(color.shape),
                              intrinsic_matrix=intrinsic, extrinsic_matrix=extrinsic)
        views[target] = dict(intrinsic=intrinsic, pose=extrinsic, shape=list(color.shape[:2]))
    state = dict(camera_views=views, robot_identity=identity or robot_identity(robot))
    targets = []
    for side in ('left', 'right'):
        state[f'{side}_arm_joint_state'] = list(map(float, getattr(
            robot, f'get_{side}_arm_real_jointState')()[:-1]))
        state[f'{side}_ee_joint_state'] = [measured_opening(robot, side)]
        state[f'{side}_tcp_pose'] = list(map(float, getattr(robot, f'get_{side}_tcp_pose')()))
        state[f'{side}_eef_pose_native'] = list(map(float, getattr(robot, f'get_{side}_ee_pose')()))
        state[f'{side}_gripper_command'] = float(getattr(robot, f'get_{side}_gripper_val')())
        targets.extend(map(float, getattr(robot, f'get_{side}_arm_jointState')()))
    state['qpos_target14'] = targets
    return dict(state=state, vision=vision, instruction=instruction,
                additional_info={'worldterminal': dict(
                    step=int(task.take_action_cnt), step_limit=int(task.step_lim), pictures_behind=0)})


def action_vector(row):
    """One SDK joint row to RoboTwin's native qpos14 ordering."""
    values = []
    for side in ('left', 'right'):
        joints = np.asarray(row[f'{side}_arm_joint_state'], float)
        opening = np.asarray(row[f'{side}_ee_joint_state'], float)
        if joints.shape != (6,) or opening.shape != (1,) or not np.isfinite(
                np.r_[joints, opening]).all() or not 0 <= opening[0] <= 1:
            raise ValueError('RoboTwin action requires six finite joints and opening in [0,1]')
        values.extend([*joints, opening[0]])
    return np.asarray(values, dtype=np.float64)


def run_episode(task, client, *, instruction):
    """Drive an already-reset official task. Waiting consumes no native action."""
    identity = robot_identity(task.robot)
    client.reset()
    try:
        while not task.eval_success and task.take_action_cnt < task.step_lim:
            batch = client.step(observation(task, instruction=instruction, identity=identity))
            while batch.waiting:
                batch = client.poll()
            if batch.done:
                break
            for row in batch.actions:
                if task.eval_success or task.take_action_cnt >= task.step_lim:
                    break
                task.take_action(action_vector(row), action_type='qpos')
    finally:
        result = dict(success=bool(task.eval_success), step=int(task.take_action_cnt),
                      step_limit=int(task.step_lim))
        client.end(result)
    return result
