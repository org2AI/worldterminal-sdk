"""Only the created table's public setup geometry crosses the observation boundary."""

from types import SimpleNamespace

import pytest

from test_robotwin_evaluator import loaded_task
from worldterminal_sdk.robotwin import scene_setup
from worldterminal_sdk.robotwin_evaluator import measured_observation


def pose(z):
    return SimpleNamespace(p=[.12, -.08, z], q=[1., 0., 0., 0.])


def table():
    box = SimpleNamespace(get_local_pose=lambda: pose(-.013),
                          get_half_size=lambda: [.42, .31, .013])
    body = SimpleNamespace(get_collision_shapes=lambda: [box])
    return SimpleNamespace(get_pose=lambda: pose(.713), get_components=lambda: [body])


def test_adapter_reads_post_randomization_entity_and_box_without_nominal_height(tmp_path):
    task, raw = loaded_task(tmp_path)
    task.table, task.task_name, task.ep_num = table(), 'setup_test', 4
    task.table_z_bias = 999  # Neither this nor a nominal .74 is a surface measurement.
    value = measured_observation(raw, task_env=task)['additional_info']['worldterminal']
    setup = value['scene_setup']
    assert setup['source'] == 'env_setup'
    assert setup['provider'] == 'robotwin.task.table'
    assert setup['tick'] == value['step'] == task.take_action_cnt
    assert setup['table']['pose_wxyz'] == [.12, -.08, .713, 1., 0., 0., 0.]
    assert setup['table']['boxes'][0]['half_size_m'] == [.42, .31, .013]
    assert setup['episode'] == dict(task_name='setup_test', episode_index=4,
                                   table_instance=str(id(task.table)))
    assert 'actual_seed' not in setup['episode'] and 'seed' not in setup['episode']


def test_missing_table_supplies_no_setup_fallback():
    assert scene_setup(SimpleNamespace()) is None


def test_nonbox_table_is_refused():
    entity = table()
    entity.get_components = lambda: [SimpleNamespace(
        get_collision_shapes=lambda: [SimpleNamespace()])]
    task = SimpleNamespace(table=entity, task_name='setup_test', ep_num=0, take_action_cnt=0)
    with pytest.raises(ValueError, match='box collision geometry'):
        scene_setup(task)
