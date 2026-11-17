# WorldTerminal for RoboTwin 2.0

The evaluator owns RoboTwin physics and its official verdict. The public SDK sends measured
observations to a RoboTwin-enabled hosted worker and applies returned qpos14 joint rows.

Use the official RoboTwin source at `ea8b21121ebb3cd201ff5b3fe361944ac94eda3f` and its assets.
Install this SDK in the simulator environment. After the official evaluator has reset its
`TASK_ENV` and selected the episode instruction:

```python
from worldterminal_sdk import Client, robotwin_aloha
from worldterminal_sdk.robotwin import run_episode

with Client(embodiment=robotwin_aloha()) as client:
    result = run_episode(TASK_ENV, client, instruction=episode_instruction)
```

Set `WORLDTERMINAL_API_URL` and, when required, `WORLDTERMINAL_API_TOKEN`. The server must
advertise `robotwin/aloha-agilex/joint-v1` and run the matching task/seed/configuration.
For a separately configured dual ARX-X5 station use `robotwin_arx_x5()` and the matching
`robotwin/ARX-X5/joint-v1` endpoint. Its actual loaded roots/mount separation remain in
`state.robot_identity`; this profile never supplies Aloha mounts or a guessed separation.
`run_episode` does not reset the simulator or choose a seed, task, language split or budget.
It stops at the official `eval_success` or `step_lim`, reports that checker outcome and
`take_action_cnt`, and polls `waiting` replies without consuming an environment action.

`observation()` reads actual arm qpos and the base finger joint's measured qpos, normalized
by the loaded embodiment's gripper scale. Drive targets and commanded grippers remain
separate fields. Camera `extrinsic_cv` is converted to GL camera-to-world by inversion and
axis conversion; it is not renamed. Loaded URDF hashes, joint names, scales and root poses
remain in `state.robot_identity`. Only robot/camera state and RGB are read; task objects or
checker details are not observation inputs.

The old unmodified XPolicyLab RoboTwin observation conversion sends drive targets and
commanded grippers. Staging the shared XPolicyLab policy directory alone does not satisfy
the measured-state contract. Use the evaluator-side adapter above; the server rejects
missing robot identity rather than treating an action-vector layout as model identity.

Source checks and transport tests establish no official RoboTwin score or direct/API parity.
