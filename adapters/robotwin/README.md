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

## Explicit evaluator launcher

The explicit SDK launcher boundary is accepted; native simulator validation remains
pending. The standard unmodified RoboTwin launcher remains unsupported for the measured
state contract. This launcher is not enabled by `stage_xpolicylab.sh`.

The pinned official evaluator has no policy callback receiving `TASK_ENV`. The explicit SDK
launcher loads its unchanged `scripts/eval_policy_xpolicylab.py`, verifies its SHA256, and
overrides the module's named `robotwin_obs_to_xpolicylab` function **in memory**. This is a
runtime converter override, not unchanged standard-launcher compatibility. It then calls
the original `Sapien_TEST`, `parse_args`, and sequential `main` in the official order. The
official loop, action conversion, seeds, task setup, instructions and checker are retained.

Run from the RoboTwin root with the SDK source on the simulator Python path:

```bash
cd "$ROBOTWIN_ROOT"
PYTHONPATH="$SDK_ROOT/src:$ROBOTWIN_ROOT:$ROBOTWIN_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}" \
  python -m worldterminal_sdk.robotwin_evaluator --robotwin-root "$ROBOTWIN_ROOT" -- \
  --bench_name robotwin --task_name "$TASK" --policy_name WorldTerminal \
  --root_dir "$ROBOTWIN_ROOT" --port "$POLICY_PORT" --device_id "$ASSIGNED_DEVICE" \
  --task_config demo_randomized --seed "$OFFICIAL_SEED_INDEX" --test_num 1 \
  --eval_batch false
```

Arguments after `--` go to the official parser unchanged. Its `--seed` is an index: the
official starting seed is `100000 * (1 + index)`, and the evaluator can skip/retry seeds.
Record the actual selected seed; this launcher does not enforce an exact seed or change
that behavior. GPU isolation and policy endpoint setup remain the coordinator's existing
responsibility. Batch mode is refused because spawned workers would not inherit this
module-local override.

The converter consumes the evaluator's existing raw observation without a second
`get_obs()` call. It derives robot identity and measured joints/finger opening from that
episode's actual `TASK_ENV.robot`, and retains the official instruction, environment index
and frequency. No task objects, seed selection or scorer state are added to the observation.
The evaluator file must have SHA256
`78821388d58daf8aa65db55c5952df3742486dd93e717ce6a96374a3795dfc7a`.

The focused test can use the actual official file without running the simulator:

```bash
ROBOTWIN_OFFICIAL_EVALUATOR="$ROBOTWIN_ROOT/scripts/eval_policy_xpolicylab.py" \
  PYTHONPATH=src python -m pytest tests/test_robotwin_evaluator.py -q
```

It checks the real parser/boundary, measured metadata, byte-identical source before and
after, no duplicate observation capture and explicit batch refusal. Physics execution and
the official checker are not validated by this test.
