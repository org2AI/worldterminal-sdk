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
It stops when WT returns `done` or the official `eval_success`/`step_lim` ends the episode,
reports that checker outcome and `take_action_cnt`, and polls `waiting` replies without
consuming an environment action.

`observation()` reads actual arm qpos and the base finger joint's measured qpos, normalized
by the loaded embodiment's gripper scale. Drive targets and commanded grippers remain
separate fields. Camera `extrinsic_cv` is converted to GL camera-to-world by inversion and
axis conversion; it is not renamed. Loaded URDF hashes, joint names, scales and root poses
remain in `state.robot_identity`. Robot/camera state, RGB and the public table setup are
read; task objects and checker details are not observation inputs.

`additional_info.worldterminal.scene_setup` carries the actual created table entity pose
and its collision-box local poses and half sizes, after the episode's randomization. It is
labelled `env_setup`, with native tick, task name, episode index and an opaque table-instance
identity. No actual seed is inferred. The server derives its tabletop surface from these
boxes and keeps setup authority distinct from a visual measurement. Missing setup metadata
supplies no nominal-height fallback. The direct debug host's optional RPC extension requires
this updated SDK helper in its simulator environment as well.

The source geometry was checked against RoboTwin `ea8b211`:
`envs/_base_task.py` SHA256 `76a12a3ce2254c019e5c291e29c99b421dabbb0c6e0b9ec24db9155b2dcf81f5`
applies height randomization before creating `task.table`;
`envs/utils/create_actor.py` SHA256 `6bababee8e70da2460b2bbf47d3b5fbb20ccf73368782a7be596a63c962dce6d`
places the tabletop's upper face at the entity origin. The adapter reads the live geometry,
rather than reconstructing either nominal dimensions or the randomization formula.

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

The XPolicyLab `Model` adapter's RoboTwin mode must keep returning joint rows while that
official loop is running. When WT returns `done`, it latches that policy stop and returns
HOLD rows copied from the latest actual measured arm joints and normalized finger openings.
Each new observation refreshes the local HOLD; it is no longer sent to the ended WT worker,
and that worker is not polled again. Missing or invalid measured state refuses the action
rather than using a guessed pose, drive target or stale observation. Reset clears this state.
RoboDojo's existing client calls and `done` behavior are unchanged.

These HOLD rows are real native actions: physics continues, and the official checker or
native action budget ends the episode. An early WT failure is not turned into a success
claim or an empty-action transport error that skips to another seed. This changes the
client adapter's terminal behavior; it does not patch the evaluator. Keep the private WT
stop status separate from the official outcome. Physics may still evolve during holding;
this loop is not equivalent to a direct runner that stops immediately at WT `done`.
Positive direct/API parity requires the official checker to end before terminal holding,
or a separate comparison using the same terminal-loop semantics. This change does not
enforce an exact seed or change the evaluator's expert/setup/error skip rules.

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
and frequency. It also carries the public post-randomization table setup described above.
No task objects, seed selection or scorer state are added to the observation.
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

`tests/test_robotwin_terminal_hold.py` also exercises the unchanged pinned official
`eval_remote_policy` loop with fake physics: zero-step WT completion consumes measured
HOLD actions to a counted budget failure without a seed retry, and an earlier checker
success ends before holding. These CPU tests establish no native physics result.
