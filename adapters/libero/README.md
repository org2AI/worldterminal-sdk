# WorldTerminal for LIBERO

Install LIBERO and its MuJoCo dependencies in an environment that can run the upstream OffScreenRenderEnv example. Install this SDK in the same Python environment. Configure a WorldTerminal endpoint that supports the LIBERO contract below. From the SDK repository root:

```sh
python -m pip install .
# Set WORLDTERMINAL_API_URL and optionally WORLDTERMINAL_API_TOKEN
python adapters/libero/evaluate.py --suite libero_spatial --task 0 --initial-state 0 --seed 0
```

The runner loads the selected BDDL task and its recorded initial state. It sends the task's language instruction, native robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos and robot0_joint_pos, and two 128×128 RGB views (agentview and robot0_eye_in_hand) in the orientation returned by LIBERO. Each returned action row uses the key action with seven normalized control values: three translation deltas, three rotation controls and the gripper command in LIBERO order. LIBERO evaluates success. The default budget is 300 environment steps and 600 seconds, checked between API operations.

[Upstream LIBERO installation and task example](https://github.com/Lifelong-Robot-Learning/LIBERO#readme)
