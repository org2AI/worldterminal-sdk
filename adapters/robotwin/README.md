# WorldTerminal for RoboTwin 2.0

Use the RoboTwin release with its XPolicyLab submodule (reference revision ea8b21121ebb3cd201ff5b3fe361944ac94eda3f). The shared WorldTerminal model selects the RoboTwin action-list interface when started with bench_name=RoboTwin. It polls waiting responses internally. A ready batch becomes a joint-action list. If the policy stops early, the evaluator treats the empty chunk as a failed rollout. RoboTwin owns success detection and reports its verdict to the server.

Install RoboTwin, its simulator assets and XPolicyLab following the upstream guide. Use a WorldTerminal deployment supporting RoboTwin observations and the aloha-agilex joint action layout for demo_clean. From the SDK repository root:

```sh
bash scripts/stage_xpolicylab.sh "$ROBOTWIN_ROOT/XPolicyLab"
cd "$ROBOTWIN_ROOT/XPolicyLab/policy/WorldTerminal"
# In the policy environment
bash install.sh
# Set WORLDTERMINAL_API_URL and optionally WORLDTERMINAL_API_TOKEN here
bash setup_eval_policy_server.sh RoboTwin adjust_bottle worldterminal arx_x5 joint 0 policy_env 19000
```

In a second terminal, activate the simulator environment and run one episode:

```sh
cd "$ROBOTWIN_ROOT"
bash scripts/eval_policy.sh --task_name adjust_bottle --task_config demo_clean \
  --env_cfg_type arx_x5 --policy_name WorldTerminal --host localhost --port 19000 \
  --protocol ws --eval_batch false --test_num 1 --seed 0 \
  --additional_info ckpt_name=worldterminal,action_type=joint
```

The benchmark maps its cameras, instruction and state into XPolicyLab observations and converts action rows to simulator controls. For joint actions send the left and right arm joint targets and both gripper commands. The arx_x5 command-line value selects RoboTwin's XPolicyLab layout; the simulator embodiment is selected by task_config.

[Official installation and evaluation commands](https://github.com/RoboTwin-Platform/RoboTwin#5-evaluate-policies-via-xpolicylab)
