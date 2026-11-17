# WorldTerminal for RoboDojo

This evaluation adapter connects RoboDojo to a WorldTerminal v1 endpoint. It supports one environment per session and joint control. The split-process setup uses the benchmark environment for simulation and the policy environment for the adapter and SDK.

## Install

From the SDK repository root, stage this folder and build the matching SDK wheel. Supply a RoboDojo checkout with its XPolicyLab submodule and simulation dependencies installed.

```sh
bash scripts/stage_xpolicylab.sh "$ROBODOJO_ROOT/XPolicyLab"
cd "$ROBODOJO_ROOT/XPolicyLab/policy/WorldTerminal"
# Activate the policy environment first
bash install.sh
```

## Evaluate

Set WORLDTERMINAL_API_URL and, for authenticated deployments, WORLDTERMINAL_API_TOKEN on the policy host. Replace the last two arguments with your policy and simulator Conda environment names.

```sh
bash eval.sh RoboDojo stack_blocks worldterminal arx_x5 joint 0 0 0 policy_env sim_env
```

The example uses seed 0 and GPU 0. The evaluator controls its task layouts. The adapter forwards observations and polls on waiting before applying joint targets. Episode completion and scoring come from the environment. To enable SDK checks for a matching endpoint, set embodiment to arx_x5/joint-v1 in deploy.yml.

[Upstream RoboDojo setup](https://robodojo-benchmark.com/doc/usage/)
