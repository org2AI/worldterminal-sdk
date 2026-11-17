# WorldTerminal SDK

Python SDK for the WorldTerminal hosted policy API, with adapters for RoboDojo, LIBERO, and RoboTwin. An adapter sends an instruction, robot state and camera observations, then applies the returned actions in the benchmark environment.

## Install

Use Python 3.10 or later. From the repository root:

```sh
python -m pip install .
export WORLDTERMINAL_API_URL='wss://your-policy-host.example'
# Set WORLDTERMINAL_API_TOKEN in the policy process environment if required.
```

Obtain the endpoint, credentials and supported robot/task configuration from the service operator. Installing this repository installs the client. Simulator dependencies belong in the corresponding benchmark environment.

## Run a benchmark

| Benchmark | Entry point | Validation scope |
| --- | --- | --- |
| [RoboDojo](adapters/xpolicylab/WorldTerminal/README.md) | XPolicyLab policy server and simulator client | **90.5 / 86.7%** |
| [RoboTwin 2.0](adapters/robotwin/README.md) | RoboTwin evaluation client and XPolicyLab policy server | Client-side integration for the XPolicyLab-based release. |
| [LIBERO](adapters/libero/README.md) | Single-episode Python runner | Client-side integration for the LIBERO observation and action contract. |

The service must support the chosen benchmark's cameras, state and action semantics.

## Control loop

Use one client per environment. Start each episode with `client.reset()`. Send each fresh observation once with `client.step(observation)`. Poll on `waiting`, apply each row on `ready`, and stop on `done`. Report the outcome measured by the environment with `client.end(result)`. In API v1, the raw response is a mapping with `status` and `actions`. Only a ready response contains action rows.

```python
with Client() as client:
    client.reset()
    while not robot.stopped():
        batch = client.step(robot.observe())
        while batch.waiting and not robot.stopped():
            batch = client.poll()
        if batch.done or robot.stopped():
            break
        for row in batch.actions:
            robot.execute(row)
            if robot.stopped():
                break
    client.end(robot.outcome())
```

Import `Client` from `worldterminal_sdk`. See [a bounded control loop](examples/control_loop.py) or [ARX X5 with named cameras and joint validation](examples/arx_x5.py). Using `Client(embodiment=arx_x5())` checks that the endpoint advertises the profile and validates observation and action dimensions.

## Development

```sh
python -m pip install -e . pytest
python -m pytest
```

The SDK tests use a local WebSocket endpoint. XPolicyLab model integration tests also use the upstream ModelTemplate when XPolicyLab is installed.
