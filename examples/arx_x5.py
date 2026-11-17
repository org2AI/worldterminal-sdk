"""Run one observation/action loop on an ARX X5 with a connected driver.

driver supplies joint_state(), camera_view(name), stopped(), apply_joint_targets(row) and outcome().
Configure WORLDTERMINAL_API_URL and WORLDTERMINAL_API_TOKEN for the hosted service.
The calling control loop owns sampling cadence, hardware stops and the measured outcome.
"""

from worldterminal_sdk import Client, Observation, arx_x5


def run(driver, instruction):
    profile = arx_x5()
    profile["video"] = {"head": "overhead_usb", "left_wrist": "left_usb", "right_wrist": "right_usb"}
    with Client(embodiment=profile) as client:
        client.reset()
        while not driver.stopped():
            observation = Observation(
                instruction=instruction,
                state=driver.joint_state(),
                vision={name: driver.camera_view(name) for name in profile["video"].values()},
            )
            batch = client.step(observation)
            while batch.waiting and not driver.stopped():
                batch = client.poll()
            if batch.done or driver.stopped():
                break
            for row in batch.actions:
                if driver.stopped():
                    break
                driver.apply_joint_targets(row)
        client.end(driver.outcome())
