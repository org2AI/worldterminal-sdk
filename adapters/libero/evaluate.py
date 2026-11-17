"""Run a LIBERO episode against a WorldTerminal API v1 endpoint."""

import argparse
from contextlib import suppress
from pathlib import Path
from time import monotonic

import numpy as np
from worldterminal_sdk import Client, Observation


def observation(raw, instruction):
    state = {key: raw[key] for key in (
        "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_joint_pos"
    )}
    # Preserve LIBERO's native camera orientation on both streams.
    vision = {name: {"color": raw[name + "_image"]} for name in ("agentview", "robot0_eye_in_hand")}
    return Observation(state=state, vision=vision, instruction=instruction)


def run_episode(env, initial_state, instruction, client, max_steps, timeout):
    env.reset()
    raw = env.set_init_state(initial_state)
    deadline = monotonic() + timeout
    steps = 0
    client.reset()
    try:
        while steps < max_steps:
            result = client.step(observation(raw, instruction))
            while result.waiting:
                if monotonic() >= deadline:
                    raise TimeoutError("LIBERO policy wait exceeded the episode deadline")
                result = client.poll()
            if result.done:
                break
            for row in result.actions:
                if monotonic() >= deadline:
                    raise TimeoutError("LIBERO episode deadline exceeded")
                action = np.asarray(row["action"], dtype=np.float32)
                if action.shape != (7,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
                    raise ValueError("expected seven finite LIBERO control values within [-1, 1]")
                raw, _, terminal, _ = env.step(action)
                steps += 1
                if terminal or env.check_success() or steps == max_steps:
                    break
            if terminal or env.check_success():
                break
        outcome = {"success": bool(env.check_success()), "steps": steps}
    except Exception:
        with suppress(Exception):
            client.end({"aborted": True, "success": False, "steps": steps})
        raise
    client.end(outcome)
    return outcome


def main():
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--initial-state", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if args.max_steps < 1 or args.timeout <= 0 or min(args.task, args.initial_state) < 0:
        parser.error("use a positive step/time budget and non-negative indices")
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task)
    initial_state = suite.get_task_init_states(args.task)[args.initial_state]
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128)
    try:
        env.seed(args.seed)
        with Client() as client:
            print(run_episode(env, initial_state, task.language, client, args.max_steps, args.timeout))
    finally:
        env.close()


if __name__ == "__main__":
    main()
