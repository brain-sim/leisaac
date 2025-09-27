"""Simple utility to spawn a LeIsaac environment and step it with zero actions."""

import multiprocessing
import torch
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Spawn a LeIsaac environment for a quick sanity run.")
parser.add_argument("--task", type=str, required=True, help="Registered Gym task name (e.g. LeIsaac-Fridge-Stocking-v0).")
parser.add_argument("--num_steps", type=int, default=240, help="Number of simulation steps to execute.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments to spawn.")
parser.add_argument("--seed", type=int, default=None, help="Optional environment seed.")

# expose Isaac AppLauncher CLI flags (device/headless/etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import gymnasium as gym

from isaaclab_tasks.utils import parse_env_cfg

import leisaac  # noqa: F401  # ensure environments are registered


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    # keep things light for smoke tests
    env_cfg.recorders = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # reset once before stepping
    obs, _ = env.reset()

    sample_action = env.action_space.sample()
    zero_action = np.zeros_like(sample_action, dtype=env.action_space.dtype if hasattr(env.action_space, "dtype") else np.float32)

    step_count = 0
    while simulation_app.is_running() and step_count < args_cli.num_steps:
        action = zero_action
        if not isinstance(action, torch.Tensor):
            action = torch.from_numpy(np.asarray(action)).to(env.device)
            action = action.view(env.num_envs, -1)
            action[:, :3] = 0.0  # zero out base actions if present
        print(action)
        obs, reward, terminated, truncated, info = env.step(action)

        if hasattr(terminated, "cpu"):
            terminated_np = terminated.detach().cpu().numpy()
        else:
            terminated_np = np.asarray(terminated)

        if hasattr(truncated, "cpu"):
            truncated_np = truncated.detach().cpu().numpy()
        else:
            truncated_np = np.asarray(truncated)

        done = terminated_np | truncated_np
        if np.any(done):
            done_ids = np.nonzero(done)[0]
            if hasattr(env, "reset_done"):
                env.reset_done(env_ids=done_ids)
            else:
                env.reset()
        step_count += 1

    print(f"Completed {step_count} simulation steps for task '{args_cli.task}'.")

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(traceback.print_exc()) 
    finally:
        simulation_app.close()
