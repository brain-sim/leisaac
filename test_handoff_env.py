#!/usr/bin/env python3
"""Sanity test for the LiftCube bi-arm handoff environment."""

from isaaclab.app import AppLauncher

# Launch simulator
app_launcher = AppLauncher(headless=False, enable_cameras=True)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import leisaac  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from leisaac.utils.env_utils import get_task_type


def _to_device_tensor(action, device: str) -> torch.Tensor:
    tensor = torch.as_tensor(action, device=device, dtype=torch.float32)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _to_bool_tensor(flag) -> torch.Tensor:
    if isinstance(flag, torch.Tensor):
        return flag.bool()
    if isinstance(flag, np.ndarray):
        return torch.from_numpy(flag).bool()
    return torch.tensor(flag, dtype=torch.bool)


def test_environment_loading(num_steps: int = 1_000) -> bool:
    try:
        print("Testing environment loading...")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        env_name = "LeIsaac-SO101-LiftCube-BiArm-Handoff-v0"
        env_cfg = parse_env_cfg(env_name, device=device, num_envs=1)
        env_cfg.use_teleop_device(get_task_type(env_name))

        env = gym.make(env_name, cfg=env_cfg)
        print("✓ Environment created successfully!")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")

        obs, info = env.reset()
        print("✓ Environment reset successfully!")
        if isinstance(obs, dict):
            print(f"Observation keys: {list(obs.keys())}")

        cumulative_reward = 0.0
        print(f"Running {num_steps} random steps...")

        for step_idx in range(num_steps):
            action = _to_device_tensor(env.action_space.sample(), device)
            obs, reward, terminated, truncated, info = env.step(action)

            if isinstance(reward, torch.Tensor):
                cumulative_reward += reward.mean().item()
            elif isinstance(reward, np.ndarray):
                cumulative_reward += float(np.mean(reward))
            else:
                cumulative_reward += float(reward)

            done_tensor = _to_bool_tensor(terminated) | _to_bool_tensor(truncated)
            if done_tensor.any().item():
                env.reset()

            if (step_idx + 1) % 100 == 0:
                print(f"  Completed {step_idx + 1} / {num_steps} steps")

        avg_reward = cumulative_reward / num_steps
        print(f"✓ Random rollout finished. Average reward: {avg_reward:.4f}")

        env.close()
        print("✓ Environment closed successfully!")
        return True

    except Exception as exc:  # noqa: BLE001
        print(f"✗ Error loading environment: {exc}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = test_environment_loading()
        if success:
            print("\n🎉 Environment test passed!")
        else:
            print("\n❌ Environment test failed!")
    finally:
        simulation_app.close()
