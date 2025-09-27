from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv

from .observations import (
    fridge_door_closed,
    fridge_door_opened,
    object_grasped,
    object_placed,
    robot_close_to_object,
    robot_close_to_target,
    robot_to_object_distance,
)


@dataclass
class FridgeStockingSequentialReward:
    """Stateful reward that unlocks stage rewards sequentially for the fridge stocking task."""

    stage_rewards: Sequence[float]
    approach_distance_threshold: float = 0.75
    fridge_distance_threshold: float = 0.8
    distance_shaping_scale: float = 0.1

    def __post_init__(self) -> None:
        self._stage_state: torch.Tensor | None = None
        self._stage_completion: torch.Tensor | None = None

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if self._stage_state is None or self._stage_completion is None:
            return
        if env_ids is None:
            self._stage_state[:] = 0
            self._stage_completion[:] = False
        else:
            self._stage_state[env_ids] = 0
            self._stage_completion[env_ids] = False

    def _ensure_buffers(self, env: ManagerBasedRLEnv) -> None:
        num_envs = env.num_envs
        device = env.device
        num_stages = len(self.stage_rewards)
        if self._stage_state is None or self._stage_state.shape[0] != num_envs:
            self._stage_state = torch.zeros(num_envs, dtype=torch.long, device=device)
        if self._stage_completion is None or self._stage_completion.shape != (num_envs, num_stages):
            self._stage_completion = torch.zeros((num_envs, num_stages), dtype=torch.bool, device=device)

    def __call__(
            self,
            env: ManagerBasedRLEnv,
            *,
            robot_cfg: SceneEntityCfg,
            object_cfg: SceneEntityCfg,
            fridge_cfg: SceneEntityCfg,
            ee_frame_cfg: SceneEntityCfg | None = None,
            target_cfg: SceneEntityCfg | None = None,
            target_position: tuple[float, float, float] | None = None,
            xy_threshold: float = 0.18,
            z_threshold: float = 0.06,
    ) -> torch.Tensor:
        self._ensure_buffers(env)
        assert self._stage_state is not None
        assert self._stage_completion is not None

        reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        active_states = self._stage_state

        # Stage 0: approach the object
        mask0 = active_states == 0
        if torch.any(mask0):
            close_mask = robot_close_to_object(env, robot_cfg, object_cfg, self.approach_distance_threshold)
            newly_completed = mask0 & close_mask & (~self._stage_completion[:, 0])
            reward[newly_completed] += self.stage_rewards[0]
            self._stage_completion[newly_completed, 0] = True
            active_states[newly_completed] = 1

            # shaping on distance when still approaching
            pending = mask0 & (~close_mask)
            distances = robot_to_object_distance(env, robot_cfg, object_cfg)
            if distances is not None:
                reward[pending] += torch.clamp(-distances[pending] * self.distance_shaping_scale, min=-1.0)

        # Stage 1: grasp the object
        mask1 = active_states == 1
        if torch.any(mask1):
            grasped = object_grasped(env, robot_cfg=robot_cfg, ee_frame_cfg=ee_frame_cfg, object_cfg=object_cfg)
            newly_completed = mask1 & grasped & (~self._stage_completion[:, 1])
            reward[newly_completed] += self.stage_rewards[1]
            self._stage_completion[newly_completed, 1] = True
            active_states[newly_completed] = 2

        # Stage 2: move towards fridge while holding object
        mask2 = active_states == 2
        if torch.any(mask2):
            near_fridge = robot_close_to_target(
                env,
                robot_cfg,
                target_cfg=fridge_cfg,
                target_position=None,
                distance_threshold=self.fridge_distance_threshold,
            )
            newly_completed = mask2 & near_fridge & (~self._stage_completion[:, 2])
            reward[newly_completed] += self.stage_rewards[2]
            self._stage_completion[newly_completed, 2] = True
            active_states[newly_completed] = 3

            pending = mask2 & (~near_fridge)
            if torch.any(pending):
                distances = robot_to_object_distance(env, robot_cfg, fridge_cfg)
                if distances is not None:
                    reward[pending] += torch.clamp(-distances[pending] * self.distance_shaping_scale, min=-1.0)

        # Stage 3: open fridge door
        mask3 = active_states == 3
        if torch.any(mask3):
            door_open = fridge_door_opened(env, fridge_cfg=fridge_cfg)
            newly_completed = mask3 & door_open & (~self._stage_completion[:, 3])
            reward[newly_completed] += self.stage_rewards[3]
            self._stage_completion[newly_completed, 3] = True
            active_states[newly_completed] = 4

        # Stage 4: place object inside fridge
        mask4 = active_states == 4
        if torch.any(mask4):
            placed = object_placed(
                env,
                object_cfg=object_cfg,
                target_cfg=target_cfg,
                target_position=target_position,
                robot_cfg=robot_cfg,
                ee_frame_cfg=ee_frame_cfg,
                xy_threshold=xy_threshold,
                z_threshold=z_threshold,
            )
            newly_completed = mask4 & placed & (~self._stage_completion[:, 4])
            reward[newly_completed] += self.stage_rewards[4]
            self._stage_completion[newly_completed, 4] = True
            active_states[newly_completed] = 5

        # Stage 5: close fridge door
        mask5 = active_states == 5
        if torch.any(mask5):
            door_closed = fridge_door_closed(env, fridge_cfg=fridge_cfg)
            newly_completed = mask5 & door_closed & (~self._stage_completion[:, 5])
            reward[newly_completed] += self.stage_rewards[5]
            self._stage_completion[newly_completed, 5] = True
            active_states[newly_completed] = 6

        return reward
