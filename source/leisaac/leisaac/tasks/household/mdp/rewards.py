from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv

if TYPE_CHECKING:
    from .subtasks import ResolvedSubtask

from termcolor import colored

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

    sequence: Sequence["ResolvedSubtask"]
    distance_shaping_scale: float = 0.1
    debug: bool = False

    def __post_init__(self) -> None:
        self._stage_state: torch.Tensor | None = None
        self._stage_completion: torch.Tensor | None = None
        self._stage_rewards = [entry.reward for entry in self.sequence]
        self._stage_names = [entry.name for entry in self.sequence] + ["completed"]
        self._prev_episode_length: torch.Tensor | None = None

    def _mask_to_indices(self, mask: torch.Tensor) -> list[int]:
        return mask.nonzero(as_tuple=True)[0].detach().cpu().tolist()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if self._stage_state is None or self._stage_completion is None:
            return
        if env_ids is None:
            self._stage_state[:] = 0
            self._stage_completion[:] = False
            if self._prev_episode_length is not None:
                self._prev_episode_length[:] = 0
        else:
            self._stage_state[env_ids] = 0
            self._stage_completion[env_ids] = False
            if self._prev_episode_length is not None:
                self._prev_episode_length[env_ids] = 0

    def _ensure_buffers(self, env: ManagerBasedRLEnv) -> None:
        num_envs = env.num_envs
        device = env.device
        num_stages = len(self._stage_rewards)
        if self._stage_state is None or self._stage_state.shape[0] != num_envs:
            self._stage_state = torch.zeros(num_envs, dtype=torch.long, device=device)
        if self._stage_completion is None or self._stage_completion.shape != (
            num_envs,
            num_stages,
        ):
            self._stage_completion = torch.zeros(
                (num_envs, num_stages), dtype=torch.bool, device=device
            )
        if (
            self._prev_episode_length is None
            or self._prev_episode_length.shape[0] != num_envs
        ):
            self._prev_episode_length = torch.zeros(
                num_envs, dtype=torch.long, device=device
            )

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        self._ensure_buffers(env)
        assert self._stage_state is not None
        assert self._stage_completion is not None

        reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        active_states = self._stage_state
        if hasattr(env, "episode_length_buf"):
            episode_lengths = env.episode_length_buf.to(
                device=active_states.device, dtype=torch.long
            )
            prev_lengths = self._prev_episode_length
            reset_mask = episode_lengths == 0
            if (
                prev_lengths is not None
                and prev_lengths.shape[0] == episode_lengths.shape[0]
            ):
                reset_mask = torch.logical_or(
                    reset_mask, episode_lengths < prev_lengths
                )
            if torch.any(reset_mask):
                active_states[reset_mask] = 0
                self._stage_completion[reset_mask] = False
            if prev_lengths is not None:
                prev_lengths.copy_(episode_lengths)

        # Stage 0: approach the object
        mask0 = active_states == 0
        if torch.any(mask0):
            # print("Stage 0 active on envs")
            params0 = self.sequence[0].params
            distance_threshold0 = float(params0.get("distance_threshold", 0.9))
            close_mask = robot_close_to_object(
                env,
                robot_cfg=params0["robot_cfg"],
                object_cfg=params0["object_cfg"],
                distance_threshold=distance_threshold0,
            )

            newly_completed = mask0 & close_mask & (~self._stage_completion[:, 0])
            reward[newly_completed] += self._stage_rewards[0]
            self._stage_completion[newly_completed, 0] = True
            active_states[newly_completed] = 1
            param3 = self.sequence[3].params
            door_open = fridge_door_opened(
                env,
                fridge_prim_path=param3.get("fridge_prim_path", ""),
                door_joint_names=param3.get("door_joint_names", []),
            )
            if door_open.any():
                print(colored("Door open ", "green", attrs=["bold"]))
            pending = mask0 & (~close_mask)
            distances = robot_to_object_distance(
                env, params0["robot_cfg"], params0["object_cfg"]
            )
            if distances is not None:
                reward[pending] += torch.clamp(
                    -distances[pending] * self.distance_shaping_scale, min=-1.0
                )

        # Stage 1: grasp the object
        mask1 = active_states == 1
        if torch.any(mask1):
            params1 = self.sequence[1].params

            grasped = object_grasped(
                env,
                robot_cfg=params1["robot_cfg"],
                left_ee_frame_cfg=params1["left_ee_frame_cfg"],
                right_ee_frame_cfg=params1["right_ee_frame_cfg"],
                object_cfg=params1["object_cfg"],
            )
            newly_completed = mask1 & grasped & (~self._stage_completion[:, 1])
            reward[newly_completed] += self._stage_rewards[1]
            self._stage_completion[newly_completed, 1] = True
            active_states[newly_completed] = 2

        # Stage 2: move towards fridge while holding object
        mask2 = active_states == 2
        if torch.any(mask2):
            print("Stage 2 active on envs")
            params2 = self.sequence[2].params
            distance_threshold2 = float(params2.get("distance_threshold", 1.5))
            near_fridge = robot_close_to_target(
                env,
                robot_cfg=params2["robot_cfg"],
                target_cfg=params2.get("target_cfg"),
                target_position=params2.get("target_position"),
                distance_threshold=distance_threshold2,
                include_z=False,
            )
            still_holding = object_grasped(
                env,
                robot_cfg=params2["robot_cfg"],
                left_ee_frame_cfg=params2["left_ee_frame_cfg"],
                right_ee_frame_cfg=params2["right_ee_frame_cfg"],
                object_cfg=params2["object_cfg"],
            )

            newly_completed = (
                mask2 & near_fridge & still_holding & (~self._stage_completion[:, 2])
            )
            reward[newly_completed] += self._stage_rewards[2]
            self._stage_completion[newly_completed, 2] = True
            active_states[newly_completed] = 3

            pending = mask2 & (~near_fridge) & still_holding
            if torch.any(pending):
                distances = None
                target_cfg = params2.get("target_cfg")
                if target_cfg is not None:
                    distances = robot_to_object_distance(
                        env, params2["robot_cfg"], target_cfg
                    )
                if distances is not None:
                    reward[pending] += torch.clamp(
                        -distances[pending] * self.distance_shaping_scale, min=-1.0
                    )

            dropped_mask = mask2 & (~still_holding)
            if torch.any(dropped_mask):
                self._stage_completion[dropped_mask, 2] = False

        # Stage 3: open fridge door
        mask3 = active_states == 3
        if torch.any(mask3):
            params3 = self.sequence[3].params
            door_open = fridge_door_opened(
                env,
                fridge_prim_path=params3["fridge_prim_path"],
                door_joint_names=params3.get("door_joint_names", []),
            )
            newly_completed = mask3 & door_open & (~self._stage_completion[:, 3])
            reward[newly_completed] += self._stage_rewards[3]
            self._stage_completion[newly_completed, 3] = True
            active_states[newly_completed] = 4

        # Stage 4: place object inside fridge
        mask4 = active_states == 4
        if torch.any(mask4):
            params4 = self.sequence[4].params
            placed = object_placed(env, **params4, debug=True)
            newly_completed = mask4 & placed & (~self._stage_completion[:, 4])
            reward[newly_completed] += self._stage_rewards[4]
            self._stage_completion[newly_completed, 4] = True
            active_states[newly_completed] = 5

        # Stage 5: close fridge door
        mask5 = active_states == 5
        if torch.any(mask5):
            params5 = self.sequence[5].params
            door_closed = fridge_door_closed(
                env,
                fridge_prim_path=params5["fridge_prim_path"],
                door_joint_names=params5.get("door_joint_names", []),
            )
            newly_completed = mask5 & door_closed & (~self._stage_completion[:, 5])
            reward[newly_completed] += self._stage_rewards[5]
            self._stage_completion[newly_completed, 5] = True
            active_states[newly_completed] = 6

        return reward


@dataclass
class SequentialSubtaskReward:
    """Generic sequential reward that dispenses bonuses for ordered subtask completion."""

    stage_funcs: Sequence[Callable[..., torch.Tensor]]
    stage_rewards: Sequence[float]

    def __post_init__(self) -> None:
        if len(self.stage_funcs) != len(self.stage_rewards):
            raise ValueError("stage_funcs and stage_rewards must have the same length")
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
        num_stages = len(self.stage_funcs)
        if self._stage_state is None or self._stage_state.shape[0] != num_envs:
            self._stage_state = torch.zeros(num_envs, dtype=torch.long, device=device)
        if self._stage_completion is None or self._stage_completion.shape != (
            num_envs,
            num_stages,
        ):
            self._stage_completion = torch.zeros(
                (num_envs, num_stages), dtype=torch.bool, device=device
            )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        *,
        stage_params: Sequence[dict[str, Any]],
    ) -> torch.Tensor:
        if len(stage_params) != len(self.stage_funcs):
            raise ValueError(
                "stage_params must provide parameters for each stage function"
            )

        self._ensure_buffers(env)
        assert self._stage_state is not None
        assert self._stage_completion is not None

        reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        active_states = self._stage_state

        for stage_idx, (stage_fn, stage_reward) in enumerate(
            zip(self.stage_funcs, self.stage_rewards)
        ):
            mask = active_states == stage_idx
            if not torch.any(mask):
                continue

            outcome = stage_fn(env, **stage_params[stage_idx])
            if outcome.dtype != torch.bool:
                outcome = outcome > 0.0

            newly_completed = mask & outcome & (~self._stage_completion[:, stage_idx])
            reward[newly_completed] += stage_reward
            self._stage_completion[newly_completed, stage_idx] = True
            active_states[newly_completed] = stage_idx + 1

        return reward
