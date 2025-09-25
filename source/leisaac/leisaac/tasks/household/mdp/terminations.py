from __future__ import annotations

import torch

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv

from .observations import fridge_door_closed, object_placed


def cube_height_above_base(env: ManagerBasedRLEnv, cube_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg, robot_base_name: str = "base", height_threshold: float = 0.20) -> torch.Tensor:
    """Determine if the cube is above the robot base.

    This function checks whether all success conditions for the task have been met:
    1. cube is above the robot base

    Args:
        env: The RL environment instance.
        cube_cfg: Configuration for the cube entity.
        robot_cfg: Configuration for the robot entity.
        robot_base_name: Name of the robot base.
        height_threshold: Threshold for the cube height above the robot base.
    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    cube: RigidObject = env.scene[cube_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    cube_height = cube.data.root_pos_w[:, 2]
    base_index = robot.data.body_names.index(robot_base_name)
    robot_base_height = robot.data.body_pos_w[:, base_index, 2]
    above_base = cube_height - robot_base_height > height_threshold
    done = torch.logical_and(done, above_base)

    return done


def fridge_stocking_completed(
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg,
        target_cfg: SceneEntityCfg | None = None,
        fridge_cfg: SceneEntityCfg = SceneEntityCfg("fridge"),
        robot_cfg: SceneEntityCfg | None = None,
        ee_frame_cfg: SceneEntityCfg | None = None,
        target_position: tuple[float, float, float] | None = None,
        close_threshold: float = 0.1,
        xy_threshold: float = 0.15,
        z_threshold: float = 0.05) -> torch.Tensor:
    """Evaluate success for the fridge stocking task.

    The task succeeds when the object is placed near the target location and the fridge door
    is closed again.
    """

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
    closed = fridge_door_closed(
        env,
        fridge_cfg=fridge_cfg,
        angle_threshold=close_threshold,
    )

    return torch.logical_and(placed, closed)
