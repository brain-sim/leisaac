"""Observation functions for lift cube environments."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    distance_threshold: float = 0.04,
) -> torch.Tensor:
    """Check if object is grasped by the robot end-effector."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    
    # Get positions
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    obj_pos = obj.data.root_pos_w
    
    # Calculate distance
    distance = torch.norm(ee_pos - obj_pos, dim=-1)
    
    # Return binary indicator
    return (distance < distance_threshold).float()


def object_not_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    distance_threshold: float = 0.06,
) -> torch.Tensor:
    """Check if object is NOT grasped by the robot end-effector."""
    grasped = object_grasped(env, robot_cfg, ee_frame_cfg, object_cfg, distance_threshold)
    return 1.0 - grasped


def ee_frames_distance(
    env: ManagerBasedRLEnv,
    left_ee_cfg: SceneEntityCfg,
    right_ee_cfg: SceneEntityCfg,
    distance_threshold: float = 0.10,
) -> torch.Tensor:
    """Check if two end-effector frames are close enough for handoff."""
    left_ee: FrameTransformer = env.scene[left_ee_cfg.name]
    right_ee: FrameTransformer = env.scene[right_ee_cfg.name]
    
    left_pos = left_ee.data.target_pos_w[..., 0, :]
    right_pos = right_ee.data.target_pos_w[..., 0, :]
    
    distance = torch.norm(left_pos - right_pos, dim=-1)
    
    return (distance < distance_threshold).float()


def cube_on_table(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg,
    table_height: float = 0.79,
    height_tolerance: float = 0.05,
) -> torch.Tensor:
    """Check if cube is placed back on the table."""
    cube: RigidObject = env.scene[cube_cfg.name]
    cube_pos = cube.data.root_pos_w
    
    # Check if cube is at table height
    height_diff = torch.abs(cube_pos[..., 2] - table_height)
    
    return (height_diff < height_tolerance).float()
