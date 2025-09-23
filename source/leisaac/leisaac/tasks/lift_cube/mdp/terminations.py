"""Termination functions for lift cube environments."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cube_height_above_base(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    robot_base_name: str = "base",
    height_threshold: float = 0.20,
) -> torch.Tensor:
    """Check if cube is lifted above a certain height from robot base."""
    robot: Articulation = env.scene[robot_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]
    
    # Get robot base position
    base_link_idx = robot.find_bodies(robot_base_name)[0]
    base_pos = robot.data.body_pos_w[:, base_link_idx]
    
    # Get cube position
    cube_pos = cube.data.root_pos_w
    
    # Calculate height difference
    height_diff = cube_pos[..., 2] - base_pos[..., 2]
    
    return height_diff > height_threshold


def handoff_sequence_complete(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg,
    left_robot_cfg: SceneEntityCfg,
    right_robot_cfg: SceneEntityCfg,
    left_ee_cfg: SceneEntityCfg,
    right_ee_cfg: SceneEntityCfg,
    table_height: float = 0.79,
    height_tolerance: float = 0.05,
) -> torch.Tensor:
    """Check if the complete handoff sequence is done: cube on table and both arms released."""
    cube: RigidObject = env.scene[cube_cfg.name]
    left_ee: FrameTransformer = env.scene[left_ee_cfg.name]
    right_ee: FrameTransformer = env.scene[right_ee_cfg.name]
    
    cube_pos = cube.data.root_pos_w
    left_ee_pos = left_ee.data.target_pos_w[..., 0, :]
    right_ee_pos = right_ee.data.target_pos_w[..., 0, :]
    
    # Check if cube is on table
    height_diff = torch.abs(cube_pos[..., 2] - table_height)
    cube_on_table = height_diff < height_tolerance
    
    # Check if both arms are not grasping (some distance away)
    left_distance = torch.norm(left_ee_pos - cube_pos, dim=-1)
    right_distance = torch.norm(right_ee_pos - cube_pos, dim=-1)
    
    both_released = (left_distance > 0.08) & (right_distance > 0.08)
    
    return cube_on_table & both_released


def cube_dropped_during_handoff(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg,
    left_robot_cfg: SceneEntityCfg,
    right_robot_cfg: SceneEntityCfg,
    left_ee_cfg: SceneEntityCfg,
    right_ee_cfg: SceneEntityCfg,
    min_height: float = 0.05,
    table_height: float = 0.79,
) -> torch.Tensor:
    """Check if cube was dropped during handoff (fell below minimum height and not on table)."""
    cube: RigidObject = env.scene[cube_cfg.name]
    left_ee: FrameTransformer = env.scene[left_ee_cfg.name]
    right_ee: FrameTransformer = env.scene[right_ee_cfg.name]
    
    cube_pos = cube.data.root_pos_w
    left_ee_pos = left_ee.data.target_pos_w[..., 0, :]
    right_ee_pos = right_ee.data.target_pos_w[..., 0, :]
    
    # Check if cube is below minimum height
    cube_too_low = cube_pos[..., 2] < min_height
    
    # Check if cube is not on table (not at table height)
    height_diff = torch.abs(cube_pos[..., 2] - table_height)
    not_on_table = height_diff > 0.1
    
    # Check if neither arm is grasping
    left_distance = torch.norm(left_ee_pos - cube_pos, dim=-1)
    right_distance = torch.norm(right_ee_pos - cube_pos, dim=-1)
    
    neither_grasping = (left_distance > 0.06) & (right_distance > 0.06)
    
    return cube_too_low & not_on_table & neither_grasping
