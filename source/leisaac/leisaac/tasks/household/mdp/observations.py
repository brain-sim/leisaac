from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv


def object_grasped(
        env: ManagerBasedRLEnv,
        robot_cfg: SceneEntityCfg | None = None,
        ee_frame_cfg: SceneEntityCfg | None = None,
        object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
        diff_threshold: float = 0.02,
        grasp_threshold: float = 0.26) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""

    default_result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    if robot_cfg is None or robot_cfg.name == "" or ee_frame_cfg is None or ee_frame_cfg.name == "":
        return default_result

    try:
        robot: Articulation = env.scene[robot_cfg.name]
        ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
        object: RigidObject = env.scene[object_cfg.name]
    except KeyError:
        return default_result

    if not hasattr(robot, "data") or not hasattr(ee_frame, "data"):
        return default_result

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 1, :]
    pos_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    grasped = torch.logical_and(pos_diff < diff_threshold, robot.data.joint_pos[:, -1] < grasp_threshold)

    return grasped


def _select_articulation_joints(articulation: Articulation, joint_keywords: tuple[str, ...]) -> torch.Tensor | None:
    """Return joint positions that contain any of the provided keywords.

    Args:
        articulation: Articulation instance to query.
        joint_keywords: Keywords to search for inside the joint names (case-insensitive).

    Returns:
        A tensor with shape (num_envs, num_matching_joints) containing the joint positions. If
        no joints match, ``None`` is returned.
    """

    if not hasattr(articulation, "data"):
        return None

    joint_names = getattr(articulation.data, "joint_names", [])
    if not joint_keywords or joint_names is None:
        return None

    matching_indices = [idx for idx, name in enumerate(joint_names) if any(keyword in name.lower() for keyword in joint_keywords)]
    if not matching_indices:
        return None

    return articulation.data.joint_pos[:, matching_indices]


def fridge_door_opened(
        env: ManagerBasedRLEnv,
        fridge_cfg: SceneEntityCfg = SceneEntityCfg("fridge"),
        joint_keywords: tuple[str, ...] = ("door",),
        angle_threshold: float = 0.35) -> torch.Tensor:
    """Detect whether the fridge door articulation exceeds an opening threshold."""

    try:
        fridge: Articulation = env.scene[fridge_cfg.name]
    except KeyError:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    joint_positions = _select_articulation_joints(fridge, joint_keywords)
    if joint_positions is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    door_angle = torch.abs(joint_positions).amax(dim=1)
    return door_angle > angle_threshold


def fridge_door_closed(
        env: ManagerBasedRLEnv,
        fridge_cfg: SceneEntityCfg = SceneEntityCfg("fridge"),
        joint_keywords: tuple[str, ...] = ("door",),
        angle_threshold: float = 0.1) -> torch.Tensor:
    """Detect whether the fridge door articulation is below the closing threshold."""

    try:
        fridge: Articulation = env.scene[fridge_cfg.name]
    except KeyError:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    joint_positions = _select_articulation_joints(fridge, joint_keywords)
    if joint_positions is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    door_angle = torch.abs(joint_positions).amax(dim=1)
    return door_angle < angle_threshold


def object_near_target(
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
        target_cfg: SceneEntityCfg | None = None,
        target_position: tuple[float, float, float] | None = None,
        xy_threshold: float = 0.15,
        z_threshold: float = 0.05) -> torch.Tensor:
    """Check whether the object is positioned close to the given target."""

    try:
        obj: RigidObject = env.scene[object_cfg.name]
    except KeyError:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    object_pos = obj.data.root_pos_w

    target_pos = None
    if target_cfg is not None:
        try:
            target = env.scene[target_cfg.name]
            if hasattr(target, "data") and hasattr(target.data, "root_pos_w"):
                target_pos = target.data.root_pos_w
        except KeyError:
            target_pos = None

    if target_pos is None:
        if target_position is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        target_tensor = torch.tensor(target_position, dtype=object_pos.dtype, device=env.device)
        target_pos = target_tensor.repeat(object_pos.shape[0], 1)

    horizontal_delta = object_pos[:, :2] - target_pos[:, :2]
    horizontal_dist = torch.linalg.vector_norm(horizontal_delta, dim=1)
    vertical_delta = torch.abs(object_pos[:, 2] - target_pos[:, 2])

    close_horizontal = horizontal_dist < xy_threshold
    close_vertical = vertical_delta < z_threshold

    return torch.logical_and(close_horizontal, close_vertical)


def object_placed(
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
        target_cfg: SceneEntityCfg | None = None,
        target_position: tuple[float, float, float] | None = None,
        robot_cfg: SceneEntityCfg | None = None,
        ee_frame_cfg: SceneEntityCfg | None = None,
        xy_threshold: float = 0.15,
        z_threshold: float = 0.05) -> torch.Tensor:
    """Determine whether the object has been placed at the target and released."""

    near_target = object_near_target(
        env,
        object_cfg=object_cfg,
        target_cfg=target_cfg,
        target_position=target_position,
        xy_threshold=xy_threshold,
        z_threshold=z_threshold,
    )
    if not torch.any(near_target):
        return near_target

    if robot_cfg is not None and ee_frame_cfg is not None:
        grasped = object_grasped(
            env,
            robot_cfg=robot_cfg,
            ee_frame_cfg=ee_frame_cfg,
            object_cfg=object_cfg,
        )
    else:
        grasped = torch.zeros_like(near_target)

    return torch.logical_and(near_target, torch.logical_not(grasped))
