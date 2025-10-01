from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg | None = None,
    left_ee_frame_cfg: SceneEntityCfg | None = None,
    right_ee_frame_cfg: SceneEntityCfg | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    diff_threshold: float = 0.15,
    grasp_threshold: float = 0.4,
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""

    default_result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    if (
        robot_cfg is None
        or robot_cfg.name == ""
        or left_ee_frame_cfg is None
        or left_ee_frame_cfg.name == ""
        or right_ee_frame_cfg is None
        or right_ee_frame_cfg.name == ""
    ):
        return default_result

    try:
        robot: Articulation = env.scene[robot_cfg.name]
        left_ee_frame: FrameTransformer = env.scene[left_ee_frame_cfg.name]
        right_ee_frame: FrameTransformer = env.scene[right_ee_frame_cfg.name]
        obj = env.scene[object_cfg.name]
    except KeyError:
        return default_result

    if (
        not hasattr(robot, "data")
        or not hasattr(left_ee_frame, "data")
        or not hasattr(right_ee_frame, "data")
    ):
        return default_result

    if not hasattr(obj, "data") or not hasattr(obj.data, "root_pos_w"):
        return default_result

    object_pos = obj.data.root_pos_w
    left_target_pos = getattr(left_ee_frame.data, "target_pos_w", None)
    right_target_pos = getattr(right_ee_frame.data, "target_pos_w", None)
    if (left_target_pos is None or left_target_pos.shape[1] == 0) and (
        right_target_pos is None or right_target_pos.shape[1] == 0
    ):
        return default_result

    left_end_effector_pos = left_target_pos[:, -1, :]
    right_end_effector_pos = right_target_pos[:, -1, :]
    left_pos_diff = torch.linalg.vector_norm(object_pos - left_end_effector_pos, dim=1)
    right_pos_diff = torch.linalg.vector_norm(
        object_pos - right_end_effector_pos, dim=1
    )
    left_grasped = torch.logical_and(
        left_pos_diff < diff_threshold, robot.data.joint_pos[:, -2] < grasp_threshold
    )
    right_grasped = torch.logical_and(
        right_pos_diff < diff_threshold, robot.data.joint_pos[:, -1] < grasp_threshold
    )
    grasped = torch.logical_or(left_grasped, right_grasped)
    return grasped


def _select_articulation_joints(
    articulation: Articulation, joint_keywords: tuple[str, ...]
) -> torch.Tensor | None:
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

    matching_indices = [
        idx
        for idx, name in enumerate(joint_names)
        if any(keyword in name.lower() for keyword in joint_keywords)
    ]
    if not matching_indices:
        return None

    return articulation.data.joint_pos[:, matching_indices]


def fridge_door_opened(
    env: ManagerBasedRLEnv,
    fridge_cfg: SceneEntityCfg = SceneEntityCfg("fridge"),
    joint_keywords: tuple[str, ...] = ("door",),
    angle_threshold: float = 0.35,
    debug: bool = False,
) -> torch.Tensor:
    """Detect whether the fridge door articulation exceeds an opening threshold."""

    try:
        fridge: Articulation = env.scene[fridge_cfg.name]
    except KeyError:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    if debug:
        import ipdb

        ipdb.set_trace()

    joint_positions = _select_articulation_joints(fridge, joint_keywords)
    if joint_positions is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    door_angle = torch.abs(joint_positions).amax(dim=1)
    return door_angle > angle_threshold


def fridge_door_closed(
    env: ManagerBasedRLEnv,
    fridge_cfg: SceneEntityCfg = SceneEntityCfg("fridge"),
    joint_keywords: tuple[str, ...] = ("door",),
    angle_threshold: float = 0.1,
) -> torch.Tensor:
    """Detect whether the fridge door articulation is below the closing threshold."""

    try:
        fridge: Articulation = env.scene[fridge_cfg.name]
    except KeyError:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    joint_positions = _select_articulation_joints(fridge, joint_keywords)
    if joint_positions is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    door_angle = torch.abs(joint_positions).amax(dim=1)
    # print(f"Door angles: {door_angle} <=> {angle_threshold}")
    return door_angle < angle_threshold


def object_near_target(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    target_cfg: SceneEntityCfg | None = None,
    target_position: tuple[float, float, float] | None = None,
    xy_threshold: float = 0.15,
    z_threshold: float = 0.05,
) -> torch.Tensor:
    """Check whether the object is positioned close to the given target."""

    try:
        obj = env.scene[object_cfg.name]
    except KeyError:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    if not hasattr(obj, "data") or not hasattr(obj.data, "root_pos_w"):
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
        target_tensor = torch.tensor(
            target_position, dtype=object_pos.dtype, device=env.device
        )
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
    left_ee_frame_cfg: SceneEntityCfg | None = None,
    right_ee_frame_cfg: SceneEntityCfg | None = None,
    xy_threshold: float = 0.15,
    z_threshold: float = 0.05,
) -> torch.Tensor:
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

    if (
        robot_cfg is not None
        and left_ee_frame_cfg is not None
        and right_ee_frame_cfg is not None
    ):
        grasped = object_grasped(
            env,
            robot_cfg=robot_cfg,
            left_ee_frame_cfg=left_ee_frame_cfg,
            right_ee_frame_cfg=right_ee_frame_cfg,
            object_cfg=object_cfg,
        )
    else:
        grasped = torch.zeros_like(near_target)

    return torch.logical_and(near_target, torch.logical_not(grasped))


def _resolve_root_position(
    env: ManagerBasedRLEnv,
    entity_cfg: SceneEntityCfg | None,
) -> torch.Tensor | None:
    if entity_cfg is None or entity_cfg.name == "":
        return None

    try:
        entity = env.scene[entity_cfg.name]
    except KeyError:
        return None

    if hasattr(entity, "data") and hasattr(entity.data, "root_pos_w"):
        return entity.data.root_pos_w
    return None


def robot_to_object_distance(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor | None:
    robot_pos = _resolve_root_position(env, robot_cfg)
    object_pos = _resolve_root_position(env, object_cfg)
    if robot_pos is None or object_pos is None:
        return None
    return torch.linalg.vector_norm(robot_pos - object_pos, dim=1)


def robot_close_to_object(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    distance_threshold: float,
) -> torch.Tensor:
    """Check if the robot is within a certain distance to the object."""
    distances = robot_to_object_distance(env, robot_cfg, object_cfg)
    if distances is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return distances < distance_threshold


def robot_close_to_target(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg | None = None,
    target_position: tuple[float, float, float] | None = None,
    distance_threshold: float = 0.5,
    include_z: bool = True,
) -> torch.Tensor:
    """Check if the robot is within a certain distance to the target."""
    robot_pos = _resolve_root_position(env, robot_cfg)

    if robot_pos is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    target_pos = (
        _resolve_root_position(env, target_cfg) if target_cfg is not None else None
    )

    if target_pos is None:
        if target_position is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        target_pos = torch.tensor(
            target_position, dtype=robot_pos.dtype, device=env.device
        ).repeat(robot_pos.shape[0], 1)

    if not include_z:
        robot_pos = robot_pos[:, :2]
        target_pos = target_pos[:, :2]

    distances = torch.linalg.vector_norm(robot_pos - target_pos, dim=1)
    return distances < distance_threshold
