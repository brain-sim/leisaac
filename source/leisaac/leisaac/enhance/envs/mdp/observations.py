from collections.abc import Sequence

import omni.usd
import torch

import isaaclab.utils.math as math_utils

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.envs.mdp.observations import image


def overlay_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Overlay the background image on the sim render image.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb", and only "rgb" is supported for overlay_image.
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        normalize: Whether to normalize the images. This depends on the selected data type.
            Defaults to True.

    Returns:
        The images produced at the last time-step, with the background image overlayed.
    """
    assert data_type == "rgb", "Only 'rgb' is supported for overlay_image."

    sim_image = image(env, sensor_cfg, data_type, convert_perspective_to_orthogonal, normalize)

    def image_overlapping(
        back_image: torch.Tensor,  # [num_env, H, W, C]
        fore_image: torch.Tensor,  # [num_env, H, W, C]
        back_mask: torch.Tensor | None = None,   # [num_env, H, W]
        fore_mask: torch.Tensor | None = None,   # [num_env, H, W]
        back_alpha: float = 0.5,
        fore_alpha: float = 0.5,
    ) -> torch.Tensor:
        """
        Overlap two images with masks.

        Args:
            back_image: background image [num_env, H, W, C]
            fore_image: foreground image [num_env, H, W, C]
            back_mask: background mask [num_env, H, W]
            fore_mask: foreground mask [num_env, H, W]
            back_alpha: background opacity (0-1)
            fore_alpha: foreground opacity (0-1)

        Returns:
            Overlapped image [num_env, H, W, C]
        """
        if back_mask is None:
            back_mask = torch.ones_like(back_image[:, :, :, 0], dtype=torch.bool, device=back_image.device).unsqueeze(-1)
        if fore_mask is None:
            fore_mask = torch.ones_like(fore_image[:, :, :, 0], dtype=torch.bool, device=fore_image.device).unsqueeze(-1)
        image = back_alpha * back_image * back_mask + fore_alpha * fore_image * fore_mask
        return torch.clamp(image, 0.0, 255.0).to(torch.uint8)

    semantic_id = env.foreground_semantic_id_mapping.get(sensor_cfg.name, None)
    if semantic_id is not None:
        camera_output = env.scene.sensors[sensor_cfg.name].data.output
        if env.cfg.rgb_overlay_mode == 'background':
            semantic_mask = camera_output["semantic_segmentation"]
            overlay_mask = semantic_mask == semantic_id
            sim_image = image_overlapping(
                back_image=env.rgb_overlay_images[sensor_cfg.name],
                fore_image=sim_image,
                back_mask=torch.logical_not(overlay_mask),
                fore_mask=overlay_mask,
                back_alpha=1.0,
                fore_alpha=1.0,
            )
        elif env.cfg.rgb_overlay_mode == 'debug':
            sim_image = image_overlapping(
                back_image=env.rgb_overlay_images[sensor_cfg.name],
                fore_image=sim_image,
                back_alpha=0.5,
                fore_alpha=0.5,
            )

    return sim_image


def ee_frame_state(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"), robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Return the state of the end effector frame in the robot coordinate system.
    """
    robot = env.scene[robot_cfg.name]
    robot_root_pos, robot_root_quat = robot.data.root_pos_w, robot.data.root_quat_w
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos, ee_frame_quat = ee_frame.data.target_pos_w[:, 0, :], ee_frame.data.target_quat_w[:, 0, :]
    ee_frame_pos_robot, ee_frame_quat_robot = math_utils.subtract_frame_transforms(
        robot_root_pos, robot_root_quat, ee_frame_pos, ee_frame_quat
    )
    ee_frame_state = torch.cat([ee_frame_pos_robot, ee_frame_quat_robot], dim=1)

    return ee_frame_state


def joint_pos_target(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions target of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos_target[:, asset_cfg.joint_ids]


def _resolve_env_prim_path(env: ManagerBasedEnv, expr: str, env_index: int) -> str:
    env_roots = getattr(env.scene, "env_prim_paths", None)
    env_root = (
        env_roots[env_index]
        if env_roots is not None
        else f"{env.scene.env_ns}/env_{env_index}"
    )
    resolved = (
        expr.replace("{ENV_REGEX_NS}", env_root)
        .replace("ENV_REGEX_NS", env_root)
        .replace("{ENV_NS}", env.scene.env_ns)
        .replace("ENV_NS", env.scene.env_ns)
        .replace("env_.*", f"env_{env_index}")
    )
    if not resolved.startswith("/"):
        resolved = f"{env_root.rstrip('/')}/{resolved.lstrip('/')}"
    return resolved


def _rotation_tensor_from_prim(prim, device: torch.device) -> torch.Tensor:
    matrix = omni.usd.get_world_transform_matrix(prim)
    rows = []
    for row_idx in range(3):
        usd_row = matrix.GetRow(row_idx)
        rows.append([float(usd_row[0]), float(usd_row[1]), float(usd_row[2])])
    return torch.tensor(rows, dtype=torch.float32, device=device)


def fridge_door_angles_from_prim_paths(
    env: ManagerBasedRLEnv,
    door_prim_paths: Sequence[str],
    parent_prim_paths: Sequence[str] | None = None,
) -> torch.Tensor:
    """Return fridge door angles (radians) computed from prim transforms.

    Args:
        env: The environment providing access to the USD stage.
        door_prim_paths: Prim path templates for the door xforms, one per angle column.
        parent_prim_paths: Optional parent prim templates. When omitted, the door's
            immediate parent path is used.

    Returns:
        Tensor of shape ``(num_envs, len(door_prim_paths))`` with angles in radians.
        Entries remain ``nan`` when either prim in the pair is missing.
    """

    door_exprs = list(door_prim_paths)
    if not door_exprs:
        return torch.empty(env.num_envs, 0, dtype=torch.float32, device=env.device)

    if parent_prim_paths is None:
        parent_exprs = [expr.rsplit("/", 1)[0] if "/" in expr else expr for expr in door_exprs]
    else:
        parent_exprs = list(parent_prim_paths)
        if len(parent_exprs) != len(door_exprs):
            raise ValueError("parent_prim_paths must match door_prim_paths length.")

    angles = torch.full(
        (env.num_envs, len(door_exprs)),
        float("nan"),
        dtype=torch.float32,
        device=env.device,
    )

    stage = env.scene.stage

    for env_index in range(env.num_envs):
        for col_index, (door_expr, parent_expr) in enumerate(zip(door_exprs, parent_exprs)):
            door_path = _resolve_env_prim_path(env, door_expr, env_index)
            parent_path = _resolve_env_prim_path(env, parent_expr, env_index)
            door_prim = stage.GetPrimAtPath(door_path)
            parent_prim = stage.GetPrimAtPath(parent_path)
            if not door_prim.IsValid() or not parent_prim.IsValid():
                continue

            door_rot = _rotation_tensor_from_prim(door_prim, env.device).unsqueeze(0)
            parent_rot = _rotation_tensor_from_prim(parent_prim, env.device).unsqueeze(0)

            door_quat = math_utils.quat_from_matrix(door_rot)[0]
            parent_quat = math_utils.quat_from_matrix(parent_rot)[0]
            relative_quat = math_utils.quat_mul(
                math_utils.quat_inv(parent_quat.unsqueeze(0)),
                door_quat.unsqueeze(0),
            )[0]
            axis_angle = math_utils.axis_angle_from_quat(relative_quat.unsqueeze(0))[0]
            angles[env_index, col_index] = torch.linalg.vector_norm(axis_angle, dim=-1)

    return angles
