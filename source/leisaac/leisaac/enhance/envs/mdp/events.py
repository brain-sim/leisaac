import torch

from typing import Literal, Sequence

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera
from isaaclab.envs import ManagerBasedRLEnv


def randomize_objects_permutation(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfgs: Sequence[SceneEntityCfg],
    position_noise: float = 0.05,
    yaw_range: tuple[float | torch.Tensor, float | torch.Tensor] = (-torch.pi, torch.pi),
    noise_axes: Sequence[str] = ("x", "y"),
    static_defaults: dict[str, dict[str, tuple[float, ...]]] | None = None,
    reference_asset_cfg: SceneEntityCfg | None = None,
):
    """Permute rigid or static prims across their default poses with optional noise and yaw variation."""

    num_assets = len(asset_cfgs)
    if num_assets == 0 or env_ids.numel() == 0:
        return

    device = torch.device(env.device)
    env_origins = env.scene.env_origins[env_ids].to(device)
    num_envs = env_ids.numel()

    if static_defaults is None:
        static_defaults = {}

    assets = []
    base_local_positions = []
    base_orientations = []
    base_velocities = []
    prim_paths = []

    reference_positions: torch.Tensor
    reference_orientations: torch.Tensor
    if reference_asset_cfg is not None:
        reference_asset = env.scene[reference_asset_cfg.name]
        if hasattr(reference_asset, "data") and hasattr(reference_asset.data, "default_root_state"):
            reference_positions = reference_asset.data.default_root_state[env_ids].to(device)[:, 0:3]
            reference_orientations = reference_asset.data.default_root_state[env_ids].to(device)[:, 3:7]
        else:
            defaults = static_defaults.get(reference_asset_cfg.name) if static_defaults else None
            if defaults is None:
                raise KeyError(
                    f"Static defaults for reference asset '{reference_asset_cfg.name}' not provided."
                )
            pos_tuple = defaults.get("pos")
            rot_tuple = defaults.get("rot")
            if pos_tuple is None or rot_tuple is None:
                raise ValueError(f"Incomplete static defaults for reference asset '{reference_asset_cfg.name}'.")
            ref_pos = torch.tensor(pos_tuple, dtype=torch.float32, device=device)
            reference_positions = ref_pos.unsqueeze(0) + env_origins
            ref_rot = torch.tensor(rot_tuple, dtype=torch.float32, device=device)
            reference_orientations = ref_rot.unsqueeze(0).expand(num_envs, -1)
    else:
        reference_positions = env_origins.clone()
        reference_orientations = torch.zeros((num_envs, 4), dtype=torch.float32, device=device)
        reference_orientations[:, 0] = 1.0

    reference_orientations = reference_orientations / reference_orientations.norm(dim=-1, keepdim=True).clamp_min(1e-9)

    for cfg in asset_cfgs:
        asset = env.scene[cfg.name]
        assets.append(asset)

        if hasattr(asset, "data") and hasattr(asset.data, "default_root_state"):
            default_state = asset.data.default_root_state[env_ids].to(device)
            world_positions = default_state[:, 0:3]
            delta_positions = world_positions - reference_positions
            local_positions = math_utils.quat_apply_inverse(reference_orientations, delta_positions)
            base_local_positions.append(local_positions)
            base_orientations.append(default_state[:, 3:7])
            base_velocities.append(default_state[:, 7:13])
            prim_paths.append(None)
        else:
            defaults = static_defaults.get(cfg.name)
            if defaults is None:
                raise KeyError(
                    f"Static defaults for '{cfg.name}' not provided."
                    " Pass the originating scene config to randomize_objects_permutation so"
                    " default poses can be captured."
                )
            prim_path = defaults.get("prim_path")
            pos_tuple = defaults.get("pos")
            rot_tuple = defaults.get("rot")
            if prim_path is None or pos_tuple is None or rot_tuple is None:
                raise ValueError(f"Incomplete static defaults for '{cfg.name}'.")

            local_positions = torch.tensor(pos_tuple, dtype=torch.float32, device=device).unsqueeze(0).expand(num_envs, -1)
            local_orientations = torch.tensor(rot_tuple, dtype=torch.float32, device=device).unsqueeze(0).expand(num_envs, -1)
            base_local_positions.append(local_positions)
            base_orientations.append(math_utils.quat_mul(reference_orientations, local_orientations))
            base_velocities.append(torch.zeros(num_envs, 6, device=device))
            prim_paths.append(prim_path)

    base_local_positions = torch.stack(base_local_positions, dim=0)  # (assets, envs, 3)
    base_orientations = torch.stack(base_orientations, dim=0)  # (assets, envs, 4)
    base_velocities = torch.stack(base_velocities, dim=0)  # (assets, envs, 6)

    permutations = torch.stack([torch.randperm(num_assets, device=device) for _ in range(num_envs)], dim=0)

    local_positions_env_first = base_local_positions.permute(1, 0, 2)  # (envs, assets, 3)
    base_local_positions_env_first = local_positions_env_first.clone()
    permuted_local_positions_env_first = torch.gather(
        local_positions_env_first,
        dim=1,
        index=permutations.unsqueeze(-1).expand(-1, -1, 3),
    )

    if position_noise > 0.0:
        axis_to_index = {"x": 0, "y": 1, "z": 2}
        valid_axes = [axis_to_index[axis.lower()] for axis in noise_axes if axis.lower() in ("x", "y")]
        if valid_axes:
            noise_extent = min(float(abs(position_noise)), 0.02)
            noise = (torch.rand((num_envs, num_assets, len(valid_axes)), device=device) * 2.0 - 1.0) * noise_extent
            for offset, axis_idx in enumerate(valid_axes):
                permuted_local_positions_env_first[:, :, axis_idx] += noise[:, :, offset]

    permuted_local_positions_env_first[:, :, 2] = base_local_positions_env_first[:, :, 2]
    world_positions_env_first = math_utils.quat_apply(
        reference_orientations.unsqueeze(1).expand(-1, num_assets, -1).reshape(-1, 4),
        permuted_local_positions_env_first.reshape(-1, 3),
    ).view(num_envs, num_assets, 3) + reference_positions.unsqueeze(1)

    world_positions = world_positions_env_first.permute(1, 0, 2)

    yaw_min, yaw_max = float(yaw_range[0]), float(yaw_range[1])
    yaw_offsets = torch.empty((num_envs, num_assets), device=device).uniform_(yaw_min, yaw_max)
    zeros = torch.zeros_like(yaw_offsets)
    delta_quats = math_utils.quat_from_euler_xyz(
        zeros.flatten(),
        zeros.flatten(),
        yaw_offsets.flatten(),
    ).view(num_envs, num_assets, 4)

    base_orientations_env_first = base_orientations.permute(1, 0, 2)
    new_orientations_env_first = math_utils.quat_mul(base_orientations_env_first, delta_quats)
    new_orientations = new_orientations_env_first.permute(1, 0, 2)

    env_id_list = env_ids.tolist()

    for idx, (asset, positions, orientations, velocities, prim_path) in enumerate(
        zip(assets, world_positions, new_orientations, base_velocities, prim_paths)
    ):
        if hasattr(asset, "write_root_pose_to_sim"):
            target_device = torch.device(getattr(asset, "device", env.device))
            poses = torch.cat([positions.to(target_device), orientations.to(target_device)], dim=-1)
            asset.write_root_pose_to_sim(poses, env_ids=env_ids)
            if hasattr(asset, "write_root_velocity_to_sim"):
                asset.write_root_velocity_to_sim(velocities.to(target_device), env_ids=env_ids)
            continue

        positions_cpu = positions.detach().cpu()
        orientations_cpu = orientations.detach().cpu()

        if hasattr(asset, "set_world_poses"):
            try:
                asset.set_world_poses(positions_cpu, orientations_cpu, indices=env_id_list)
                continue
            except TypeError:
                try:
                    asset.set_world_poses(positions_cpu, orientations_cpu)
                    continue
                except TypeError:
                    pass

        if hasattr(asset, "set_world_pose"):
            for env_offset, env_index in enumerate(env_id_list):
                position = positions_cpu[env_offset].numpy()
                orientation = orientations_cpu[env_offset].numpy()
                try:
                    asset.set_world_pose(
                        position=position,
                        orientation=orientation,
                        index=env_index,
                    )
                except TypeError:
                    asset.set_world_pose(position=position, orientation=orientation)
            continue

        if prim_path is not None:
            msg = (
                "Unable to randomize prim pose for static asset '{name}'. "
                "Consider providing an asset interface exposing pose setters."
            )
            raise RuntimeError(msg.format(name=asset_cfgs[idx].name))


def randomize_camera_uniform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict[str, float],
    convention: Literal["opengl", "ros", "world"] = "ros",
):
    """Reset the camera to a random position and rotation uniformly within the given ranges.

    * It samples the camera position and rotation from the given ranges and adds them to the 
      default camera position and rotation, before setting them into the physics simulation.

    The function takes a dictionary of pose ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or rotation is set to zero for that axis.
    """
    asset: Camera = env.scene[asset_cfg.name]

    ori_pos_w = asset.data.pos_w[env_ids]
    if convention == "ros":
        ori_quat_w = asset.data.quat_w_ros[env_ids]
    elif convention == "opengl":
        ori_quat_w = asset.data.quat_w_opengl[env_ids]
    elif convention == "world":
        ori_quat_w = asset.data.quat_w_world[env_ids]

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = ori_pos_w[:, 0:3] + rand_samples[:, 0:3]  # camera usually spawn with robot, so no need to add env_origins
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(ori_quat_w, orientations_delta)

    asset.set_world_poses(positions, orientations, env_ids, convention)
