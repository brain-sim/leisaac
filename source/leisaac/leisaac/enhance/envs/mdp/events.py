from typing import Literal, Sequence

import isaaclab.utils.math as math_utils
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera
from isaacsim.core.prims import XFormPrim
from pxr import UsdGeom


def randomize_objects_permutation(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfgs: Sequence[SceneEntityCfg],
    position_noise: float = 0.05,
    yaw_range: tuple[float | torch.Tensor, float | torch.Tensor] = (
        -torch.pi,
        torch.pi,
    ),
    noise_axes: Sequence[str] = ("x", "y"),
    static_defaults: dict[str, dict[str, tuple[float, ...]]] | None = None,
    reference_asset_cfg: SceneEntityCfg | None = None,
):
    """Permute rigid or static prims across their default poses with optional noise and yaw variation."""
    print("Permutation randomization on objects:")

    def _resolve_env_prim_path(prim_path_template: str, env_index: int) -> str:
        """Expand an environment regex path to a concrete environment path."""
        env_path = env.scene.env_prim_paths[env_index]
        resolved = prim_path_template
        if "{ENV_REGEX_NS}" in resolved:
            resolved = resolved.replace("{ENV_REGEX_NS}", env_path)
        regex_ns = getattr(env.scene, "env_regex_ns", "")
        if regex_ns and regex_ns in resolved:
            resolved = resolved.replace(regex_ns, env_path)
        base_env_path = env.scene.env_prim_paths[0]
        if base_env_path in resolved and env_index != 0:
            resolved = resolved.replace(base_env_path, env_path)
        if "{ENV_NS}" in resolved:
            resolved = resolved.replace("{ENV_NS}", env.scene.env_ns)
        return resolved

    def _set_static_asset_pose(
        asset_obj,
        prim_path_template: str,
        positions_cpu: torch.Tensor,
        orientations_cpu: torch.Tensor,
        env_indices: Sequence[int],
    ) -> None:
        """Update static prim transforms for each environment."""
        handle_cache: dict[int, XFormPrim] = getattr(asset_obj, "_per_env_handles", {})
        if not isinstance(handle_cache, dict):
            handle_cache = {}
        for env_offset, env_index in enumerate(env_indices):
            resolved_path = _resolve_env_prim_path(prim_path_template, env_index)
            xform_prim = handle_cache.get(env_index)
            if xform_prim is None or not xform_prim.is_valid():
                xform_prim = XFormPrim(resolved_path, reset_xform_properties=True)
                handle_cache[env_index] = xform_prim

            positions = positions_cpu[env_offset].unsqueeze(0)
            orientations = orientations_cpu[env_offset].unsqueeze(0)

            xform_prim.set_world_poses(positions, orientations, usd=False)
            xform_prim.set_world_poses(positions, orientations, usd=True)
            xform_prim.set_default_state(positions=positions, orientations=orientations)

        try:
            setattr(asset_obj, "_per_env_handles", handle_cache)
        except AttributeError:
            pass

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
        if hasattr(reference_asset, "data") and hasattr(
            reference_asset.data, "default_root_state"
        ):
            reference_positions = reference_asset.data.default_root_state[env_ids].to(
                device
            )[:, 0:3]
            reference_orientations = reference_asset.data.default_root_state[
                env_ids
            ].to(device)[:, 3:7]
        else:
            defaults = (
                static_defaults.get(reference_asset_cfg.name)
                if static_defaults
                else None
            )
            if defaults is None:
                raise KeyError(
                    f"Static defaults for reference asset '{reference_asset_cfg.name}' not provided."
                )
            pos_tuple = defaults.get("pos")
            rot_tuple = defaults.get("rot")
            if pos_tuple is None or rot_tuple is None:
                raise ValueError(
                    f"Incomplete static defaults for reference asset '{reference_asset_cfg.name}'."
                )
            ref_pos = torch.tensor(pos_tuple, dtype=torch.float32, device=device)
            reference_positions = ref_pos.unsqueeze(0) + env_origins
            ref_rot = torch.tensor(rot_tuple, dtype=torch.float32, device=device)
            reference_orientations = ref_rot.unsqueeze(0).expand(num_envs, -1)
    else:
        reference_positions = env_origins.clone()
        reference_orientations = torch.zeros(
            (num_envs, 4), dtype=torch.float32, device=device
        )
        reference_orientations[:, 0] = 1.0
    reference_orientations = reference_orientations / reference_orientations.norm(
        dim=-1, keepdim=True
    ).clamp_min(1e-9)
    for cfg in asset_cfgs:
        asset = env.scene[cfg.name]
        assets.append(asset)
        if hasattr(asset, "data") and hasattr(asset.data, "default_root_state"):
            default_state = asset.data.default_root_state[env_ids].to(device)
            world_positions = default_state[:, 0:3]
            delta_positions = world_positions - reference_positions
            local_positions = math_utils.quat_apply_inverse(
                reference_orientations, delta_positions
            )
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
            local_positions = (
                torch.tensor(pos_tuple, dtype=torch.float32, device=device)
                .unsqueeze(0)
                .expand(num_envs, -1)
            )
            local_orientations = (
                torch.tensor(rot_tuple, dtype=torch.float32, device=device)
                .unsqueeze(0)
                .expand(num_envs, -1)
            )
            base_local_positions.append(local_positions)
            base_orientations.append(
                math_utils.quat_mul(reference_orientations, local_orientations)
            )
            base_velocities.append(torch.zeros(num_envs, 6, device=device))
            prim_paths.append(prim_path)
    base_local_positions = torch.stack(base_local_positions, dim=0)  # (assets, envs, 3)
    base_orientations = torch.stack(base_orientations, dim=0)  # (assets, envs, 4)
    base_velocities = torch.stack(base_velocities, dim=0)  # (assets, envs, 6)
    permutations = torch.stack(
        [torch.randperm(num_assets, device=device) for _ in range(num_envs)], dim=0
    )
    base_local_positions_env_first = base_local_positions.permute(1, 0, 2)
    base_orientations_env_first = base_orientations.permute(1, 0, 2)
    base_velocities_env_first = base_velocities.permute(1, 0, 2)
    gather_index = permutations.unsqueeze(-1)
    permuted_local_positions_env_first = torch.gather(
        base_local_positions_env_first,
        dim=1,
        index=gather_index.expand(-1, -1, 3),
    )
    permuted_orientations_env_first = torch.gather(
        base_orientations_env_first,
        dim=1,
        index=gather_index.expand(-1, -1, 4),
    )
    permuted_velocities_env_first = torch.gather(
        base_velocities_env_first,
        dim=1,
        index=gather_index.expand(-1, -1, 6),
    )
    if position_noise > 0.0:
        axis_to_index = {"x": 0, "y": 1, "z": 2}
        valid_axes = [
            axis_to_index[axis.lower()]
            for axis in noise_axes
            if axis.lower() in ("x", "y")
        ]
        if valid_axes:
            noise_extent = min(float(abs(position_noise)), 0.02)
            noise = (
                torch.rand((num_envs, num_assets, len(valid_axes)), device=device) * 2.0
                - 1.0
            ) * noise_extent
            for offset, axis_idx in enumerate(valid_axes):
                permuted_local_positions_env_first[:, :, axis_idx] += noise[
                    :, :, offset
                ]
    world_positions_env_first = math_utils.quat_apply(
        reference_orientations.unsqueeze(1).expand(-1, num_assets, -1).reshape(-1, 4),
        permuted_local_positions_env_first.reshape(-1, 3),
    ).view(num_envs, num_assets, 3) + reference_positions.unsqueeze(1)
    world_positions = world_positions_env_first.permute(1, 0, 2)
    yaw_min, yaw_max = float(yaw_range[0]), float(yaw_range[1])
    yaw_offsets = torch.empty((num_envs, num_assets), device=device).uniform_(
        yaw_min, yaw_max
    )
    zeros = torch.zeros_like(yaw_offsets)
    delta_quats = math_utils.quat_from_euler_xyz(
        zeros.flatten(),
        zeros.flatten(),
        yaw_offsets.flatten(),
    ).view(num_envs, num_assets, 4)
    new_orientations_env_first = math_utils.quat_mul(
        permuted_orientations_env_first, delta_quats
    )
    new_orientations_env_first = (
        new_orientations_env_first
        / new_orientations_env_first.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    )
    new_orientations = new_orientations_env_first.permute(1, 0, 2)
    permuted_velocities = permuted_velocities_env_first.permute(1, 0, 2)
    env_id_list = env_ids.tolist()
    for idx, (asset, positions, orientations, velocities, prim_path) in enumerate(
        zip(assets, world_positions, new_orientations, permuted_velocities, prim_paths)
    ):
        if hasattr(asset, "write_root_pose_to_sim"):
            target_device = torch.device(getattr(asset, "device", env.device))
            poses = torch.cat(
                [positions.to(target_device), orientations.to(target_device)], dim=-1
            )
            asset.write_root_pose_to_sim(poses, env_ids=env_ids)
            if hasattr(asset, "write_root_velocity_to_sim"):
                asset.write_root_velocity_to_sim(
                    velocities.to(target_device), env_ids=env_ids
                )
            continue
        positions_cpu = positions.detach().cpu()
        orientations_cpu = orientations.detach().cpu()
        if hasattr(asset, "set_world_poses"):
            try:
                asset.set_world_poses(
                    positions_cpu, orientations_cpu, indices=env_id_list
                )
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
            _set_static_asset_pose(
                asset,
                prim_path,
                positions_cpu,
                orientations_cpu,
                env_id_list,
            )
            continue
        msg = (
            "Unable to randomize prim pose for static asset '{name}'. "
            "Consider providing an asset interface exposing pose setters."
        )
        raise RuntimeError(msg.format(name=asset_cfgs[idx].name))


def _pick_randomization_target(env, base_path: str) -> str:
    """Return the prim we should actually move (payload if present, else the parent)."""
    stage = env.scene.stage
    prim = stage.GetPrimAtPath(base_path)
    if not prim or not prim.IsValid():
        return base_path

    # Prefer the single Xform child under this prim (typical for referenced props).
    children = [child for child in prim.GetChildren() if child.IsA(UsdGeom.Xform)]
    if len(children) == 1:
        return children[0].GetPath().pathString
    return base_path


def randomize_object_asset_uniform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict[str, tuple[float, float]],
) -> None:
    """Uniformly randomize the pose of a static asset (AssetBaseCfg)."""

    if env_ids is None or env_ids.numel() == 0:
        return

    device = torch.device(env.device)
    env_id_list = env_ids.tolist()
    asset_obj = env.scene[asset_cfg.name]
    prim_path_template = getattr(asset_obj, "prim_path", None)
    if prim_path_template is None:
        prim_paths = getattr(asset_obj, "prim_paths", None)
        if isinstance(prim_paths, (list, tuple)) and prim_paths:
            prim_path_template = prim_paths[0]
    if prim_path_template is None:
        raise AttributeError(
            f"Unable to resolve prim path for asset '{asset_cfg.name}'."
        )

    def _resolve_env_prim_path(prim_path_template: str, env_index: int) -> str:
        env_path = env.scene.env_prim_paths[env_index]
        resolved = prim_path_template
        if "{ENV_REGEX_NS}" in resolved:
            resolved = resolved.replace("{ENV_REGEX_NS}", env_path)
        regex_ns = getattr(env.scene, "env_regex_ns", "")
        if regex_ns and regex_ns in resolved:
            resolved = resolved.replace(regex_ns, env_path)
        base_env_path = env.scene.env_prim_paths[0]
        if base_env_path in resolved and env_index != 0:
            resolved = resolved.replace(base_env_path, env_path)
        if "{ENV_NS}" in resolved:
            resolved = resolved.replace("{ENV_NS}", env.scene.env_ns)
        return resolved

    handle_cache: dict[int, XFormPrim] = getattr(asset_obj, "_per_env_handles", {})
    if not isinstance(handle_cache, dict):
        handle_cache = {}
    default_pos_cache: dict[int, torch.Tensor] = getattr(
        asset_obj, "_per_env_default_pos", {}
    )
    if not isinstance(default_pos_cache, dict):
        default_pos_cache = {}
    default_rot_cache: dict[int, torch.Tensor] = getattr(
        asset_obj, "_per_env_default_rot", {}
    )
    if not isinstance(default_rot_cache, dict):
        default_rot_cache = {}

    base_positions = []
    base_orientations = []
    for env_index in env_id_list:
        resolved_path = _resolve_env_prim_path(prim_path_template, env_index)
        resolved_path = _pick_randomization_target(env, resolved_path)
        xform_prim = handle_cache.get(env_index)
        if xform_prim is None or not xform_prim.is_valid():
            xform_prim = XFormPrim(resolved_path)
            handle_cache[env_index] = xform_prim
        if env_index not in default_pos_cache or env_index not in default_rot_cache:
            poses, quats = xform_prim.get_world_poses()
            pos = poses[0]
            quat = quats[0]
            default_pos_cache[env_index] = torch.as_tensor(
                pos, dtype=torch.float32, device=device
            )
            default_rot_cache[env_index] = torch.as_tensor(
                quat, dtype=torch.float32, device=device
            )
        base_positions.append(default_pos_cache[env_index])
        base_orientations.append(default_rot_cache[env_index])

    base_positions = torch.stack(base_positions, dim=0).to(device)
    base_orientations = torch.stack(base_orientations, dim=0).to(device)

    def _sample_component(name: str) -> torch.Tensor:
        low, high = pose_range.get(name, (0.0, 0.0))
        low_t = torch.tensor(float(low), dtype=torch.float32, device=device)
        high_t = torch.tensor(float(high), dtype=torch.float32, device=device)
        return torch.rand(len(env_id_list), device=device) * (high_t - low_t) + low_t

    delta_pos = torch.zeros((len(env_id_list), 3), dtype=torch.float32, device=device)
    for axis, idx in (("x", 0), ("y", 1), ("z", 2)):
        if axis in pose_range:
            delta_pos[:, idx] = _sample_component(axis)

    roll = (
        _sample_component("roll")
        if "roll" in pose_range
        else torch.zeros(len(env_id_list), device=device)
    )
    pitch = (
        _sample_component("pitch")
        if "pitch" in pose_range
        else torch.zeros(len(env_id_list), device=device)
    )
    yaw = (
        _sample_component("yaw")
        if "yaw" in pose_range
        else torch.zeros(len(env_id_list), device=device)
    )

    delta_quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    new_orientations = math_utils.quat_mul(base_orientations, delta_quat)
    new_orientations = new_orientations / new_orientations.norm(
        dim=-1, keepdim=True
    ).clamp_min(1e-9)
    new_positions = base_positions + delta_pos

    new_positions_cpu = new_positions.detach().cpu()
    new_orientations_cpu = new_orientations.detach().cpu()

    for env_offset, env_index in enumerate(env_id_list):
        xform_prim = handle_cache[env_index]
        positions = new_positions_cpu[env_offset].unsqueeze(0)
        orientations = new_orientations_cpu[env_offset].unsqueeze(0)
        xform_prim.set_world_poses(positions, orientations)
        # xform_prim.set_default_state(positions=positions, orientations=orientations)
        # xform_prim.post_reset()

    try:
        setattr(asset_obj, "_per_env_handles", handle_cache)
        setattr(asset_obj, "_per_env_default_pos", new_positions)
        setattr(asset_obj, "_per_env_default_rot", new_orientations)
    except AttributeError:
        pass


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
    range_list = [
        pose_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device
    )
    positions = (
        ori_pos_w[:, 0:3] + rand_samples[:, 0:3]
    )  # camera usually spawn with robot, so no need to add env_origins
    orientations_delta = math_utils.quat_from_euler_xyz(
        rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    orientations = math_utils.quat_mul(ori_quat_w, orientations_delta)


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

    range_list = [
        pose_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device
    )

    positions = (
        ori_pos_w[:, 0:3] + rand_samples[:, 0:3]
    )  # camera usually spawn with robot, so no need to add env_origins
    orientations_delta = math_utils.quat_from_euler_xyz(
        rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    orientations = math_utils.quat_mul(ori_quat_w, orientations_delta)

    asset.set_world_poses(positions, orientations, env_ids, convention)
