import math
<<<<<<< HEAD

=======
>>>>>>> a6f5944 (fix : working rl environment for fridge stocking.)
from typing import List, Literal, Sequence

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

import leisaac.enhance.envs.mdp as enhance_mdp


def randomize_objects_permutation(
    names: Sequence[str],
    position_noise: float = 0.05,
    yaw_range: tuple[float, float] = (-math.pi, math.pi),
    noise_axes: Sequence[str] = ("x", "y"),
    scene_cfg=None,
    reference_name: str | None = None,
) -> EventTerm:
    static_defaults: dict[str, dict[str, tuple[float, ...]]] = {}
    if scene_cfg is not None:
        names_to_extract = set(names)
        if reference_name is not None:
            names_to_extract.add(reference_name)
        for name in names_to_extract:
            cfg = getattr(scene_cfg, name, None)
            if cfg is None:
                continue
            prim_path = getattr(cfg, "prim_path", None)
            init_state = getattr(cfg, "init_state", None)
            pos = getattr(init_state, "pos", None) if init_state is not None else None
            rot = getattr(init_state, "rot", None) if init_state is not None else None
            if prim_path is not None and pos is not None and rot is not None:
                static_defaults[name] = {
                    "prim_path": prim_path,
                    "pos": tuple(pos),
                    "rot": tuple(rot),
                }
    params = {
        "asset_cfgs": [SceneEntityCfg(name) for name in names],
        "position_noise": position_noise,
        "yaw_range": yaw_range,
        "noise_axes": noise_axes,
        "static_defaults": static_defaults,
    }
    if reference_name is not None:
        params["reference_asset_cfg"] = SceneEntityCfg(reference_name)
    return EventTerm(
        func=enhance_mdp.randomize_objects_permutation,
        mode="reset",
        params=params,
    )


def randomize_object_uniform(
    name: str,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]] | None = None,
) -> EventTerm:
    if velocity_range is None:
        velocity_range = {}
    return EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": pose_range,
            "velocity_range": velocity_range,
            "asset_cfg": SceneEntityCfg(name),
        },
    )


def randomize_object_asset_uniform(
    name: str,
    pose_range: dict[str, tuple[float, float]],
) -> EventTerm:
    return EventTerm(
        func=enhance_mdp.randomize_object_asset_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name),
            "pose_range": pose_range,
        },
    )


def randomize_camera_uniform(
    name: str,
    pose_range: dict[str, tuple[float, float]],
    convention: Literal["ros", "opengl", "world"] = "ros",
) -> EventTerm:
    print(f"Domain randomization on camera: {name}")
    return EventTerm(
        func=enhance_mdp.randomize_camera_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name),
            "pose_range": pose_range,
            "convention": convention,
        },
    )


def domain_randomization(env_cfg, random_options: List[EventTerm]):
    for idx, event_item in enumerate(random_options):
        setattr(env_cfg.events, f"domain_randomize_{idx}", event_item)


def randomize_objects_permutation(
    names: Sequence[str],
    position_noise: float = 0.05,
    yaw_range: tuple[float, float] = (-math.pi, math.pi),
    noise_axes: Sequence[str] = ("x", "y"),
    scene_cfg=None,
    reference_name: str | None = None,
) -> EventTerm:
    static_defaults: dict[str, dict[str, tuple[float, ...]]] = {}
    if scene_cfg is not None:
        names_to_extract = set(names)
        if reference_name is not None:
            names_to_extract.add(reference_name)
        for name in names_to_extract:
            cfg = getattr(scene_cfg, name, None)
            if cfg is None:
                continue
            prim_path = getattr(cfg, "prim_path", None)
            init_state = getattr(cfg, "init_state", None)
            pos = getattr(init_state, "pos", None) if init_state is not None else None
            rot = getattr(init_state, "rot", None) if init_state is not None else None
            if prim_path is not None and pos is not None and rot is not None:
                static_defaults[name] = {
                    "prim_path": prim_path,
                    "pos": tuple(pos),
                    "rot": tuple(rot),
                }
    params = {
        "asset_cfgs": [SceneEntityCfg(name) for name in names],
        "position_noise": position_noise,
        "yaw_range": yaw_range,
        "noise_axes": noise_axes,
        "static_defaults": static_defaults,
    }
    if reference_name is not None:
        params["reference_asset_cfg"] = SceneEntityCfg(reference_name)
    return EventTerm(
        func=enhance_mdp.randomize_objects_permutation,
        mode="reset",
        params=params,
    )
