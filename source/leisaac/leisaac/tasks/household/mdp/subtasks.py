from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence

from isaaclab.envs.mimic_env_cfg import SubTaskConfig
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .. import mdp
from .rewards import SequentialSubtaskReward


@dataclass(frozen=True)
class SubtaskFunctionSpec:
    """Specification describing how to evaluate a subtask."""

    func: Callable[..., Any]
    required_params: tuple[str, ...] = ()
    default_params: Dict[str, Any] = field(default_factory=dict)
    default_mimic_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedSubtask:
    """Concrete subtask entry with resolved runtime configuration."""

    name: str
    description: str
    reward: float
    func: Callable[..., Any]
    params: Dict[str, Any]
    next_description: str | None
    mimic_kwargs: Dict[str, Any]


_SUBTASK_REGISTRY: Dict[str, SubtaskFunctionSpec] = {
    "approach_object": SubtaskFunctionSpec(
        func=mdp.robot_close_to_object,
        required_params=("robot_cfg", "object_cfg"),
        default_params={"distance_threshold": 0.7},
        default_mimic_kwargs={
            "selection_strategy": "nearest_neighbor_object",
            "selection_strategy_kwargs": {"nn_k": 3},
            "action_noise": 0.002,
            "num_interpolation_steps": 4,
            "num_fixed_steps": 0,
            "apply_noise_during_interpolation": False,
        },
    ),
    "grasp_object": SubtaskFunctionSpec(
        func=mdp.object_grasped,
        required_params=(
            "robot_cfg",
            "left_ee_frame_cfg",
            "right_ee_frame_cfg",
            "object_cfg",
        ),
        default_mimic_kwargs={
            "selection_strategy": "nearest_neighbor_object",
            "selection_strategy_kwargs": {"nn_k": 3},
            "action_noise": 0.002,
            "num_interpolation_steps": 5,
            "num_fixed_steps": 0,
            "apply_noise_during_interpolation": False,
        },
    ),
    "reach_target": SubtaskFunctionSpec(
        func=mdp.robot_close_to_target,
        required_params=("robot_cfg",),
        default_params={"distance_threshold": 1.3},
        default_mimic_kwargs={
            "selection_strategy": "nearest_neighbor_object",
            "selection_strategy_kwargs": {"nn_k": 3},
            "action_noise": 0.002,
            "num_interpolation_steps": 6,
            "num_fixed_steps": 0,
            "apply_noise_during_interpolation": False,
        },
    ),
    "door_open": SubtaskFunctionSpec(
        func=mdp.fridge_door_opened,
        required_params=("fridge_prim_path", "door_joint_names"),
        default_mimic_kwargs={
            "selection_strategy": "nearest_neighbor_object",
            "selection_strategy_kwargs": {"nn_k": 3},
            "action_noise": 0.002,
            "num_interpolation_steps": 5,
            "num_fixed_steps": 0,
            "apply_noise_during_interpolation": False,
        },
    ),
    "object_placed": SubtaskFunctionSpec(
        func=mdp.object_placed,
        required_params=("object_cfg",),
        default_params={"xy_threshold": 0.18, "z_threshold": 0.06},
        default_mimic_kwargs={
            "selection_strategy": "nearest_neighbor_object",
            "selection_strategy_kwargs": {"nn_k": 3},
            "action_noise": 0.002,
            "num_interpolation_steps": 6,
            "num_fixed_steps": 0,
            "apply_noise_during_interpolation": False,
        },
    ),
    "door_closed": SubtaskFunctionSpec(
        func=mdp.fridge_door_closed,
        required_params=("fridge_prim_path", "door_joint_names"),
        default_mimic_kwargs={
            "selection_strategy": "nearest_neighbor_object",
            "selection_strategy_kwargs": {"nn_k": 3},
            "action_noise": 0.002,
            "num_interpolation_steps": 5,
            "num_fixed_steps": 0,
            "apply_noise_during_interpolation": False,
        },
    ),
}

_SUBTASK_ALIASES: Dict[str, str] = {
    "reach_fridge": "reach_target",
    "reach_counter": "reach_target",
    "reach_shelf": "reach_target",
}


def _lookup_subtask_definition(name: str) -> SubtaskFunctionSpec:
    resolved_name = name
    if name not in _SUBTASK_REGISTRY:
        resolved_name = _SUBTASK_ALIASES.get(name, name)
    try:
        return _SUBTASK_REGISTRY[resolved_name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown subtask name '{name}'. Please register it in _SUBTASK_REGISTRY."
        ) from exc


def resolve_subtask_sequence(
    sequence: Sequence[Dict[str, Any]],
) -> List[ResolvedSubtask]:
    """Resolve a user-provided subtask sequence into concrete runtime configuration."""

    resolved: List[ResolvedSubtask] = []
    for entry in sequence:
        name = entry["name"]
        spec = _lookup_subtask_definition(name)

        params = dict(spec.default_params)
        params.update(entry.get("params", {}))

        for required in spec.required_params:
            if required not in params:
                raise ValueError(
                    f"Subtask '{name}' is missing required parameter '{required}'. "
                    "Provide it via the 'params' field."
                )

        mimic_kwargs = dict(spec.default_mimic_kwargs)
        mimic_kwargs.update(entry.get("mimic_kwargs", {}))

        description = entry.get("description", "")
        reward = float(entry.get("reward", 0.0))
        next_description = entry.get("next_description")

        resolved.append(
            ResolvedSubtask(
                name=name,
                description=description,
                reward=reward,
                func=spec.func,
                params=params,
                next_description=next_description,
                mimic_kwargs=mimic_kwargs,
            )
        )
    return resolved


def _filter_params_for_callable(
    params: Dict[str, Any], func: Callable[..., Any]
) -> Dict[str, Any]:
    """Return only the parameters accepted by the given callable."""

    signature = inspect.signature(func)
    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    ):
        return params

    allowed = set(signature.parameters.keys())
    return {key: value for key, value in params.items() if key in allowed}


def build_subtask_observation_group(
    class_name: str, subtasks: Sequence[ResolvedSubtask]
):
    """Create an ObservationGroupCfg subclass that exposes one ObsTerm per subtask."""

    namespace: Dict[str, Any] = {}
    annotations: Dict[str, Any] = {}
    for subtask in subtasks:
        annotations[subtask.name] = ObsTerm
        namespace[subtask.name] = ObsTerm(
            func=subtask.func,
            params=_filter_params_for_callable(subtask.params, subtask.func),
        )

    def __post_init__(self) -> None:  # noqa: D401
        self.enable_corruption = False
        self.concatenate_terms = False

    namespace["__annotations__"] = annotations
    namespace["__post_init__"] = __post_init__

    group_cls = type(class_name, (ObsGroup,), namespace)
    return configclass(group_cls)


def build_observations_cfg(
    base_cls, class_name: str, subtasks: Sequence[ResolvedSubtask]
):
    """Attach generated subtask observation group to an observation config class."""

    group_cls = build_subtask_observation_group(f"{class_name}Subtasks", subtasks)

    namespace: Dict[str, Any] = {}
    annotations = dict(getattr(base_cls, "__annotations__", {}))
    annotations["subtask_terms"] = group_cls

    namespace["__annotations__"] = annotations
    namespace["subtask_terms"] = group_cls()

    obs_cls = type(class_name, (base_cls,), namespace)
    return configclass(obs_cls)


def build_rewards_cfg(base_cls, class_name: str, subtasks: Sequence[ResolvedSubtask]):
    """Create a rewards configuration with sequential progress and L2 action regularization."""

    sequential_reward = SequentialSubtaskReward(
        stage_funcs=[entry.func for entry in subtasks],
        stage_rewards=[entry.reward for entry in subtasks],
    )

    namespace: Dict[str, Any] = {}
    annotations = dict(getattr(base_cls, "__annotations__", {}))

    annotations["sequential_progress"] = RewTerm
    namespace["sequential_progress"] = RewTerm(
        func=sequential_reward,
        weight=1.0,
        params={"stage_params": [entry.params for entry in subtasks]},
    )

    annotations["action_reg"] = RewTerm
    namespace["action_reg"] = RewTerm(func=mdp.action_l2, weight=-1.0e-4)

    rewards_cls = type(
        class_name, (base_cls,), {"__annotations__": annotations, **namespace}
    )
    return configclass(rewards_cls)


def build_mimic_subtask_configs(
    subtasks: Sequence[ResolvedSubtask],
) -> List[SubTaskConfig]:
    """Convert resolved subtasks into Mimic datagen configs."""

    configs: List[SubTaskConfig] = []
    for idx, entry in enumerate(subtasks):
        kwargs = dict(entry.mimic_kwargs)
        kwargs.setdefault("subtask_term_signal", entry.name)
        kwargs.setdefault("description", entry.description)
        if entry.next_description is not None:
            kwargs.setdefault("next_subtask_description", entry.next_description)
        elif idx + 1 < len(subtasks):
            kwargs.setdefault("next_subtask_description", subtasks[idx + 1].description)
        configs.append(SubTaskConfig(**kwargs))
    return configs


def summarize_subtasks(subtasks: Sequence[ResolvedSubtask]) -> List[Dict[str, Any]]:
    """Return a summary list for logging or debugging purposes."""

    return [
        {
            "name": entry.name,
            "description": entry.description,
            "reward": entry.reward,
            "params": entry.params,
        }
        for entry in subtasks
    ]
