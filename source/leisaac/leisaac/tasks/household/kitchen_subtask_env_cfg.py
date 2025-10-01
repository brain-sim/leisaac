from __future__ import annotations

from typing import Dict, Sequence

import isaaclab.sim as sim_utils
import torch
from brain_sim_assets.props.kitchen.scene_sets.fridge_stocking import (
    bsFridgeStockingEntitiesGenerator,
)
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from leisaac.assets.scenes.test import TEST_WITH_CUBE_CFG, TEST_WITH_CUBE_USD_PATH
from leisaac.utils.domain_randomization import (
    domain_randomization,
    randomize_camera_uniform,
    randomize_object_uniform,
)
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ..template import (
    XLeRobotObservationsCfg,
    XLeRobotRewardsCfg,
    XLeRobotTaskEnvCfg,
    XLeRobotTaskSceneCfg,
    XLeRobotTerminationsCfg,
)
from . import mdp
from .fridge_stocking_env_cfg import STORAGE_PLATE_TARGET_POSITION

ROBOT_CFG = SceneEntityCfg("robot")
RIGHT_EE_CFG = SceneEntityCfg("right_ee_frame")
FRIDGE_CFG = SceneEntityCfg("fridge")
STORAGE_PLATE_CFG = SceneEntityCfg("storage_plate")
JUICE_BOTTLE_CFG = SceneEntityCfg("juice_bottle")
ORANGE_CFG = SceneEntityCfg("fruit_bundle")

COUNTER_TOP_TARGET_POSITION = (1.10, -1.85, 0.95)
COUNTER_SHELF_TARGET_POSITION = (1.62, -1.40, 1.32)


def _standard_mimic_defaults(name_suffix: str) -> Dict[str, object]:
    return {
        "name": f"kitchen_{name_suffix}",
        "generation_guarantee": True,
        "generation_keep_failed": True,
        "generation_num_trials": 10,
        "generation_select_src_per_subtask": True,
        "generation_transform_first_robot_pose": False,
        "generation_interpolate_from_last_target_pose": True,
        "generation_relative": True,
        "max_num_failures": 30,
        "seed": 42,
    }


@configclass
class KitchenManipulationSceneCfg(XLeRobotTaskSceneCfg):
    """Kitchen scene shared by the subtask-driven manipulation tasks."""

    scene: AssetBaseCfg = TEST_WITH_CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=1500.0),
    )

    _entities = bsFridgeStockingEntitiesGenerator.get_fridge_stocking_entities()
    kitchen_background: AssetBaseCfg = _entities["kitchen_background"]
    fridge: AssetBaseCfg = _entities["fridge"]
    stock_table: AssetBaseCfg = _entities["stock_table"]
    storage_plate: AssetBaseCfg = _entities["storage_plate"]
    juice_bottle: AssetBaseCfg = _entities["juice_bottle"]
    fruit_bundle: AssetBaseCfg = _entities["fruit_bundle"]
    prep_knives: AssetBaseCfg = _entities["prep_knives"]


@configclass
class KitchenSubtaskBaseEnvCfg(XLeRobotTaskEnvCfg):
    """Common setup (actions, randomization) for the kitchen manipulation tasks."""

    scene: KitchenManipulationSceneCfg = KitchenManipulationSceneCfg(env_spacing=8.0)
    randomized_objects: Sequence[str] = ()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (3.2, -2.4, 1.6)
        self.viewer.lookat = (4.4, -2.4, 1.0)

        self.actions.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "Rotation_2",
                "Pitch_2",
                "Elbow_2",
                "Wrist_Pitch_2",
                "Wrist_Roll_2",
                "Jaw_2",
            ],
            scale=1.0,
        )
        self.actions.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "Rotation",
                "Pitch",
                "Elbow",
                "Wrist_Pitch",
                "Wrist_Roll",
                "Jaw",
            ],
            scale=1.0,
        )
        self.actions.base_motion_action = mdp.JointVelocityActionCfg(
            asset_name="robot",
            joint_names=["axle_0_joint", "axle_1_joint", "axle_2_joint"],
            scale=20.0,
        )
        self.actions.head_pan_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["head_pan_joint"],
            scale=0.4,
        )

        parse_usd_and_create_subassets(TEST_WITH_CUBE_USD_PATH, self)

        randomizers = []
        for object_name in self.randomized_objects:
            randomizers.append(
                randomize_object_uniform(
                    object_name,
                    pose_range={
                        "x": (-0.075, 0.075),
                        "y": (-0.075, 0.075),
                        "z": (0.0, 0.0),
                        "yaw": (-30 * torch.pi / 180, 30 * torch.pi / 180),
                    },
                )
            )
        randomizers.append(
            randomize_camera_uniform(
                "top",
                pose_range={
                    "x": (-0.005, 0.005),
                    "y": (-0.005, 0.005),
                    "z": (-0.005, 0.005),
                    "roll": (-0.05 * torch.pi / 180, 0.05 * torch.pi / 180),
                    "pitch": (-0.05 * torch.pi / 180, 0.05 * torch.pi / 180),
                    "yaw": (-0.05 * torch.pi / 180, 0.05 * torch.pi / 180),
                },
                convention="opengl",
            )
        )

        domain_randomization(self, random_options=randomizers)


# ---------------------------------------------------------------------------
# Task 1a: Orange into fridge
# ---------------------------------------------------------------------------

_FRIDGE_ORANGE_SEQUENCE = mdp.resolve_subtask_sequence(
    [
        {
            "name": "approach_object",
            "description": "Approach the orange on the prep table.",
            "reward": 2.0,
            "params": {
                "robot_cfg": ROBOT_CFG,
                "object_cfg": ORANGE_CFG,
                "distance_threshold": 0.65,
            },
            "mimic_kwargs": {
                "object_ref": "fruit_bundle",
                "subtask_term_offset_range": (6, 12),
            },
            "next_description": "Grasp the orange.",
        },
        {
            "name": "grasp_object",
            "description": "Grasp the orange with the right gripper.",
            "reward": 3.0,
            "params": {
                "robot_cfg": ROBOT_CFG,
                "ee_frame_cfg": RIGHT_EE_CFG,
                "object_cfg": ORANGE_CFG,
            },
            "mimic_kwargs": {
                "object_ref": "fruit_bundle",
                "subtask_term_offset_range": (6, 12),
            },
            "next_description": "Drive to the fridge while holding the orange.",
        },
        {
            "name": "reach_fridge",
            "description": "Navigate to the fridge while holding the orange.",
            "reward": 2.5,
            "params": {
                "robot_cfg": ROBOT_CFG,
                "target_cfg": FRIDGE_CFG,
                "distance_threshold": 0.9,
            },
            "mimic_kwargs": {
                "object_ref": "fridge",
                "subtask_term_offset_range": (8, 16),
            },
            "next_description": "Open the fridge door.",
        },
        {
            "name": "door_open",
            "description": "Open the fridge door.",
            "reward": 3.5,
            "params": {
                "fridge_cfg": FRIDGE_CFG,
            },
            "mimic_kwargs": {
                "object_ref": "fridge",
                "subtask_term_offset_range": (6, 12),
            },
            "next_description": "Place the orange onto the storage plate.",
        },
        {
            "name": "object_placed",
            "description": "Place the orange onto the storage plate.",
            "reward": 2.5,
            "params": {
                "object_cfg": ORANGE_CFG,
                "target_position": STORAGE_PLATE_TARGET_POSITION,
                "robot_cfg": ROBOT_CFG,
                "ee_frame_cfg": RIGHT_EE_CFG,
                "xy_threshold": 0.18,
                "z_threshold": 0.06,
            },
            "mimic_kwargs": {
                "object_ref": "fruit_bundle",
                "subtask_term_offset_range": (8, 16),
            },
            "next_description": "Close the fridge door.",
        },
        {
            "name": "door_closed",
            "description": "Close the fridge door.",
            "reward": 1.0,
            "params": {
                "fridge_cfg": FRIDGE_CFG,
            },
            "mimic_kwargs": {
                "object_ref": "fridge",
                "subtask_term_offset_range": (6, 12),
            },
        },
    ]
)

FridgeOrangeObservationsCfg = mdp.build_observations_cfg(
    XLeRobotObservationsCfg,
    "FridgeOrangeObservationsCfg",
    _FRIDGE_ORANGE_SEQUENCE,
)
FridgeOrangeRewardsCfg = mdp.build_rewards_cfg(
    XLeRobotRewardsCfg,
    "FridgeOrangeRewardsCfg",
    _FRIDGE_ORANGE_SEQUENCE,
)
FridgeOrangeTerminationsCfg = configclass(
    type(
        "FridgeOrangeTerminationsCfg",
        (XLeRobotTerminationsCfg,),
        {
            "__annotations__": {
                "success": DoneTerm,
            },
            "success": DoneTerm(
                func=mdp.fridge_stocking_completed,
                params={
                    "object_cfg": ORANGE_CFG,
                    "target_cfg": None,
                    "fridge_cfg": FRIDGE_CFG,
                    "robot_cfg": ROBOT_CFG,
                    "ee_frame_cfg": RIGHT_EE_CFG,
                    "target_position": STORAGE_PLATE_TARGET_POSITION,
                    "xy_threshold": 0.18,
                    "z_threshold": 0.06,
                },
            ),
        },
    )
)
_FRIDGE_ORANGE_MIMIC = mdp.build_mimic_subtask_configs(_FRIDGE_ORANGE_SEQUENCE)


@configclass
class FridgeOrangePlacementEnvCfg(KitchenSubtaskBaseEnvCfg):
    observations: FridgeOrangeObservationsCfg = FridgeOrangeObservationsCfg()
    rewards: FridgeOrangeRewardsCfg = FridgeOrangeRewardsCfg()
    terminations: FridgeOrangeTerminationsCfg = FridgeOrangeTerminationsCfg()
    randomized_objects: Sequence[str] = ("fruit_bundle",)


@configclass
class FridgeOrangePlacementMimicEnvCfg(FridgeOrangePlacementEnvCfg, MimicEnvCfg):
    """Mimic extension for the orange fridge placement task."""

    def __post_init__(self) -> None:
        super().__post_init__()
        for key, value in _standard_mimic_defaults("fridge_orange_v0").items():
            setattr(self.datagen_config, key, value)
        self.subtask_configs["xlerobot"] = _FRIDGE_ORANGE_MIMIC


# ---------------------------------------------------------------------------
# Task 1b: Bottle into fridge
# ---------------------------------------------------------------------------

_FRIDGE_BOTTLE_SEQUENCE = mdp.resolve_subtask_sequence(
    [
        {
            "name": "approach_object",
            "description": "Approach the bottle on the prep table.",
            "reward": 2.0,
            "params": {
                "robot_cfg": ROBOT_CFG,
                "object_cfg": JUICE_BOTTLE_CFG,
                "distance_threshold": 0.65,
            },
            "mimic_kwargs": {
                "object_ref": "juice_bottle",
                "subtask_term_offset_range": (6, 12),
            },
            "next_description": "Grasp the bottle.",
        },
        {
            "name": "grasp_object",
            "description": "Grasp the bottle with the right gripper.",
            "reward": 3.0,
            "params": {
                "robot_cfg": ROBOT_CFG,
                "ee_frame_cfg": RIGHT_EE_CFG,
                "object_cfg": JUICE_BOTTLE_CFG,
            },
            "mimic_kwargs": {
                "object_ref": "juice_bottle",
                "subtask_term_offset_range": (6, 12),
            },
            "next_description": "Drive to the fridge while holding the bottle.",
        },
        {
            "name": "reach_fridge",
            "description": "Navigate to the fridge while holding the bottle.",
            "reward": 2.5,
            "params": {
                "robot_cfg": ROBOT_CFG,
                "target_cfg": FRIDGE_CFG,
                "distance_threshold": 0.9,
            },
            "mimic_kwargs": {
                "object_ref": "fridge",
                "subtask_term_offset_range": (8, 16),
            },
            "next_description": "Open the fridge door.",
        },
        {
            "name": "door_open",
            "description": "Open the fridge door.",
            "reward": 3.5,
            "params": {
                "fridge_cfg": FRIDGE_CFG,
            },
            "mimic_kwargs": {
                "object_ref": "fridge",
                "subtask_term_offset_range": (6, 12),
            },
            "next_description": "Place the bottle onto the storage plate.",
        },
        {
            "name": "object_placed",
            "description": "Place the bottle onto the storage plate.",
            "reward": 2.5,
            "params": {
                "object_cfg": JUICE_BOTTLE_CFG,
                "target_position": STORAGE_PLATE_TARGET_POSITION,
                "robot_cfg": ROBOT_CFG,
                "ee_frame_cfg": RIGHT_EE_CFG,
                "xy_threshold": 0.18,
                "z_threshold": 0.06,
            },
            "mimic_kwargs": {
                "object_ref": "juice_bottle",
                "subtask_term_offset_range": (8, 16),
            },
            "next_description": "Close the fridge door.",
        },
        {
            "name": "door_closed",
            "description": "Close the fridge door.",
            "reward": 1.0,
            "params": {
                "fridge_cfg": FRIDGE_CFG,
            },
            "mimic_kwargs": {
                "object_ref": "fridge",
                "subtask_term_offset_range": (6, 12),
            },
        },
    ]
)

FridgeBottleObservationsCfg = mdp.build_observations_cfg(
    XLeRobotObservationsCfg,
    "FridgeBottleObservationsCfg",
    _FRIDGE_BOTTLE_SEQUENCE,
)
FridgeBottleRewardsCfg = mdp.build_rewards_cfg(
    XLeRobotRewardsCfg,
    "FridgeBottleRewardsCfg",
    _FRIDGE_BOTTLE_SEQUENCE,
)
FridgeBottleTerminationsCfg = configclass(
    type(
        "FridgeBottleTerminationsCfg",
        (XLeRobotTerminationsCfg,),
        {
            "__annotations__": {
                "success": DoneTerm,
            },
            "success": DoneTerm(
                func=mdp.fridge_stocking_completed,
                params={
                    "object_cfg": JUICE_BOTTLE_CFG,
                    "target_cfg": None,
                    "fridge_cfg": FRIDGE_CFG,
                    "robot_cfg": ROBOT_CFG,
                    "ee_frame_cfg": RIGHT_EE_CFG,
                    "target_position": STORAGE_PLATE_TARGET_POSITION,
                    "xy_threshold": 0.18,
                    "z_threshold": 0.06,
                },
            ),
        },
    )
)
_FRIDGE_BOTTLE_MIMIC = mdp.build_mimic_subtask_configs(_FRIDGE_BOTTLE_SEQUENCE)


@configclass
class FridgeBottlePlacementEnvCfg(KitchenSubtaskBaseEnvCfg):
    observations: FridgeBottleObservationsCfg = FridgeBottleObservationsCfg()
    rewards: FridgeBottleRewardsCfg = FridgeBottleRewardsCfg()
    terminations: FridgeBottleTerminationsCfg = FridgeBottleTerminationsCfg()
    randomized_objects: Sequence[str] = ("juice_bottle",)


@configclass
class FridgeBottlePlacementMimicEnvCfg(FridgeBottlePlacementEnvCfg, MimicEnvCfg):
    """Mimic extension for the bottle fridge placement task."""

    def __post_init__(self) -> None:
        super().__post_init__()
        for key, value in _standard_mimic_defaults("fridge_bottle_v0").items():
            setattr(self.datagen_config, key, value)
        self.subtask_configs["xlerobot"] = _FRIDGE_BOTTLE_MIMIC


# ---------------------------------------------------------------------------
# Task 2: Bottle to counter top
# ---------------------------------------------------------------------------

_COUNTER_BOTTLE_SEQUENCE = mdp.resolve_subtask_sequence(
    [
        {
            "name": "approach_object",
            "description": "Approach the bottle on the prep table.",
            "reward": 2.0,
            "params": {
                "robot_cfg": ROBOT_CFG,
                "object_cfg": JUICE_BOTTLE_CFG,
                "distance_threshold": 0.65,
            },
            "mimic_kwargs": {
                "object_ref": "juice_bottle",
                "subtask_term_offset_range": (6, 12),
            },
            "next_description": "Grasp the bottle.",
        },
        {
            "name": "grasp_object",
            "description": "Grasp the bottle with the right gripper.",
            "reward": 3.0,
            "params": {
                "robot_cfg": ROBOT_CFG,
                "ee_frame_cfg": RIGHT_EE_CFG,
                "object_cfg": JUICE_BOTTLE_CFG,
            },
            "mimic_kwargs": {
                "object_ref": "juice_bottle",
                "subtask_term_offset_range": (6, 12),
            },
            "next_description": "Navigate to the counter top.",
        },
        {
            "name": "reach_counter",
            "description": "Position the robot next to the counter placement zone.",
            "reward": 2.5,
            "params": {
                "robot_cfg": ROBOT_CFG,
                "target_position": COUNTER_TOP_TARGET_POSITION,
                "distance_threshold": 0.25,
            },
            "mimic_kwargs": {
                "object_ref": "juice_bottle",
                "subtask_term_offset_range": (6, 12),
            },
            "next_description": "Place the bottle on the counter top.",
        },
        {
            "name": "object_placed",
            "description": "Place the bottle on the counter top target.",
            "reward": 3.5,
            "params": {
                "object_cfg": JUICE_BOTTLE_CFG,
                "target_position": COUNTER_TOP_TARGET_POSITION,
                "robot_cfg": ROBOT_CFG,
                "ee_frame_cfg": RIGHT_EE_CFG,
                "xy_threshold": 0.12,
                "z_threshold": 0.05,
            },
            "mimic_kwargs": {
                "object_ref": "juice_bottle",
                "subtask_term_offset_range": (8, 16),
            },
        },
    ]
)

CounterBottleObservationsCfg = mdp.build_observations_cfg(
    XLeRobotObservationsCfg,
    "CounterBottleObservationsCfg",
    _COUNTER_BOTTLE_SEQUENCE,
)
CounterBottleRewardsCfg = mdp.build_rewards_cfg(
    XLeRobotRewardsCfg,
    "CounterBottleRewardsCfg",
    _COUNTER_BOTTLE_SEQUENCE,
)
CounterBottleTerminationsCfg = configclass(
    type(
        "CounterBottleTerminationsCfg",
        (XLeRobotTerminationsCfg,),
        {
            "__annotations__": {
                "success": DoneTerm,
            },
            "success": DoneTerm(
                func=mdp.object_placed,
                params={
                    "object_cfg": JUICE_BOTTLE_CFG,
                    "target_position": COUNTER_TOP_TARGET_POSITION,
                    "robot_cfg": ROBOT_CFG,
                    "ee_frame_cfg": RIGHT_EE_CFG,
                    "xy_threshold": 0.12,
                    "z_threshold": 0.05,
                },
            ),
        },
    )
)
_COUNTER_BOTTLE_MIMIC = mdp.build_mimic_subtask_configs(_COUNTER_BOTTLE_SEQUENCE)


@configclass
class CounterBottlePlacementEnvCfg(KitchenSubtaskBaseEnvCfg):
    observations: CounterBottleObservationsCfg = CounterBottleObservationsCfg()
    rewards: CounterBottleRewardsCfg = CounterBottleRewardsCfg()
    terminations: CounterBottleTerminationsCfg = CounterBottleTerminationsCfg()
    randomized_objects: Sequence[str] = ("juice_bottle",)


@configclass
class CounterBottlePlacementMimicEnvCfg(CounterBottlePlacementEnvCfg, MimicEnvCfg):
    """Mimic variant for the bottle-to-counter task."""

    def __post_init__(self) -> None:
        super().__post_init__()
        for key, value in _standard_mimic_defaults("counter_bottle_v0").items():
            setattr(self.datagen_config, key, value)
        self.subtask_configs["xlerobot"] = _COUNTER_BOTTLE_MIMIC


# ---------------------------------------------------------------------------
# Task 3: Plate into counter shelf
# ---------------------------------------------------------------------------

_COUNTER_PLATE_SEQUENCE = mdp.resolve_subtask_sequence(
    [
        {
            "name": "approach_object",
            "description": "Approach the plate on the prep table.",
            "reward": 2.0,
            "params": {
                "robot_cfg": ROBOT_CFG,
                "object_cfg": STORAGE_PLATE_CFG,
                "distance_threshold": 0.65,
            },
            "mimic_kwargs": {
                "object_ref": "storage_plate",
                "subtask_term_offset_range": (6, 12),
            },
            "next_description": "Grasp the plate.",
        },
        {
            "name": "grasp_object",
            "description": "Grasp the plate with the right gripper.",
            "reward": 3.0,
            "params": {
                "robot_cfg": ROBOT_CFG,
                "ee_frame_cfg": RIGHT_EE_CFG,
                "object_cfg": STORAGE_PLATE_CFG,
            },
            "mimic_kwargs": {
                "object_ref": "storage_plate",
                "subtask_term_offset_range": (6, 12),
            },
            "next_description": "Navigate to the open counter shelf.",
        },
        {
            "name": "reach_shelf",
            "description": "Move next to the target counter shelf.",
            "reward": 2.5,
            "params": {
                "robot_cfg": ROBOT_CFG,
                "target_position": COUNTER_SHELF_TARGET_POSITION,
                "distance_threshold": 0.25,
            },
            "mimic_kwargs": {
                "object_ref": "storage_plate",
                "subtask_term_offset_range": (6, 12),
            },
            "next_description": "Slide the plate inside the shelf.",
        },
        {
            "name": "object_placed",
            "description": "Place the plate inside the counter shelf.",
            "reward": 3.5,
            "params": {
                "object_cfg": STORAGE_PLATE_CFG,
                "target_position": COUNTER_SHELF_TARGET_POSITION,
                "robot_cfg": ROBOT_CFG,
                "ee_frame_cfg": RIGHT_EE_CFG,
                "xy_threshold": 0.12,
                "z_threshold": 0.05,
            },
            "mimic_kwargs": {
                "object_ref": "storage_plate",
                "subtask_term_offset_range": (8, 16),
            },
        },
    ]
)

CounterPlateObservationsCfg = mdp.build_observations_cfg(
    XLeRobotObservationsCfg,
    "CounterPlateObservationsCfg",
    _COUNTER_PLATE_SEQUENCE,
)
CounterPlateRewardsCfg = mdp.build_rewards_cfg(
    XLeRobotRewardsCfg,
    "CounterPlateRewardsCfg",
    _COUNTER_PLATE_SEQUENCE,
)
CounterPlateTerminationsCfg = configclass(
    type(
        "CounterPlateTerminationsCfg",
        (XLeRobotTerminationsCfg,),
        {
            "__annotations__": {
                "success": DoneTerm,
            },
            "success": DoneTerm(
                func=mdp.object_placed,
                params={
                    "object_cfg": STORAGE_PLATE_CFG,
                    "target_position": COUNTER_SHELF_TARGET_POSITION,
                    "robot_cfg": ROBOT_CFG,
                    "ee_frame_cfg": RIGHT_EE_CFG,
                    "xy_threshold": 0.12,
                    "z_threshold": 0.05,
                },
            ),
        },
    )
)
_COUNTER_PLATE_MIMIC = mdp.build_mimic_subtask_configs(_COUNTER_PLATE_SEQUENCE)


@configclass
class CounterPlatePlacementEnvCfg(KitchenSubtaskBaseEnvCfg):
    observations: CounterPlateObservationsCfg = CounterPlateObservationsCfg()
    rewards: CounterPlateRewardsCfg = CounterPlateRewardsCfg()
    terminations: CounterPlateTerminationsCfg = CounterPlateTerminationsCfg()
    randomized_objects: Sequence[str] = ("storage_plate",)


@configclass
class CounterPlatePlacementMimicEnvCfg(CounterPlatePlacementEnvCfg, MimicEnvCfg):
    """Mimic variant for the plate shelving task."""

    def __post_init__(self) -> None:
        super().__post_init__()
        for key, value in _standard_mimic_defaults("counter_plate_v0").items():
            setattr(self.datagen_config, key, value)
        self.subtask_configs["xlerobot"] = _COUNTER_PLATE_MIMIC


__all__ = [
    "FridgeOrangePlacementEnvCfg",
    "FridgeOrangePlacementMimicEnvCfg",
    "FridgeBottlePlacementEnvCfg",
    "FridgeBottlePlacementMimicEnvCfg",
    "CounterBottlePlacementEnvCfg",
    "CounterBottlePlacementMimicEnvCfg",
    "CounterPlatePlacementEnvCfg",
    "CounterPlatePlacementMimicEnvCfg",
    "KitchenManipulationSceneCfg",
    "KitchenSubtaskBaseEnvCfg",
]
