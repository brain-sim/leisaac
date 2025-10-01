from __future__ import annotations

import isaaclab.sim as sim_utils
import torch
from brain_sim_assets.props.kitchen.scene_sets.fridge_stocking import (
    bsFridgeStockingEntitiesGenerator,
)
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from leisaac.utils.domain_randomization import (
    domain_randomization,
    randomize_camera_uniform,
    randomize_object_uniform,
)

from ..template import (
    XLeRobotObservationsCfg,
    XLeRobotTaskEnvCfg,
    XLeRobotTaskSceneCfg,
    XLeRobotTerminationsCfg,
)
from . import mdp

BASE_KITCHEN_OFFSET = (0.0, 0.0, 0.0)  # (4.0, -4.0, 0.0)
FRIDGE_REL_POS = (4.7, -0.45, 0.8)
TABLE_REL_POS = (2.5, -2.4, 0.0)
PLATE_REL_POS = (0.5, 0.2, 0.77)
FRUIT_REL_POS = (3.0, -2.0, 0.75)
BOTTLE_REL_POS = (0.52, 0.025, 0.9)
KNIVES_REL_POS = (0.47, -0.22, 0.9)


def _to_world(relative_pos: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(base + rel for base, rel in zip(BASE_KITCHEN_OFFSET, relative_pos))


STORAGE_PLATE_TARGET_POSITION = _to_world(PLATE_REL_POS)


FRIDGE_STOCKING_ASSETS = bsFridgeStockingEntitiesGenerator.get_fridge_stocking_entities(
    base_pos=BASE_KITCHEN_OFFSET,
    include_background=True,
    fridge_pos=FRIDGE_REL_POS,
    table_pos=TABLE_REL_POS,
    plate_pos=PLATE_REL_POS,
    fruit_pos=FRUIT_REL_POS,
    bottle_pos=BOTTLE_REL_POS,
    knife_holder_pos=KNIVES_REL_POS,
)
KITCHEN_BACKGROUND_CFG = FRIDGE_STOCKING_ASSETS.pop("kitchen_background")
KITCHEN_SCENE_CFG = KITCHEN_BACKGROUND_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

ROBOT_CFG = SceneEntityCfg("robot")
LEFT_EE_CFG = SceneEntityCfg("left_ee_frame")
RIGHT_EE_CFG = SceneEntityCfg("right_ee_frame")
FRIDGE_CFG = SceneEntityCfg("fridge")
FRUIT_CFG = SceneEntityCfg("fruit_bundle")


def _to_rigid_object_cfg(asset_cfg: AssetBaseCfg) -> RigidObjectCfg:
    init_state = getattr(asset_cfg, "init_state", AssetBaseCfg.InitialStateCfg())
    return RigidObjectCfg(
        prim_path=asset_cfg.prim_path,
        spawn=asset_cfg.spawn,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=getattr(init_state, "pos", (0.0, 0.0, 0.0)),
            rot=getattr(init_state, "rot", (1.0, 0.0, 0.0, 0.0)),
            lin_vel=getattr(init_state, "lin_vel", (0.0, 0.0, 0.0)),
            ang_vel=getattr(init_state, "ang_vel", (0.0, 0.0, 0.0)),
        ),
    )


FRIDGE_STOCKING_ASSETS["fruit_bundle"] = _to_rigid_object_cfg(
    FRIDGE_STOCKING_ASSETS["fruit_bundle"]
)


FRIDGE_STOCKING_SEQUENCE = mdp.resolve_subtask_sequence(
    [
        {
            "name": "approach_object",
            "description": "Approach the fruit bundle on the prep table.",
            "reward": 2.0,
            "params": {
                "robot_cfg": ROBOT_CFG,
                "object_cfg": FRUIT_CFG,
                "distance_threshold": 0.9,
            },
            "mimic_kwargs": {
                "object_ref": "fruit_bundle",
                "subtask_term_offset_range": (6, 12),
            },
            "next_description": "Grasp the fruit bundle.",
        },
        {
            "name": "grasp_object",
            "description": "Grasp the fruit bundle with the right gripper.",
            "reward": 3.0,
            "params": {
                "robot_cfg": ROBOT_CFG,
                "left_ee_frame_cfg": LEFT_EE_CFG,
                "right_ee_frame_cfg": RIGHT_EE_CFG,
                "object_cfg": FRUIT_CFG,
            },
            "mimic_kwargs": {
                "object_ref": "fruit_bundle",
                "subtask_term_offset_range": (6, 12),
            },
            "next_description": "Drive to the fridge while holding the fruit.",
        },
        {
            "name": "reach_fridge",
            "description": "Navigate the robot close to the fridge while holding the fruit.",
            "reward": 2.0,
            "params": {
                "robot_cfg": ROBOT_CFG,
                "target_cfg": FRIDGE_CFG,
                "target_position": FRIDGE_REL_POS,
                "distance_threshold": 1.2,
                "object_cfg": FRUIT_CFG,
                "left_ee_frame_cfg": LEFT_EE_CFG,
                "right_ee_frame_cfg": RIGHT_EE_CFG,
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
            "reward": 2.5,
            "params": {
                "fridge_cfg": FRIDGE_CFG,
            },
            "mimic_kwargs": {
                "object_ref": "fridge",
                "subtask_term_offset_range": (6, 12),
            },
            "next_description": "Place the fruit bundle onto the storage plate.",
        },
        {
            "name": "object_placed",
            "description": "Place the fruit bundle onto the storage plate inside the fridge.",
            "reward": 3.5,
            "params": {
                "object_cfg": FRUIT_CFG,
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
            "description": "Close the fridge door to finish stocking.",
            "reward": 2.5,
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

FridgeStockingObservationsCfg = mdp.build_observations_cfg(
    XLeRobotObservationsCfg,
    "FridgeStockingObservationsCfg",
    FRIDGE_STOCKING_SEQUENCE,
)


@configclass
class FridgeStockingSceneCfg(XLeRobotTaskSceneCfg):
    scene: AssetBaseCfg = KITCHEN_SCENE_CFG
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=1500.0),
    )
    fridge: AssetBaseCfg = FRIDGE_STOCKING_ASSETS["fridge"]
    stock_table: AssetBaseCfg = FRIDGE_STOCKING_ASSETS["stock_table"]
    storage_plate: AssetBaseCfg = FRIDGE_STOCKING_ASSETS["storage_plate"]
    juice_bottle: AssetBaseCfg = FRIDGE_STOCKING_ASSETS["juice_bottle"]
    fruit_bundle: RigidObjectCfg = FRIDGE_STOCKING_ASSETS["fruit_bundle"]
    prep_knives: AssetBaseCfg = FRIDGE_STOCKING_ASSETS["prep_knives"]


@configclass
class FridgeStockingRewardsCfg:
    sequential_progress = RewTerm(
        func=mdp.FridgeStockingSequentialReward(
            sequence=FRIDGE_STOCKING_SEQUENCE,
            distance_shaping_scale=0.05,
        ),
        weight=1.0,
        params={},
    )
    action_reg = RewTerm(func=mdp.action_l2, weight=-1.0e-4)

    def __post_init__(self) -> None:
        self.disable_debug_logging()

    @property
    def debug_enabled(self) -> bool:
        return bool(getattr(self.sequential_progress.func, "debug", False))

    def enable_debug_logging(self) -> None:
        self.sequential_progress.func.debug = True

    def disable_debug_logging(self) -> None:
        self.sequential_progress.func.debug = False


@configclass
class FridgeStockingTerminationsCfg(XLeRobotTerminationsCfg):
    success = DoneTerm(
        func=mdp.fridge_stocking_completed,
        params={
            "object_cfg": FRUIT_CFG,
            "fridge_cfg": FRIDGE_CFG,
            "robot_cfg": ROBOT_CFG,
            "left_ee_frame_cfg": LEFT_EE_CFG,
            "right_ee_frame_cfg": RIGHT_EE_CFG,
            "target_position": STORAGE_PLATE_TARGET_POSITION,
            "close_threshold": 0.1,
            "xy_threshold": 0.18,
            "z_threshold": 0.06,
        },
    )


@configclass
class FridgeStockingEnvCfg(XLeRobotTaskEnvCfg):
    scene = FridgeStockingSceneCfg(env_spacing=8.0)
    observations = FridgeStockingObservationsCfg()
    terminations = FridgeStockingTerminationsCfg()
    rewards = FridgeStockingRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (3.2, -2.4, 1.6)
        self.viewer.lookat = (4.4, -2.4, 1.0)
        self.scene.robot.init_state.pos = (4.3, -1.8, 0.0)

        # Ensure sequential reward state starts from the initial stage for fresh episodes.
        self.rewards.sequential_progress.func.reset()

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

        domain_randomization(
            self,
            random_options=[
                randomize_object_uniform(
                    "fruit_bundle",
                    pose_range={
                        "x": (-0.075, 0.075),
                        "y": (-0.075, 0.075),
                        "z": (0.0, 0.0),
                        "yaw": (-30 * torch.pi / 180, 30 * torch.pi / 180),
                    },
                ),
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
                ),
            ],
        )

    def enable_debug_logging(self) -> None:
        """Enable reward debug mode for teleoperation."""

        self.rewards.enable_debug_logging()

    def disable_debug_logging(self) -> None:
        """Disable reward debug mode."""

        self.rewards.disable_debug_logging()


@configclass
class FridgeStockingMimicEnvCfg(FridgeStockingEnvCfg, MimicEnvCfg):
    """Mimic configuration for the fridge stocking task with XLEROBOT."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.datagen_config.name = "fridge_stocking_xlerobot_task_v0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 30
        self.datagen_config.seed = 42

        self.subtask_configs["xlerobot"] = mdp.build_mimic_subtask_configs(
            FRIDGE_STOCKING_SEQUENCE,
        )
