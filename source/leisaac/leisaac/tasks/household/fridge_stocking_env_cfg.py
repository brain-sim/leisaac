import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from leisaac.assets.scenes.test import TEST_WITH_CUBE_CFG, TEST_WITH_CUBE_USD_PATH
from leisaac.utils.general_assets import parse_usd_and_create_subassets
from leisaac.utils.domain_randomization import (
    randomize_camera_uniform,
    randomize_object_uniform,
    domain_randomization,
)

from . import mdp
from ..template import (
    XLeRobotObservationsCfg,
    XLeRobotTaskEnvCfg,
    XLeRobotTaskSceneCfg,
    XLeRobotTerminationsCfg,
)

from brain_sim_assets.props.kitchen.scene_sets.fridge_stocking import bsFridgeStockingEntitiesGenerator


STORAGE_PLATE_TARGET_POSITION = (1.75, -1.7, 0.95)


@configclass
class FridgeStockingSceneCfg(XLeRobotTaskSceneCfg):

    scene: AssetBaseCfg = TEST_WITH_CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=1500.0),
    )

    kitchen_background: AssetBaseCfg = bsFridgeStockingEntitiesGenerator.get_fridge_stocking_entities()["kitchen_background"]
    fridge: AssetBaseCfg = bsFridgeStockingEntitiesGenerator.get_fridge_stocking_entities()["fridge"]
    stock_table: AssetBaseCfg = bsFridgeStockingEntitiesGenerator.get_fridge_stocking_entities()["stock_table"]
    storage_plate: AssetBaseCfg = bsFridgeStockingEntitiesGenerator.get_fridge_stocking_entities()["storage_plate"]
    juice_bottle: AssetBaseCfg = bsFridgeStockingEntitiesGenerator.get_fridge_stocking_entities()["juice_bottle"]
    fruit_bundle: AssetBaseCfg = bsFridgeStockingEntitiesGenerator.get_fridge_stocking_entities()["fruit_bundle"]
    prep_knives: AssetBaseCfg = bsFridgeStockingEntitiesGenerator.get_fridge_stocking_entities()["prep_knives"]


@configclass
class FridgeStockingObservationsCfg(XLeRobotObservationsCfg):

    @configclass
    class SubtaskCfg(ObsGroup):
        approach_object = ObsTerm(func=mdp.robot_close_to_object, params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("cube"),
            "distance_threshold": 0.7,
        })
        grasp_object = ObsTerm(func=mdp.object_grasped, params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
            "object_cfg": SceneEntityCfg("cube"),
        })
        reach_fridge = ObsTerm(func=mdp.robot_close_to_target, params={
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("fridge"),
            "distance_threshold": 0.9,
        })
        door_open = ObsTerm(func=mdp.fridge_door_opened, params={
            "fridge_cfg": SceneEntityCfg("fridge"),
        })
        object_placed = ObsTerm(func=mdp.object_placed, params={
            "object_cfg": SceneEntityCfg("cube"),
            "target_cfg": SceneEntityCfg("storage_plate"),
            "target_position": STORAGE_PLATE_TARGET_POSITION,
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
            "xy_threshold": 0.18,
            "z_threshold": 0.06,
        })
        door_closed = ObsTerm(func=mdp.fridge_door_closed, params={
            "fridge_cfg": SceneEntityCfg("fridge"),
        })

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class FridgeStockingTerminationsCfg(XLeRobotTerminationsCfg):

    success = DoneTerm(func=mdp.fridge_stocking_completed, params={
        "object_cfg": SceneEntityCfg("cube"),
        "target_cfg": SceneEntityCfg("storage_plate"),
        "fridge_cfg": SceneEntityCfg("fridge"),
        "target_position": STORAGE_PLATE_TARGET_POSITION,
        "close_threshold": 0.1,
        "xy_threshold": 0.18,
        "z_threshold": 0.06,
    })


@configclass
class FridgeStockingRewardsCfg:

    sequential_progress = RewTerm(
        func=mdp.FridgeStockingSequentialReward(
            stage_rewards=[2.0, 3.0, 2.0, 2.5, 3.5, 2.5],
            approach_distance_threshold=0.65,
            fridge_distance_threshold=0.9,
            distance_shaping_scale=0.05,
        ),
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("cube"),
            "fridge_cfg": SceneEntityCfg("fridge"),
            "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
            "target_cfg": SceneEntityCfg("storage_plate"),
            "target_position": STORAGE_PLATE_TARGET_POSITION,
            "xy_threshold": 0.18,
            "z_threshold": 0.06,
        },
    )
    action_reg = RewTerm(func=mdp.action_l2, weight=-1.0e-4)


@configclass
class FridgeStockingEnvCfg(XLeRobotTaskEnvCfg):

    scene: FridgeStockingSceneCfg = FridgeStockingSceneCfg(env_spacing=8.0)
    observations: FridgeStockingObservationsCfg = FridgeStockingObservationsCfg()
    terminations: FridgeStockingTerminationsCfg = FridgeStockingTerminationsCfg()
    rewards: FridgeStockingRewardsCfg = FridgeStockingRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (3.2, -2.4, 1.6)
        self.viewer.lookat = (4.4, -2.4, 1.0)

        # Configure actions for base, manipulators, and optional head pan.
        self.actions.base_motion_action = mdp.JointVelocityActionCfg(
            asset_name="robot",
            joint_names=["axle_0_joint", "axle_1_joint", "axle_2_joint"],
            scale=1.0,
        )
        self.actions.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["Rotation_2", "Pitch_2", "Elbow_2", "Wrist_Pitch_2", "Wrist_Roll_2", "Jaw_2"],
            scale=1.0,
        )
        self.actions.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
            scale=1.0,
        )
        self.actions.head_pan_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["head_pan_joint"],
            scale=0.4,
        )

        parse_usd_and_create_subassets(TEST_WITH_CUBE_USD_PATH, self)

        domain_randomization(
            self,
            random_options=[
                randomize_object_uniform(
                    "cube",
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

        subtask_configs: list[SubTaskConfig] = []

        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube",
                subtask_term_signal="approach_object",
                subtask_term_offset_range=(6, 12),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.002,
                num_interpolation_steps=4,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Approach the orange on the table",
                next_subtask_description="Grasp the orange",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube",
                subtask_term_signal="grasp_object",
                subtask_term_offset_range=(6, 12),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.002,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp the orange with the right gripper",
                next_subtask_description="Drive to the fridge while holding the orange",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="fridge",
                subtask_term_signal="reach_fridge",
                subtask_term_offset_range=(8, 16),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.002,
                num_interpolation_steps=6,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Navigate to the fridge with the object grasped",
                next_subtask_description="Open the fridge door",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="fridge",
                subtask_term_signal="door_open",
                subtask_term_offset_range=(6, 12),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.002,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Open the fridge door",
                next_subtask_description="Place the orange onto the storage plate",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube",
                subtask_term_signal="object_placed",
                subtask_term_offset_range=(8, 16),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.002,
                num_interpolation_steps=6,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Place the orange onto the storage plate",
                next_subtask_description="Close the fridge door",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="fridge",
                subtask_term_signal="door_closed",
                subtask_term_offset_range=(6, 12),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.002,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Close the fridge door",
            )
        )

        self.subtask_configs["xlerobot"] = subtask_configs
