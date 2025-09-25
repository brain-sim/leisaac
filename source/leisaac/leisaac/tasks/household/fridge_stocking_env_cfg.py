import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from leisaac.assets.scenes.test import TEST_WITH_CUBE_CFG, TEST_WITH_CUBE_USD_PATH
from leisaac.utils.general_assets import parse_usd_and_create_subassets
from leisaac.utils.domain_randomization import randomize_object_uniform, randomize_camera_uniform, domain_randomization
from leisaac.utils.env_utils import delete_attribute

from . import mdp
from ..template import BiArmTaskEnvCfg, BiArmTaskSceneCfg, BiArmTerminationsCfg, BiArmObservationsCfg

from brain_sim_assets.props.kitchen.scene_sets.fridge_stocking import bsFridgeStockingEntitiesGenerator


STORAGE_PLATE_TARGET_POSITION = (1.75, -1.7, 0.95)


@configclass
class FridgeStockingSceneCfg(BiArmTaskSceneCfg):

    scene: AssetBaseCfg = TEST_WITH_CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )
    kitchen_background: AssetBaseCfg = bsFridgeStockingEntitiesGenerator.get_fridge_stocking_entities()["kitchen_background"]

    fridge: AssetBaseCfg = bsFridgeStockingEntitiesGenerator.get_fridge_stocking_entities()["fridge"]
    stock_table: AssetBaseCfg = bsFridgeStockingEntitiesGenerator.get_fridge_stocking_entities()["stock_table"]
    storage_plate: AssetBaseCfg = bsFridgeStockingEntitiesGenerator.get_fridge_stocking_entities()["storage_plate"]
    juice_bottle: AssetBaseCfg = bsFridgeStockingEntitiesGenerator.get_fridge_stocking_entities()["juice_bottle"]
    fruit_bundle: AssetBaseCfg = bsFridgeStockingEntitiesGenerator.get_fridge_stocking_entities()["fruit_bundle"]
    prep_knives: AssetBaseCfg = bsFridgeStockingEntitiesGenerator.get_fridge_stocking_entities()["prep_knives"]

    def __post_init__(self):
        super().__post_init__()
        delete_attribute(self, "front")


@configclass
class FridgeStockingObservationsCfg(BiArmObservationsCfg):

    @configclass
    class SubtaskCfg(ObsGroup):
        open_fridge = ObsTerm(func=mdp.fridge_door_opened, params={
            "fridge_cfg": SceneEntityCfg("fridge"),
        })
        place_object = ObsTerm(func=mdp.object_placed, params={
            "object_cfg": SceneEntityCfg("cube"),
            "target_cfg": SceneEntityCfg("storage_plate"),
            "target_position": STORAGE_PLATE_TARGET_POSITION,
            "xy_threshold": 0.18,
            "z_threshold": 0.06,
        })
        close_fridge = ObsTerm(func=mdp.fridge_door_closed, params={
            "fridge_cfg": SceneEntityCfg("fridge"),
        })

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()

    def __post_init__(self):
        super().__post_init__()
        delete_attribute(self.policy, "wrist")


@configclass
class FridgeStockingTerminationsCfg(BiArmTerminationsCfg):

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
class FridgeStockingEnvCfg(BiArmTaskEnvCfg):

    scene: FridgeStockingSceneCfg = FridgeStockingSceneCfg(env_spacing=8.0)
    observations: FridgeStockingObservationsCfg = FridgeStockingObservationsCfg()
    terminations: FridgeStockingTerminationsCfg = FridgeStockingTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (-0.4, -0.6, 0.5)
        self.viewer.lookat = (0.9, 0.0, -0.3)

        # Position the arms close to the table for the handoff task
        self.scene.left_arm.init_state.pos = (4.5, -3.2, 0.75)
        self.scene.left_arm.init_state.rot = (0.38268, 0.0, 0.0, -0.92388)

        self.scene.right_arm.init_state.pos = (4.688, -3.012, 0.75)
        self.scene.right_arm.init_state.rot = (0.38268, 0.0, 0.0, -0.92388)

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

        # TODO: Introduce subtask-specific reward shaping when porting this mimic setup to an RL task.


@configclass
class FridgeStockingMimicEnvCfg(FridgeStockingEnvCfg, MimicEnvCfg):
    """Mimic configuration for the fridge stocking bi-arm task."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.datagen_config.name = "fridge_stocking_biarm_task_v0"
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
                object_ref="fridge",
                subtask_term_signal="open_fridge",
                subtask_term_offset_range=(8, 16),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Open the fridge door",
                next_subtask_description="Place the object onto the storage plate",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube",
                subtask_term_signal="place_object",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.002,
                num_interpolation_steps=6,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Place the cube onto the storage plate",
                next_subtask_description="Close the fridge door",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="fridge",
                subtask_term_signal="close_fridge",
                subtask_term_offset_range=(8, 16),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.002,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Close the fridge door",
                next_subtask_description="Stabilize the robot",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref=None,
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="random",
                selection_strategy_kwargs={},
                action_noise=0.001,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )

        self.subtask_configs["so101_follower"] = subtask_configs
