import torch

from typing import Dict, List

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from leisaac.assets.scenes.test import TEST_WITH_CUBE_CFG, TEST_WITH_CUBE_USD_PATH
from leisaac.utils.general_assets import parse_usd_and_create_subassets
from leisaac.utils.domain_randomization import randomize_object_uniform, randomize_camera_uniform, domain_randomization
from leisaac.utils.env_utils import delete_attribute

from . import mdp
from ..template import SingleArmTaskSceneCfg, SingleArmTaskEnvCfg, SingleArmTerminationsCfg, SingleArmObservationsCfg

from brain_sim_assets.props.kitchen.scene_sets.microwaving import bsMicrowavingEntitiesGenerator


@configclass
class MicrowavingSceneCfg(SingleArmTaskSceneCfg):

    scene: AssetBaseCfg = TEST_WITH_CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    front: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.6, -0.75, 0.38), rot=(0.77337, 0.55078, -0.2374, -0.20537), convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=40.6,
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )

    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )
    kitchen_background: AssetBaseCfg = bsMicrowavingEntitiesGenerator.get_microwaving_entities()["kitchen_background"]

    microwave: AssetBaseCfg = bsMicrowavingEntitiesGenerator.get_microwaving_entities()["microwave"]
    serving_table: AssetBaseCfg = bsMicrowavingEntitiesGenerator.get_microwaving_entities()["serving_table"]
    heating_plate: AssetBaseCfg = bsMicrowavingEntitiesGenerator.get_microwaving_entities()["heating_plate"]
    drink_bottle: AssetBaseCfg = bsMicrowavingEntitiesGenerator.get_microwaving_entities()["drink_bottle"]
    snack_fruit: AssetBaseCfg = bsMicrowavingEntitiesGenerator.get_microwaving_entities()["snack_fruit"]
    prep_knives: AssetBaseCfg = bsMicrowavingEntitiesGenerator.get_microwaving_entities()["prep_knives"]

    def __post_init__(self):
        super().__post_init__()
        delete_attribute(self, "wrist")


@configclass
class MicrowavingObservationsCfg(SingleArmObservationsCfg):

    @configclass
    class SubtaskCfg(ObsGroup):
        pick_cube = ObsTerm(func=mdp.object_grasped, params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("cube"),
        })

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()

    def __post_init__(self):
        super().__post_init__()
        delete_attribute(self.policy, "wrist")


@configclass
class MicrowavingTerminationsCfg(SingleArmTerminationsCfg):

    success = DoneTerm(func=mdp.cube_height_above_base, params={
        "cube_cfg": SceneEntityCfg("cube"),
        "robot_cfg": SceneEntityCfg("robot"),
        "robot_base_name": "base",
        "height_threshold": 0.20,
    })


@configclass
class MicrowavingEnvCfg(SingleArmTaskEnvCfg):

    scene: MicrowavingSceneCfg = MicrowavingSceneCfg(env_spacing=8.0)
    observations: MicrowavingObservationsCfg = MicrowavingObservationsCfg()
    terminations: MicrowavingTerminationsCfg = MicrowavingTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (-0.4, -0.6, 0.5)
        self.viewer.lookat = (0.9, 0.0, -0.3)

        self.scene.robot.init_state.pos = (0.35, -0.64, 0.01)

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
                    "front",
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
