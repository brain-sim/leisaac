import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from leisaac.assets.scenes.test import TEST_WITH_CUBE_CFG, TEST_WITH_CUBE_USD_PATH
from leisaac.utils.general_assets import parse_usd_and_create_subassets
from leisaac.utils.domain_randomization import (
    randomize_object_uniform,
    randomize_camera_uniform,
    randomize_objects_permutation,
    domain_randomization,
)
from leisaac.utils.env_utils import delete_attribute

from . import mdp
from ..template import BiArmTaskEnvCfg, BiArmTaskSceneCfg, BiArmTerminationsCfg, BiArmObservationsCfg

from brain_sim_assets.props.kitchen.scene_sets.breakfast_setup import bsBreakfastSetupEntitiesGenerator


@configclass
class BreakfastSetupSceneCfg(BiArmTaskSceneCfg):

    scene: AssetBaseCfg = TEST_WITH_CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )
    kitchen_background: AssetBaseCfg = bsBreakfastSetupEntitiesGenerator.get_breakfast_setup_entities()["kitchen_background"]

    toaster: AssetBaseCfg = bsBreakfastSetupEntitiesGenerator.get_breakfast_setup_entities()["toaster"]
    breakfast_table: AssetBaseCfg = bsBreakfastSetupEntitiesGenerator.get_breakfast_setup_entities()["breakfast_table"]
    toast_plate: AssetBaseCfg = bsBreakfastSetupEntitiesGenerator.get_breakfast_setup_entities()["toast_plate"]
    juice_bottle: AssetBaseCfg = bsBreakfastSetupEntitiesGenerator.get_breakfast_setup_entities()["juice_bottle"]
    fruit_side: AssetBaseCfg = bsBreakfastSetupEntitiesGenerator.get_breakfast_setup_entities()["fruit_side"]
    butter_knives: AssetBaseCfg = bsBreakfastSetupEntitiesGenerator.get_breakfast_setup_entities()["butter_knives"]

    def __post_init__(self):
        super().__post_init__()
        delete_attribute(self, "front")


@configclass
class BreakfastSetupObservationsCfg(BiArmObservationsCfg):

    def __post_init__(self):
        super().__post_init__()
        delete_attribute(self.policy, "wrist")


@configclass
class BreakfastSetupTerminationsCfg(BiArmTerminationsCfg):

    success = DoneTerm(func=mdp.cube_height_above_base, params={
        "cube_cfg": SceneEntityCfg("cube"),
        "robot_cfg": SceneEntityCfg("left_arm"),
        "robot_base_name": "base",
        "height_threshold": 0.20,
    })


@configclass
class BreakfastSetupEnvCfg(BiArmTaskEnvCfg):

    scene: BreakfastSetupSceneCfg = BreakfastSetupSceneCfg(env_spacing=8.0)
    observations: BreakfastSetupObservationsCfg = BreakfastSetupObservationsCfg()
    terminations: BreakfastSetupTerminationsCfg = BreakfastSetupTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (-0.4, -0.6, 0.5)
        self.viewer.lookat = (0.9, 0.0, -0.3)

        self.scene.left_arm.init_state.pos = (0.35, -0.64, 0.01)
        self.scene.left_arm.init_state.rot = (0.0, 0.0, 0.0, 1.0)

        self.scene.right_arm.init_state.pos = (0.65, -0.64, 0.01)
        self.scene.right_arm.init_state.rot = (0.0, 0.0, 0.0, 1.0)

        parse_usd_and_create_subassets(TEST_WITH_CUBE_USD_PATH, self)

        domain_randomization(
            self,
            random_options=[
                randomize_objects_permutation(
                    ["toast_plate", "juice_bottle", "fruit_side", "butter_knives"],
                    position_noise=0.07,
                    scene_cfg=self.scene,
                    reference_name="breakfast_table",
                ),
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
