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

from brain_sim_assets.props.kitchen.scene_sets.fruit_display import bsFruitDisplayEntitiesGenerator


@configclass
class FruitDisplaySceneCfg(BiArmTaskSceneCfg):

    scene: AssetBaseCfg = TEST_WITH_CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )
    kitchen_background: AssetBaseCfg = bsFruitDisplayEntitiesGenerator.get_fruit_display_entities()["kitchen_background"]

    display_shelf: AssetBaseCfg = bsFruitDisplayEntitiesGenerator.get_fruit_display_entities()["display_shelf"]
    display_table: AssetBaseCfg = bsFruitDisplayEntitiesGenerator.get_fruit_display_entities()["display_table"]
    display_plate: AssetBaseCfg = bsFruitDisplayEntitiesGenerator.get_fruit_display_entities()["display_plate"]
    decor_bottle: AssetBaseCfg = bsFruitDisplayEntitiesGenerator.get_fruit_display_entities()["decor_bottle"]
    orange_a: AssetBaseCfg = bsFruitDisplayEntitiesGenerator.get_fruit_display_entities()["orange_a"]
    orange_b: AssetBaseCfg = bsFruitDisplayEntitiesGenerator.get_fruit_display_entities()["orange_b"]
    display_knives: AssetBaseCfg = bsFruitDisplayEntitiesGenerator.get_fruit_display_entities()["display_knives"]

    def __post_init__(self):
        super().__post_init__()
        delete_attribute(self, "front")


@configclass
class FruitDisplayObservationsCfg(BiArmObservationsCfg):

    def __post_init__(self):
        super().__post_init__()
        delete_attribute(self.policy, "wrist")


@configclass
class FruitDisplayTerminationsCfg(BiArmTerminationsCfg):

    success = DoneTerm(func=mdp.cube_height_above_base, params={
        "cube_cfg": SceneEntityCfg("cube"),
        "robot_cfg": SceneEntityCfg("left_arm"),
        "robot_base_name": "base",
        "height_threshold": 0.20,
    })


@configclass
class FruitDisplayEnvCfg(BiArmTaskEnvCfg):

    scene: FruitDisplaySceneCfg = FruitDisplaySceneCfg(env_spacing=8.0)
    observations: FruitDisplayObservationsCfg = FruitDisplayObservationsCfg()
    terminations: FruitDisplayTerminationsCfg = FruitDisplayTerminationsCfg()

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
                    ["display_plate", "decor_bottle", "orange_a", "orange_b", "display_knives"],
                    position_noise=0.07,
                    scene_cfg=self.scene,
                    reference_name="display_table",
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
