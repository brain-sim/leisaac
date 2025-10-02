import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from leisaac.assets.scenes.empty import EMPTY_CFG, EMPTY_USD_PATH
from leisaac.utils.general_assets import parse_usd_and_create_subassets
from leisaac.utils.domain_randomization import (
    randomize_object_uniform,
    randomize_camera_uniform,
    randomize_objects_permutation,
    domain_randomization,
)

from . import mdp
from ..template import (
    XLeRobotObservationsCfg,
    XLeRobotTaskEnvCfg,
    XLeRobotTaskSceneCfg,
    XLeRobotTerminationsCfg,
)

from brain_sim_assets.props.kitchen.scene_sets.dishwasher_restock import bsDishwasherRestockEntitiesGenerator


@configclass
class DishwasherRestockSceneCfg(XLeRobotTaskSceneCfg):

    scene: AssetBaseCfg = EMPTY_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )
    kitchen_background: AssetBaseCfg = bsDishwasherRestockEntitiesGenerator.get_dishwasher_restock_entities()["kitchen_background"]

    dishwasher: AssetBaseCfg = bsDishwasherRestockEntitiesGenerator.get_dishwasher_restock_entities()["dishwasher"]
    loading_table: AssetBaseCfg = bsDishwasherRestockEntitiesGenerator.get_dishwasher_restock_entities()["loading_table"]
    dish_plate: AssetBaseCfg = bsDishwasherRestockEntitiesGenerator.get_dishwasher_restock_entities()["dish_plate"]
    soap_bottle: AssetBaseCfg = bsDishwasherRestockEntitiesGenerator.get_dishwasher_restock_entities()["soap_bottle"]
    produce_item: AssetBaseCfg = bsDishwasherRestockEntitiesGenerator.get_dishwasher_restock_entities()["produce_item"]
    counter_knives: AssetBaseCfg = bsDishwasherRestockEntitiesGenerator.get_dishwasher_restock_entities()["counter_knives"]

@configclass
class DishwasherRestockEnvCfg(XLeRobotTaskEnvCfg):

    scene: DishwasherRestockSceneCfg = DishwasherRestockSceneCfg(env_spacing=8.0)
    observations: XLeRobotObservationsCfg = XLeRobotObservationsCfg()
    terminations: XLeRobotTerminationsCfg = XLeRobotTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (3.2, -2.4, 1.6)
        self.viewer.lookat = (4.4, -2.4, 1.0)
        self.scene.robot.init_state.pos = (5.0, -1.8, 0.0)

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

        parse_usd_and_create_subassets(EMPTY_USD_PATH, self)

        domain_randomization(
            self,
            random_options=[
                randomize_objects_permutation(
                    ["dish_plate", "soap_bottle", "produce_item", "counter_knives"],
                    position_noise=0.07,
                    scene_cfg=self.scene,
                    reference_name="loading_table",
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
