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

from leisaac.assets.scenes.simple import TABLE_WITH_CUBE_CFG, TABLE_WITH_CUBE_USD_PATH
from leisaac.utils.general_assets import parse_usd_and_create_subassets
from leisaac.utils.domain_randomization import randomize_object_uniform, domain_randomization
from leisaac.utils.env_utils import delete_attribute
from leisaac.enhance.envs.manager_based_rl_digital_twin_env_cfg import ManagerBasedRLDigitalTwinEnvCfg
from .lift_cube_env_cfg import LiftCubeSceneCfg, ObservationsCfg, TerminationsCfg
from ..template import BiArmTaskSceneCfg, BiArmTaskEnvCfg, BiArmTerminationsCfg, BiArmObservationsCfg
from ..template import mdp as template_mdp


@configclass
class LiftCubeBiArmHandoffSceneCfg(BiArmTaskSceneCfg):
    """Scene configuration for the bi-arm lift cube handoff task."""

    scene: AssetBaseCfg = TABLE_WITH_CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )

    head: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Left_Robot/base/head_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.4137, 0.0316, -0.4266),
            rot=(0.24184, -0.66446, 0.66446, 0.24184),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )

    def __post_init__(self):
        super().__post_init__()
        delete_attribute(self, "front")
        delete_attribute(self, "top")


@configclass
class LiftCubeBiArmHandoffObservationsCfg(BiArmObservationsCfg):
    """Observations for the bi-arm lift cube handoff task."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations using wrist and head cameras."""

        left_joint_pos = ObsTerm(func=template_mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("left_arm")})
        left_joint_vel = ObsTerm(func=template_mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("left_arm")})
        left_joint_pos_rel = ObsTerm(func=template_mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("left_arm")})
        left_joint_vel_rel = ObsTerm(func=template_mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("left_arm")})
        left_joint_pos_target = ObsTerm(func=template_mdp.joint_pos_target, params={"asset_cfg": SceneEntityCfg("left_arm")})

        right_joint_pos = ObsTerm(func=template_mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("right_arm")})
        right_joint_vel = ObsTerm(func=template_mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("right_arm")})
        right_joint_pos_rel = ObsTerm(func=template_mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("right_arm")})
        right_joint_vel_rel = ObsTerm(func=template_mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("right_arm")})
        right_joint_pos_target = ObsTerm(func=template_mdp.joint_pos_target, params={"asset_cfg": SceneEntityCfg("right_arm")})

        actions = ObsTerm(func=template_mdp.last_action)
        left = ObsTerm(
            func=template_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("left_wrist"), "data_type": "rgb", "normalize": False},
        )
        right = ObsTerm(
            func=template_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("right_wrist"), "data_type": "rgb", "normalize": False},
        )
        head = ObsTerm(
            func=template_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("head"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""
        
        # Simple placeholder observation for now
        dummy_obs = ObsTerm(func=lambda env: torch.zeros(env.num_envs, 1, device=env.device))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()

    def __post_init__(self):
        super().__post_init__()


@configclass
class LiftCubeBiArmHandoffTerminationsCfg(BiArmTerminationsCfg):
    """Terminations for the bi-arm lift cube handoff task."""
    
    # Use a simple time limit for now
    time_out = DoneTerm(func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))


@configclass
class LiftCubeBiArmHandoffEnvCfg(BiArmTaskEnvCfg):
    """Configuration for the bi-arm lift cube handoff environment."""

    scene: LiftCubeBiArmHandoffSceneCfg = LiftCubeBiArmHandoffSceneCfg(env_spacing=8.0)
    observations: LiftCubeBiArmHandoffObservationsCfg = LiftCubeBiArmHandoffObservationsCfg()
    terminations: LiftCubeBiArmHandoffTerminationsCfg = LiftCubeBiArmHandoffTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # Adjust viewer position for bi-arm setup
        self.viewer.eye = (0.9, -1.6, 1.3)
        self.viewer.lookat = (0.5, -0.6, 0.4)

        # Position the arms close to the table for the handoff task
        self.scene.left_arm.init_state.pos = (0.15, -0.65, 0.01)
        self.scene.left_arm.init_state.rot = (0.0, 0.0, 0.0, 1.0)

        self.scene.right_arm.init_state.pos = (0.65, -0.65, 0.01)
        self.scene.right_arm.init_state.rot = (0.0, 0.0, 0.0, 1.0)

        parse_usd_and_create_subassets(TABLE_WITH_CUBE_USD_PATH, self)

        # Simple domain randomization
        domain_randomization(self, random_options=[
            randomize_object_uniform("cube", pose_range={
                "x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0),
                "yaw": (-15 * torch.pi / 180, 15 * torch.pi / 180)}),
        ])


# @configclass
# class LiftCubeBiArmHandoffDigitalTwinEnvCfg(LiftCubeBiArmHandoffEnvCfg, ManagerBasedRLDigitalTwinEnvCfg):
#     """Configuration for the bi-arm lift cube handoff digital twin environment."""

#     rgb_overlay_mode: str = "background"

#     rgb_overlay_paths: Dict[str, str] = {
#         "front": "greenscreen/background-lift-cube-handoff.png"
#     }

#     render_objects: List[SceneEntityCfg] = [
#         SceneEntityCfg("cube"),
#         SceneEntityCfg("left_arm"),
#         SceneEntityCfg("right_arm"),
#     ]
