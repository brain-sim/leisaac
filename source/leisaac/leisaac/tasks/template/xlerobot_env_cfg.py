from __future__ import annotations

from dataclasses import MISSING
from typing import Any

from termcolor import colored

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg as RecordTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg, FrameTransformerCfg, OffsetCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from leisaac.assets.robots.xlerobot import XLEROBOT_CFG
from leisaac.devices.action_process import init_action_cfg, preprocess_device_action

from . import mdp


@configclass
class XLeRobotTaskSceneCfg(InteractiveSceneCfg):
    """Scene configuration for tasks with the XLEROBOT platform."""

    scene: AssetBaseCfg = MISSING

    robot: ArticulationCfg = XLEROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/XLeRobot")

    right_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/XLeRobot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/XLeRobot/Fixed_Jaw",
                name="right_gripper",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
        ],
    )

    left_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/XLeRobot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/XLeRobot/Fixed_Jaw_2",
                name="left_gripper",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
        ],
    )

    left_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/XLeRobot/Fixed_Jaw_2/Left_Arm_Camera/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), convention="usd"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=35.0,
            focus_distance=200.0,
            horizontal_aperture=36.0,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )

    right_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/XLeRobot/Fixed_Jaw/Right_Arm_Camera/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), convention="usd"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=35.0,
            focus_distance=200.0,
            horizontal_aperture=36.0,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )

    top: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/XLeRobot/head_tilt_link/Camera/top_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.966, -0.259, 0.0, 0.0), convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.0,
            focus_distance=400.0,
            horizontal_aperture=40.0,
            clipping_range=(0.01, 60.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )

    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


@configclass
class XLeRobotActionsCfg:
    """Action configuration placeholders."""

    left_arm_action: mdp.ActionTermCfg = MISSING
    right_arm_action: mdp.ActionTermCfg = MISSING
    base_motion_action: mdp.ActionTermCfg = MISSING
    head_pan_action: mdp.ActionTermCfg = MISSING


@configclass
class XLeRobotEventCfg:
    """Configuration for events in XLEROBOT tasks."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class XLeRobotObservationsCfg:
    """Observation specifications for the XLEROBOT MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observation group."""

        base_root_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        base_root_quat = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        base_root_lin_vel = ObsTerm(func=mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("robot")})
        base_root_ang_vel = ObsTerm(func=mdp.root_ang_vel_w, params={"asset_cfg": SceneEntityCfg("robot")})

        left_joint_pos = ObsTerm(func=mdp.joint_pos, params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Rotation_2", "Pitch_2", "Elbow_2", "Wrist_Pitch_2", "Wrist_Roll_2", "Jaw_2"])
        })
        left_joint_vel = ObsTerm(func=mdp.joint_vel, params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Rotation_2", "Pitch_2", "Elbow_2", "Wrist_Pitch_2", "Wrist_Roll_2", "Jaw_2"])
        })
        right_joint_pos = ObsTerm(func=mdp.joint_pos, params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"])
        })
        right_joint_vel = ObsTerm(func=mdp.joint_vel, params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"])
        })
        head_pan_pos = ObsTerm(func=mdp.joint_pos, params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["head_pan_joint"])
        })
        head_pan_vel = ObsTerm(func=mdp.joint_vel, params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["head_pan_joint"])
        })
        wheel_joint_vel = ObsTerm(func=mdp.joint_vel, params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["axle_0_joint", "axle_1_joint", "axle_2_joint"])
        })
        left_ee_state = ObsTerm(func=mdp.ee_frame_state, params={
            "ee_frame_cfg": SceneEntityCfg("left_ee_frame"),
            "robot_cfg": SceneEntityCfg("robot"),
        })
        right_ee_state = ObsTerm(func=mdp.ee_frame_state, params={
            "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
            "robot_cfg": SceneEntityCfg("robot"),
        })
        

        actions = ObsTerm(func=mdp.last_action)
        left_rgb = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("left_wrist"), "data_type": "rgb", "normalize": False})
        right_rgb = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("right_wrist"), "data_type": "rgb", "normalize": False})
        top_rgb = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("top"), "data_type": "rgb", "normalize": False})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class XLeRobotRewardsCfg:
    """Placeholder for rewards configuration."""


@configclass
class XLeRobotTerminationsCfg:
    """Default termination configuration."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class XLeRobotTaskEnvCfg(ManagerBasedRLEnvCfg):
    """Base manager-based RL environment configuration for XLEROBOT tasks."""

    scene: XLeRobotTaskSceneCfg = MISSING
    observations: XLeRobotObservationsCfg = MISSING
    actions: XLeRobotActionsCfg = XLeRobotActionsCfg()
    events: XLeRobotEventCfg = XLeRobotEventCfg()
    rewards: XLeRobotRewardsCfg = XLeRobotRewardsCfg()
    terminations: XLeRobotTerminationsCfg = MISSING

    recorders: RecordTerm = RecordTerm()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.decimation = 1
        self.episode_length_s = 15.0
        self.viewer.eye = (3.5, -1.5, 2.0)
        self.viewer.lookat = (0.0, 0.0, 1.2)

        self.sim.physx.bounce_threshold_velocity = 0.1
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.render.enable_translucency = True

    def use_teleop_device(self, teleop_device) -> None:
        self.task_type = teleop_device
        self.actions = init_action_cfg(self.actions, device=teleop_device)

    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> Any:
        return preprocess_device_action(action, teleop_device)
