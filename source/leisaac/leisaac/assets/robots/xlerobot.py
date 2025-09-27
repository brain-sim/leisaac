from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from leisaac.utils.constant import ASSETS_ROOT


"""Configuration for the XLEROBOT dual-arm mobile robot."""
XLEROBOT_ASSET_PATH = Path(ASSETS_ROOT) / "robots" / "xlerobot.usd"

XLEROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(XLEROBOT_ASSET_PATH),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(2.2, -0.61, 0.89),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "shoulder_pan_l": 0.0,
            "shoulder_lift_l": 0.0,
            "elbow_flex_l": 0.0,
            "wrist_flex_l": 0.0,
            "wrist_roll_l": 0.0,
            "gripper_l": 0.0,
            "shoulder_pan_r": 0.0,
            "shoulder_lift_r": 0.0,
            "elbow_flex_r": 0.0,
            "wrist_flex_r": 0.0,
            "wrist_roll_r": 0.0,
            "gripper_r": 0.0,
            "axle_0_joint": 0.0,
            "axle_1_joint": 0.0,
            "axle_2_joint": 0.0,
        }
    ),
    actuators={
        "sts3215-gripper-l": ImplicitActuatorCfg(
            joint_names_expr=["gripper_l"],
            effort_limit_sim=10,
            velocity_limit_sim=10,
            stiffness=17.8,
            damping=0.60,
        ),
        "sts3215-arm-l": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_l",
                "shoulder_lift_l",
                "elbow_flex_l",
                "wrist_flex_l",
                "wrist_roll_l",
            ],
            effort_limit_sim=10,
            velocity_limit_sim=10,
            stiffness=17.8,
            damping=0.60,
        ),
        "sts3215-gripper-r": ImplicitActuatorCfg(
            joint_names_expr=["gripper_r"],
            effort_limit_sim=10,
            velocity_limit_sim=10,
            stiffness=17.8,
            damping=0.60,
        ),
        "sts3215-arm-r": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_r",
                "shoulder_lift_r",
                "elbow_flex_r",
                "wrist_flex_r",
                "wrist_roll_r",
            ],
            effort_limit_sim=10,
            velocity_limit_sim=10,
            stiffness=17.8,
            damping=0.60,
        ),
        "omni-wheel-drive": ImplicitActuatorCfg(
            joint_names_expr=["axle_0_joint", "axle_1_joint", "axle_2_joint"],
            effort_limit_sim=10,
            velocity_limit_sim=10,
            stiffness=17.8,
            damping=0.60,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# joint limit written in USD (degree)
XLEROBOT_USD_JOINT_LIMLITS = {
    "shoulder_pan_l": (-110.0, 110.0),
    "shoulder_lift_l": (-100.0, 100.0),
    "elbow_flex_l": (-100.0, 90.0),
    "wrist_flex_l": (-95.0, 95.0),
    "wrist_roll_l": (-160.0, 160.0),
    "gripper_l": (-10.0, 100.0),
    "shoulder_pan_r": (-110.0, 110.0),
    "shoulder_lift_r": (-100.0, 100.0),
    "elbow_flex_r": (-100.0, 90.0),
    "wrist_flex_r": (-95.0, 95.0),
    "wrist_roll_r": (-160.0, 160.0),
    "gripper_r": (-10.0, 100.0),
    "axle_0_joint": (-360.0, 360.0),
    "axle_1_joint": (-360.0, 360.0),
    "axle_2_joint": (-360.0, 360.0),
}

# motor limit written in real device (normalized to related range)
XLEROBOT_MOTOR_LIMITS = {
    "shoulder_pan_l": (-100.0, 100.0),
    "shoulder_lift_l": (-100.0, 100.0),
    "elbow_flex_l": (-100.0, 100.0),
    "wrist_flex_l": (-100.0, 100.0),
    "wrist_roll_l": (-100.0, 100.0),
    "gripper_l": (0.0, 100.0),
    "shoulder_pan_r": (-100.0, 100.0),
    "shoulder_lift_r": (-100.0, 100.0),
    "elbow_flex_r": (-100.0, 100.0),
    "wrist_flex_r": (-100.0, 100.0),
    "wrist_roll_r": (-100.0, 100.0),
    "gripper_r": (0.0, 100.0),
    "axle_0_joint": (-100.0, 100.0),
    "axle_1_joint": (-100.0, 100.0),
    "axle_2_joint": (-100.0, 100.0),
}


XLEROBOT_REST_POSE_RANGE = {
    "shoulder_pan_l": (-30.0, 30.0),  # 0 degree
    "shoulder_lift_l": (-130.0, -70.0),  # -100 degree
    "elbow_flex_l": (60.0, 120.0),  # 90 degree
    "wrist_flex_l": (20.0, 80.0),  # 50 degree
    "wrist_roll_l": (-30.0, 30.0),  # 0 degree
    "gripper_l": (-40.0, 20.0),  # -10 degree
    "shoulder_pan_r": (-30.0, 30.0),  # 0 degree
    "shoulder_lift_r": (-130.0, -70.0),  # -100 degree
    "elbow_flex_r": (60.0, 120.0),  # 90 degree
    "wrist_flex_r": (20.0, 80.0),  # 50 degree
    "wrist_roll_r": (-30.0, 30.0),  # 0 degree
    "gripper_r": (-40.0, 20.0),  # -10 degree
    "axle_0_joint": (-5.0, 5.0),  # centered wheel
    "axle_1_joint": (-5.0, 5.0),  # centered wheel
    "axle_2_joint": (-5.0, 5.0),  # centered wheel
}
