from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from leisaac.utils.constant import ASSETS_ROOT


XLEROBOT_ASSET_PATH = Path(ASSETS_ROOT) / "robots" / "xlerobot.usd"


XLEROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(XLEROBOT_ASSET_PATH),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(5.0, -3.9, 0.01),
        rot=(0.17365, 0, 0, -0.98481),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            ".*": 0.0,
        },
    ),
    actuators={
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["Rotation_2", "Pitch_2", "Elbow_2", "Wrist_Pitch_2", "Wrist_Roll_2", "Jaw_2"],
            effort_limit_sim=10,
            velocity_limit_sim=10,
            stiffness=17.8,
            damping=0.60,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
            effort_limit_sim=10,
            velocity_limit_sim=10,
            stiffness=17.8,
            damping=0.60,
        ),
        "base_wheels": ImplicitActuatorCfg(
            joint_names_expr=["axle_0_joint", "axle_1_joint", "axle_2_joint"],
            damping=None,
            stiffness=None,
            velocity_limit_sim=50.0,
        ),
        "head_pan": ImplicitActuatorCfg(
            joint_names_expr=["head_pan_joint"],
            stiffness=15.0,
            damping=1.0,
            effort_limit_sim=10.0,
            velocity_limit_sim=5.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
