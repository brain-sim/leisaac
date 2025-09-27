import torch
from typing import Any

import isaaclab.envs.mdp as mdp

from leisaac.assets.robots.lerobot import SO101_FOLLOWER_USD_JOINT_LIMLITS


def init_action_cfg(action_cfg, device):
    if device in ['so101leader']:
        action_cfg.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=1.0,
        )
        action_cfg.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale=1.0,
        )
    elif device in ['keyboard']:
        action_cfg.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=1.0,
        )
        action_cfg.gripper_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale=0.7,
        )
    elif device in ['bi-so101leader']:
        action_cfg.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="left_arm",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=1.0,
        )
        action_cfg.left_gripper_action = mdp.JointPositionActionCfg(
            asset_name="left_arm",
            joint_names=["gripper"],
            scale=1.0,
        )
        action_cfg.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="right_arm",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=1.0,
        )
        action_cfg.right_gripper_action = mdp.JointPositionActionCfg(
            asset_name="right_arm",
            joint_names=["gripper"],
            scale=1.0,
        )
    elif device in ['mimic_so101leader']:
        action_cfg.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            body_name="gripper",
            controller=mdp.DifferentialIKControllerCfg(command_type="pose", ik_method="dls", use_relative_mode=False),
        )
        action_cfg.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale=1.0,
        )
    elif device in ['mimic_keyboard']:
        action_cfg.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            body_name="gripper",
            controller=mdp.DifferentialIKControllerCfg(command_type="pose", ik_method="dls", use_relative_mode=False),
        )
        action_cfg.gripper_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale=0.7,
        )
    elif device in ['xlerobot', 'xlerobot_keyboard']:
        action_cfg.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
            scale=1.0,
        )
        action_cfg.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["Rotation_2", "Pitch_2", "Elbow_2", "Wrist_Pitch_2", "Wrist_Roll_2", "Jaw_2"],
            scale=1.0,
        )
        action_cfg.base_motion_action = mdp.JointVelocityActionCfg(
            asset_name="robot",
            joint_names=["axle_0_joint", "axle_1_joint", "axle_2_joint"],
            scale=20.0,
        )
        action_cfg.head_pan_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["head_pan_joint"],
            scale=0.4,
        )
    else:
        action_cfg.arm_action = None
        action_cfg.gripper_action = None
    return action_cfg


joint_names_to_motor_ids = {
    "shoulder_pan": 0,
    "shoulder_lift": 1,
    "elbow_flex": 2,
    "wrist_flex": 3,
    "wrist_roll": 4,
    "gripper": 5,
}


def convert_action_from_so101_leader(joint_state: dict[str, float], motor_limits: dict[str, tuple[float, float]], teleop_device) -> torch.Tensor:
    processed_action = torch.zeros(teleop_device.env.num_envs, 6, device=teleop_device.env.device)
    joint_limits = SO101_FOLLOWER_USD_JOINT_LIMLITS
    for joint_name, motor_id in joint_names_to_motor_ids.items():
        motor_limit_range = motor_limits[joint_name]
        joint_limit_range = joint_limits[joint_name]
        processed_degree = (joint_state[joint_name] - motor_limit_range[0]) / (motor_limit_range[1] - motor_limit_range[0]) \
            * (joint_limit_range[1] - joint_limit_range[0]) + joint_limit_range[0]
        processed_radius = processed_degree / 180.0 * torch.pi  # convert degree to radius
        processed_action[:, motor_id] = processed_radius
    return processed_action


def preprocess_device_action(action: dict[str, Any], teleop_device) -> torch.Tensor:
    if action.get('so101_leader') is not None:
        processed_action = convert_action_from_so101_leader(action['joint_state'], action['motor_limits'], teleop_device)
    elif action.get('keyboard') is not None:
        processed_action = torch.zeros(teleop_device.env.num_envs, 6, device=teleop_device.env.device)
        processed_action[:, :] = action['joint_state']
    elif action.get('bi_so101_leader') is not None:
        processed_action = torch.zeros(teleop_device.env.num_envs, 12, device=teleop_device.env.device)
        processed_action[:, :6] = convert_action_from_so101_leader(action['joint_state']['left_arm'], action['motor_limits']['left_arm'], teleop_device)
        processed_action[:, 6:] = convert_action_from_so101_leader(action['joint_state']['right_arm'], action['motor_limits']['right_arm'], teleop_device)
    elif action.get('xlerobot') is not None:
        joint_state = action.get('joint_state', {})
        device = teleop_device.env.device
        num_envs = teleop_device.env.num_envs

        processed_action = torch.zeros(num_envs, 16, device=device)

        def _to_tensor(values, expected_len):
            if isinstance(values, torch.Tensor):
                return values.to(device=device, dtype=torch.float32)
            return torch.tensor(values, dtype=torch.float32, device=device).view(-1)[:expected_len]

        base_cmd = joint_state.get('base', [0.0, 0.0, 0.0])
        right_cmd = joint_state.get('right_arm', [0.0] * 6)
        left_cmd = joint_state.get('left_arm', [0.0] * 6)
        head_cmd = joint_state.get('head_pan', [0.0]) if isinstance(joint_state.get('head_pan'), (list, tuple, torch.Tensor)) else [joint_state.get('head_pan', 0.0)]

        base_tensor = _to_tensor(base_cmd, 3)

        if isinstance(right_cmd, dict):
            right_tensor = convert_action_from_so101_leader(right_cmd, action['motor_limits']['right_arm'], teleop_device)
        else:
            right_tensor = _to_tensor(right_cmd, 6).view(1, -1)

        if isinstance(left_cmd, dict):
            left_tensor = convert_action_from_so101_leader(left_cmd, action['motor_limits']['left_arm'], teleop_device)
        else:
            left_tensor = _to_tensor(left_cmd, 6).view(1, -1)

        head_tensor = _to_tensor(head_cmd, 1)

        processed_action[:, 0:6] = left_tensor.view(-1, 6)
        processed_action[:, 6:12] = right_tensor.view(-1, 6)
        processed_action[:, 12:15] = base_tensor.view(1, -1)
        processed_action[:, 15:] = head_tensor.view(1, -1)
    else:
        raise NotImplementedError("Only teleoperation with so101_leader, bi_so101_leader, keyboard, and xlerobot is supported for now.")
    return processed_action
