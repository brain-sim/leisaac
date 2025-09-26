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
        if hasattr(action_cfg, 'left_arm_action') and hasattr(action_cfg, 'right_arm_action'):
            action_cfg.left_arm_action = mdp.DifferentialInverseKinematicsActionCfg(
                asset_name="left_arm",
                joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                body_name="gripper",
                controller=mdp.DifferentialIKControllerCfg(command_type="pose", ik_method="dls", use_relative_mode=True),
            )
            action_cfg.left_gripper_action = mdp.RelativeJointPositionActionCfg(
                asset_name="left_arm",
                joint_names=["gripper"],
                scale=0.7,
            )
            action_cfg.right_arm_action = mdp.DifferentialInverseKinematicsActionCfg(
                asset_name="right_arm",
                joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                body_name="gripper",
                controller=mdp.DifferentialIKControllerCfg(command_type="pose", ik_method="dls", use_relative_mode=True),
            )
            action_cfg.right_gripper_action = mdp.RelativeJointPositionActionCfg(
                asset_name="right_arm",
                joint_names=["gripper"],
                scale=0.7,
            )
        else:
            action_cfg.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                body_name="gripper",
                controller=mdp.DifferentialIKControllerCfg(command_type="pose", ik_method="dls", use_relative_mode=True),
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
        if hasattr(action_cfg, 'left_arm_action') and hasattr(action_cfg, 'right_arm_action'):
            action_cfg.left_arm_action = mdp.DifferentialInverseKinematicsActionCfg(
                asset_name="left_arm",
                joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                body_name="gripper",
                controller=mdp.DifferentialIKControllerCfg(command_type="pose", ik_method="dls", use_relative_mode=True),
            )
            action_cfg.left_gripper_action = mdp.RelativeJointPositionActionCfg(
                asset_name="left_arm",
                joint_names=["gripper"],
                scale=0.7,
            )
            action_cfg.right_arm_action = mdp.DifferentialInverseKinematicsActionCfg(
                asset_name="right_arm",
                joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                body_name="gripper",
                controller=mdp.DifferentialIKControllerCfg(command_type="pose", ik_method="dls", use_relative_mode=True),
            )
            action_cfg.right_gripper_action = mdp.RelativeJointPositionActionCfg(
                asset_name="right_arm",
                joint_names=["gripper"],
                scale=0.7,
            )
        else:
            action_cfg.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                body_name="gripper",
                controller=mdp.DifferentialIKControllerCfg(command_type="pose", ik_method="dls", use_relative_mode=True),
            )
            action_cfg.gripper_action = mdp.RelativeJointPositionActionCfg(
                asset_name="robot",
                joint_names=["gripper"],
                scale=0.7,
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
        twist = action.get('ee_twist') if action.get('ee_twist') is not None else action.get('joint_state')
        if twist is None:
            raise ValueError("Keyboard teleoperation expects an 'ee_twist' command.")
        if isinstance(twist, torch.Tensor):
            twist_tensor = twist.clone()
        else:
            twist_tensor = torch.as_tensor(twist)
        if twist_tensor.dim() == 1:
            twist_tensor = twist_tensor.unsqueeze(0)
        if twist_tensor.shape[0] != teleop_device.env.num_envs:
            twist_tensor = twist_tensor.repeat(teleop_device.env.num_envs, 1)
        twist_tensor = twist_tensor.to(device=teleop_device.env.device, dtype=torch.float32)

        gripper = action.get('gripper')
        gripper_tensor = None
        if gripper is not None:
            if isinstance(gripper, torch.Tensor):
                gripper_tensor = gripper.clone()
            else:
                gripper_tensor = torch.as_tensor(gripper)
            if gripper_tensor.dim() == 1:
                gripper_tensor = gripper_tensor.unsqueeze(0)
            if gripper_tensor.shape[0] != teleop_device.env.num_envs:
                gripper_tensor = gripper_tensor.repeat(teleop_device.env.num_envs, 1)
            gripper_tensor = gripper_tensor.to(device=teleop_device.env.device, dtype=torch.float32)

        action_segments = []
        arm_dim = 6
        num_arms = twist_tensor.shape[1] // arm_dim
        for arm_idx in range(num_arms):
            start_idx = arm_idx * arm_dim
            end_idx = start_idx + arm_dim
            action_segments.append(twist_tensor[:, start_idx:end_idx])
            if gripper_tensor is not None:
                action_segments.append(gripper_tensor[:, arm_idx : arm_idx + 1])
        processed_action = torch.cat(action_segments, dim=1) if action_segments else twist_tensor
    elif action.get('bi_so101_leader') is not None:
        processed_action = torch.zeros(teleop_device.env.num_envs, 12, device=teleop_device.env.device)
        processed_action[:, :6] = convert_action_from_so101_leader(action['joint_state']['left_arm'], action['motor_limits']['left_arm'], teleop_device)
        processed_action[:, 6:] = convert_action_from_so101_leader(action['joint_state']['right_arm'], action['motor_limits']['right_arm'], teleop_device)
    else:
        raise NotImplementedError("Only teleoperation with so101_leader, bi_so101_leader, keyboard is supported for now.")
    return processed_action
