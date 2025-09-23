import gymnasium as gym

# Single arm environments
gym.register(
    id="LeIsaac-SO101-LiftCube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_cube_env_cfg:LiftCubeEnvCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-LiftCube-DigitalTwin-v0",
    entry_point="leisaac.enhance.envs:ManagerBasedRLDigitalTwinEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_cube_env_cfg:LiftCubeDigitalTwinEnvCfg",
    },
)

# Bi-arm handoff environment
gym.register(
    id="LeIsaac-SO101-LiftCube-BiArm-Handoff-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_cube_bi_arm_handoff_env_cfg:LiftCubeBiArmHandoffEnvCfg",
    },
)

# gym.register(
#     id="LeIsaac-SO101-LiftCube-BiArm-Handoff-DigitalTwin-v0",
#     entry_point="leisaac.enhance.envs:ManagerBasedRLDigitalTwinEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.lift_cube_bi_arm_handoff_env_cfg:LiftCubeBiArmHandoffDigitalTwinEnvCfg",
#     },
# )

gym.register(
    id="LeIsaac-SO101-LiftCube-Mimic-v0",
    entry_point=f"leisaac.enhance.envs:ManagerBasedRLLeIsaacMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_cube_mimic_env_cfg:LiftCubeMimicEnvCfg",
    },
)