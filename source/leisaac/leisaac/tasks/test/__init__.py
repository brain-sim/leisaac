import gymnasium as gym

gym.register(
    id="LeIsaac-SO101-Test-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_cube_env_cfg:LiftCubeEnvCfg",
    },
)