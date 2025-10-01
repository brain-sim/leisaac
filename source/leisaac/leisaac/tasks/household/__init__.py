import gymnasium as gym

gym.register(
    id="Kitchen-FridgeStocking-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_fridge_stocking_env_cfg:FridgeStockingEnvCfg",
    },
)


gym.register(
    id="Kitchen-FridgeOrangePlacement-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_subtask_env_cfg:FridgeOrangePlacementEnvCfg",
    },
)

gym.register(
    id="Kitchen-FridgeBottlePlacement-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_subtask_env_cfg:FridgeBottlePlacementEnvCfg",
    },
)

gym.register(
    id="Kitchen-CounterBottlePlacement-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_subtask_env_cfg:CounterBottlePlacementEnvCfg",
    },
)

gym.register(
    id="Kitchen-CounterPlatePlacement-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_subtask_env_cfg:CounterPlatePlacementEnvCfg",
    },
)
