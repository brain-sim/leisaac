import gymnasium as gym


gym.register(
    id="Household-Dishwashing-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dishwashing_env_cfg:DishwashingEnvCfg",
    },
)

gym.register(
    id="Household-Microwaving-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.microwaving_env_cfg:MicrowavingEnvCfg",
    },
)

gym.register(
    id="Household-FridgeStocking-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fridge_stocking_env_cfg:FridgeStockingEnvCfg",
    },
)

gym.register(
    id="Household-PlateArrangement-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.plate_arrangement_env_cfg:PlateArrangementEnvCfg",
    },
)

gym.register(
    id="Household-ShelfSorting-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shelf_sorting_env_cfg:ShelfSortingEnvCfg",
    },
)

gym.register(
    id="Household-DishwasherRestock-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dishwasher_restock_env_cfg:DishwasherRestockEnvCfg",
    },
)

gym.register(
    id="Household-MicrowaveMealPrep-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.microwave_meal_prep_env_cfg:MicrowaveMealPrepEnvCfg",
    },
)

gym.register(
    id="Household-CoffeeService-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.coffee_service_env_cfg:CoffeeServiceEnvCfg",
    },
)

gym.register(
    id="Household-BreakfastSetup-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.breakfast_setup_env_cfg:BreakfastSetupEnvCfg",
    },
)

gym.register(
    id="Household-FruitDisplay-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fruit_display_env_cfg:FruitDisplayEnvCfg",
    },
)

gym.register(
    id="Household-PantryLoading-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pantry_loading_env_cfg:PantryLoadingEnvCfg",
    },
)

gym.register(
    id="Household-UtensilStation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.utensil_station_env_cfg:UtensilStationEnvCfg",
    },
)
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

