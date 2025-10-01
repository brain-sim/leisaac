from pathlib import Path

from leisaac.utils.constant import ASSETS_ROOT

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg


"""Configuration for the Table with Cube Scene"""
SCENES_ROOT = Path(ASSETS_ROOT) / "scenes"
EMPTY_USD_PATH = str(SCENES_ROOT / "empty" / "scene.usd")

EMPTY_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=EMPTY_USD_PATH,
    )
)
