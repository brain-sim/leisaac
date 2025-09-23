from pathlib import Path

from leisaac.utils.constant import ASSETS_ROOT

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg


"""Configuration for the Toy Room Scene"""
SCENES_ROOT = Path(ASSETS_ROOT) / "scenes"

LIGHTWHEEL_BEDROOM_USD_PATH = str(SCENES_ROOT / "lightwheel_bedroom" / "scene.usd")
LIGHTWHEEL_TISSUE004_USD_PATH = str(
    SCENES_ROOT / "lightwheel_bedroom" / "LW_Loft" / "Loft" / "Tissue004" / "Tissue004.usd"
)

LIGHTWHEEL_BEDROOM_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=LIGHTWHEEL_BEDROOM_USD_PATH,
    )
)
