from pathlib import Path

from leisaac.utils.constant import ASSETS_ROOT

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg


"""Configuration for the Table with Cube Scene"""
SCENES_ROOT = Path(ASSETS_ROOT) / "scenes"
TEST_WITH_CUBE_USD_PATH = str(SCENES_ROOT / "test" / "scene.usd")

TEST_WITH_CUBE_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=TEST_WITH_CUBE_USD_PATH,
    )
)
