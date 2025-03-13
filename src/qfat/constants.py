from pathlib import Path

PROJECT_PATH = Path(__file__).parents[2]
CONFIG_PATH = PROJECT_PATH / "configs"
DATA_PATH = PROJECT_PATH / "data"
KITCHEN_DATA_PATH = DATA_PATH / "relay_kitchen"
PUSHT_DATA_PATH = DATA_PATH / "pusht"
ANT_DATA_PATH = DATA_PATH / "ant"
UR3_DATA_PATH = DATA_PATH / "ur3"
DERIVED_DATA_PATH = PROJECT_PATH / "derived_data"
