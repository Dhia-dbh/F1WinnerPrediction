import pathlib
from pathlib import Path

PROJECT_ROOT: pathlib.Path = Path(__file__).parent.parent.parent.resolve()

BASE_DIR: pathlib.Path = PROJECT_ROOT# / "f1winnerprediction"
FASTF1_DATA_DIR: pathlib.Path = BASE_DIR / "data"
FASTF1_REPORTS: pathlib.Path = BASE_DIR / "reports"

FASTF1_MODELS_DIR: pathlib.Path = BASE_DIR / "models"
FASTF1_OUTPUT_DIR: pathlib.Path = BASE_DIR / "outputs"
FASTF1_LOGS_DIR: pathlib.Path = BASE_DIR / "logs"
FASTF1_PLOTS_DIR: pathlib.Path = FASTF1_REPORTS / "plots"
FASTF1_RAW_CACHE_DIR: pathlib.Path = FASTF1_DATA_DIR / "cache"
FASTF1_CHECKPOINT_DIR: pathlib.Path = FASTF1_DATA_DIR / "checkpoints"

YEARS_TO_FETCH = list(range(2021, 2025 + 1))

DEFAULT_CHECKPOINT = {
      "year": YEARS_TO_FETCH[0],
      "gp_index_start": 1,
}