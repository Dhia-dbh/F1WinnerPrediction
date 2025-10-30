import pathlib
from pathlib import Path

FASTF1_RAW_CACHE_DIR: pathlib.Path = Path("/media/dhiabenhamouda/Dhia/Work/F1WinnerPrediction/src/f1winnerprediction/cache")
FASTF1_CHECKPOINT_DIR: pathlib.Path = Path("/media/dhiabenhamouda/Dhia/Work/F1WinnerPrediction/src/f1winnerprediction/checkpoints")
FASTF1_DATA_DIR: pathlib.Path = Path("/media/dhiabenhamouda/Dhia/Work/F1WinnerPrediction/src/f1winnerprediction/data")
FASTF1_MODELS_DIR: pathlib.Path = Path("/media/dhiabenhamouda/Dhia/Work/F1WinnerPrediction/src/f1winnerprediction/models")
FASTF1_OUTPUT_DIR: pathlib.Path = Path("/media/dhiabenhamouda/Dhia/Work/F1WinnerPrediction/src/f1winnerprediction/outputs")
FASTF1_LOGS_DIR: pathlib.Path = Path("/media/dhiabenhamouda/Dhia/Work/F1WinnerPrediction/src/f1winnerprediction/logs")
FASTF1_PLOTS_DIR: pathlib.Path = Path("/media/dhiabenhamouda/Dhia/Work/F1WinnerPrediction/src/f1winnerprediction/plots")

YEARS_TO_FETCH = list(range(2021, 2025 + 1))

DEFAULT_CHECKPOINT = {
      "year": YEARS_TO_FETCH[0],
      "gp_index_start": 1,
}