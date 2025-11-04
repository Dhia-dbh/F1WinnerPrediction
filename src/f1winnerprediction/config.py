"""
Configuration Module

This module contains configuration settings and paths for the F1 prediction pipeline.
"""

import os
import pathlib
from pathlib import Path

# Get project root directory (repository root)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Base data directory - can be overridden by environment variable
BASE_DATA_DIR = Path(PROJECT_ROOT / "data")

# FastF1 directories
FASTF1_RAW_CACHE_DIR: pathlib.Path = BASE_DATA_DIR / "cache"
FASTF1_CHECKPOINT_DIR: pathlib.Path = BASE_DATA_DIR / "checkpoints"
FASTF1_DATA_DIR: pathlib.Path = BASE_DATA_DIR / "processed"
FASTF1_MODELS_DIR: pathlib.Path = PROJECT_ROOT / "models"
FASTF1_OUTPUT_DIR: pathlib.Path = PROJECT_ROOT / "outputs"
FASTF1_LOGS_DIR: pathlib.Path = PROJECT_ROOT / "logs"
FASTF1_PLOTS_DIR: pathlib.Path = PROJECT_ROOT / "plots"

# Data fetching configuration
YEARS_TO_FETCH = list(range(2021, 2025 + 1))

# Default checkpoint configuration
DEFAULT_CHECKPOINT = {
    "year": YEARS_TO_FETCH[0] if YEARS_TO_FETCH else 2021,
    "gp_index_start": 1,
}

# Model configuration
DEFAULT_MODEL_PARAMS = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": 42
}

# Feature configuration
FEATURE_COLUMNS = [
    "GridPosition",
    "DriverEncoded",
    "TeamEncoded",
    "TrackEncoded",
    "DriverAvgPosition",
    "DriverAvgPoints",
    "DriverRaceCount",
    "TeamAvgPosition",
    "TeamAvgPoints",
    "TeamRaceCount",
    "PositionChange",
    "Points",
    "DidFinish",
    "Year",
    "RoundNumber"
]

# Target column
TARGET_COLUMN = "Position"