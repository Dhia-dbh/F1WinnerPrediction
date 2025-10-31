# F1 Winner Prediction Pipeline

This directory contains the refactored, modular Python pipeline for predicting F1 race winners.

## Overview

The pipeline has been refactored from the Jupyter notebook (`notebooks/test.ipynb`) into a modular, maintainable Python package with proper separation of concerns.

## Project Structure

```
src/f1winnerprediction/
├── __init__.py              # Package initialization
├── __main__.py              # CLI entry point
├── config.py                # Configuration settings
├── pipeline.py              # Main pipeline orchestration
├── io_fastf1.py             # Data loading and caching
├── preprocessing.py         # Data preprocessing and cleaning
├── split.py                 # Train/test splitting
├── metrics.py               # Model evaluation metrics
├── visualization.py         # Plotting and visualizations
├── utils.py                 # Utility functions
├── models/                  # Model implementations
└── features/                # Feature engineering modules
```

## Installation

1. Install dependencies:
```bash
poetry install
```

Or with pip:
```bash
pip install -e .
```

## Usage

### Command Line Interface

Run the complete pipeline:
```bash
python -m f1winnerprediction
```

With custom options:
```bash
python -m f1winnerprediction \
    --years 2021 2022 2023 2024 \
    --test-size 0.2 \
    --n-estimators 150 \
    --max-depth 8 \
    --learning-rate 0.05
```

View all options:
```bash
python -m f1winnerprediction --help
```

### Programmatic API

```python
from f1winnerprediction.pipeline import F1PredictionPipeline

# Initialize pipeline
pipeline = F1PredictionPipeline(years_to_fetch=[2021, 2022, 2023, 2024])

# Run complete pipeline
results = pipeline.run(
    use_cache=True,
    test_size=0.2,
    model_params={"n_estimators": 150, "max_depth": 8},
    save_model=True,
    visualize=True
)

# Print results
from f1winnerprediction.metrics import print_evaluation_summary
print_evaluation_summary(results)
```

### Step-by-Step Execution

```python
from f1winnerprediction.pipeline import F1PredictionPipeline

pipeline = F1PredictionPipeline()

# Load data
sessions = pipeline.load_data(use_cache=True)

# Preprocess
df = pipeline.preprocess_data()

# Split data
pipeline.split_data(df, test_size=0.2)

# Prepare features
pipeline.prepare_features(pipeline.df_train, pipeline.df_test)

# Train model
model = pipeline.train_model(model_params={"n_estimators": 100})

# Evaluate
results = pipeline.evaluate()

# Visualize
pipeline.visualize_results()

# Save model
pipeline.save_model("my_model.pkl")
```

## Pipeline Components

### 1. Data Loading (`io_fastf1.py`)
- Fetches race session data from FastF1
- Implements caching and checkpointing
- Handles session loading errors

### 2. Preprocessing (`preprocessing.py`)
- Extracts race results from sessions
- Cleans data (removes invalid entries, DNFs, etc.)
- Creates basic features (grid position, time, points)
- Creates driver and team features (rolling averages)
- Encodes categorical variables

### 3. Data Splitting (`split.py`)
- Splits data into train/test sets
- Supports stratified splitting
- Separates features and target

### 4. Model Training (`pipeline.py`)
- Trains XGBoost classifier
- Supports custom hyperparameters
- Handles multi-class position prediction

### 5. Evaluation (`metrics.py`)
- Calculates accuracy, F1, precision, recall
- Computes MAE and R² for position prediction
- Top-k accuracy (e.g., top-3 predictions)
- ROC AUC for probabilistic metrics
- Confusion matrix and classification report

### 6. Visualization (`visualization.py`)
- Confusion matrix heatmap
- ROC curves for multiple classes
- Feature importance plots
- Prediction distribution plots

## Configuration

Configuration is centralized in `config.py`:

```python
# Customize data directories
import os
os.environ["F1_DATA_DIR"] = "/path/to/your/data"

from f1winnerprediction import config
config.YEARS_TO_FETCH = [2020, 2021, 2022, 2023, 2024]
```

Key configuration options:
- `YEARS_TO_FETCH`: Years to fetch data for
- `FASTF1_RAW_CACHE_DIR`: FastF1 cache directory
- `FASTF1_MODELS_DIR`: Model save directory
- `FASTF1_PLOTS_DIR`: Visualization output directory
- `DEFAULT_MODEL_PARAMS`: Default XGBoost parameters

## Outputs

The pipeline generates:
- **Models**: Saved to `models/` directory (pickle format)
- **Plots**: Saved to `plots/` directory (PNG format)
  - Confusion matrix
  - ROC curves
  - Prediction distribution
- **Logs**: Saved to `logs/` directory
- **Metrics**: Printed to console and logs

## Development

### Adding New Features

1. Add feature engineering logic to `preprocessing.py`
2. Update `FEATURE_COLUMNS` in `config.py`
3. Test with the pipeline

### Adding New Models

1. Create model class in `models/` directory
2. Update `train_model()` in `pipeline.py`
3. Add model-specific parameters to `config.py`

### Adding New Metrics

1. Add metric calculation to `metrics.py`
2. Update `evaluate_model()` to include new metrics
3. Update `print_evaluation_summary()` for display

## Migration from Notebook

The original `notebooks/test.ipynb` has been deprecated in favor of this modular pipeline. The notebook is kept for reference but should not be used for new development.

Key improvements over the notebook:
- **Modularity**: Code is organized into logical modules
- **Reusability**: Functions can be imported and reused
- **Testability**: Each module can be tested independently
- **Maintainability**: Easier to update and extend
- **CLI Support**: Can be run from command line
- **Configuration**: Centralized configuration management
- **Logging**: Proper logging throughout
- **Error Handling**: Robust error handling

## Troubleshooting

### Cache Issues

If you encounter cache issues:
```bash
# Clear cache
rm -rf data/cache/*

# Fetch fresh data
python -m f1winnerprediction --no-cache
```

### Memory Issues

For large datasets:
- Reduce `YEARS_TO_FETCH` in config
- Increase test_size to reduce training data
- Use a machine with more RAM

### API Rate Limits

FastF1 may rate limit requests:
- Use cached data when possible (`--cache` flag)
- Add delays between requests (modify `io_fastf1.py`)

## Contributing

When contributing:
1. Follow existing code style
2. Add docstrings to new functions
3. Update this README
4. Test your changes

## License

[Add license information]

## Contact

[Add contact information]
