# F1 Winner Prediction

Machine learning pipeline for predicting Formula 1 race winners using historical race data.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Dhia-dbh/F1WinnerPrediction.git
cd F1WinnerPrediction

# Install dependencies with Poetry
poetry install

# Or with pip
pip install -e .
```

### Run the Pipeline

```bash
# Run with default settings
python -m f1winnerprediction

# Customize parameters
python -m f1winnerprediction --years 2021 2022 2023 2024 --test-size 0.2 --n-estimators 150
```

## 📋 Features

- **Data Collection**: Automated fetching of F1 race data using FastF1
- **Preprocessing**: Cleaning, feature engineering, and data transformation
- **Model Training**: XGBoost classifier for multi-class position prediction
- **Evaluation**: Comprehensive metrics including accuracy, F1, MAE, top-k accuracy, ROC AUC
- **Visualization**: Confusion matrices, ROC curves, and distribution plots
- **CLI Interface**: Easy command-line execution with customizable parameters
- **Modular Design**: Well-organized, reusable components

## 📁 Project Structure

```
F1WinnerPrediction/
├── src/f1winnerprediction/      # Main package
│   ├── __main__.py               # CLI entry point
│   ├── pipeline.py               # Pipeline orchestration
│   ├── io_fastf1.py              # Data loading
│   ├── preprocessing.py          # Data preprocessing
│   ├── split.py                  # Train/test splitting
│   ├── metrics.py                # Evaluation metrics
│   ├── visualization.py          # Plotting
│   ├── config.py                 # Configuration
│   ├── models/                   # Model implementations
│   └── features/                 # Feature engineering
├── notebooks/                    # Jupyter notebooks (deprecated)
├── tests/                        # Unit tests
├── PIPELINE_README.md           # Detailed pipeline documentation
└── pyproject.toml               # Project dependencies
```

## 📖 Documentation

- **[PIPELINE_README.md](PIPELINE_README.md)**: Comprehensive pipeline documentation
- **[Notebooks](notebooks/)**: Original Jupyter notebooks (deprecated, kept for reference)

## 🔧 Usage

### Command Line Interface

```bash
# Basic usage
python -m f1winnerprediction

# View all options
python -m f1winnerprediction --help

# Custom configuration
python -m f1winnerprediction \
    --years 2021 2022 2023 \
    --test-size 0.2 \
    --n-estimators 100 \
    --max-depth 6 \
    --learning-rate 0.1 \
    --log-level DEBUG
```

### Python API

```python
from f1winnerprediction.pipeline import F1PredictionPipeline
from f1winnerprediction.metrics import print_evaluation_summary

# Initialize and run pipeline
pipeline = F1PredictionPipeline(years_to_fetch=[2021, 2022, 2023, 2024])
results = pipeline.run(
    use_cache=True,
    test_size=0.2,
    model_params={"n_estimators": 150, "max_depth": 8},
    save_model=True,
    visualize=True
)

# Print results
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

# Split
pipeline.split_data(df, test_size=0.2)

# Prepare features
pipeline.prepare_features(pipeline.df_train, pipeline.df_test)

# Train
model = pipeline.train_model()

# Evaluate
results = pipeline.evaluate()

# Visualize
pipeline.visualize_results()

# Save
pipeline.save_model()
```

## 🎯 Pipeline Overview

1. **Data Loading**: Fetch race sessions from FastF1 API with caching
2. **Preprocessing**: Clean data, handle missing values, remove DNFs
3. **Feature Engineering**: Create driver/team stats, encode categories
4. **Data Splitting**: Train/test split with optional stratification
5. **Model Training**: Train XGBoost multi-class classifier
6. **Evaluation**: Calculate accuracy, F1, MAE, top-k accuracy, ROC AUC
7. **Visualization**: Generate confusion matrix, ROC curves, distributions

## 📊 Results

The model predicts race finishing positions with:
- **Accuracy**: ~11-15% (20+ class problem)
- **Top-3 Accuracy**: ~28-30%
- **MAE**: ~5-6 positions
- **ROC AUC**: ~0.65-0.70

## 🛠️ Configuration

Configure the pipeline in `src/f1winnerprediction/config.py` or via environment variables:

```python
# Set custom data directory
export F1_DATA_DIR=/path/to/data

# Modify years to fetch
from f1winnerprediction import config
config.YEARS_TO_FETCH = [2020, 2021, 2022, 2023, 2024]
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=f1winnerprediction tests/
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

[Add your license here]

## 👥 Authors

- Dhia Ben Hamouda - [GitHub](https://github.com/Dhia-dbh)

## 🙏 Acknowledgments

- [FastF1](https://github.com/theOehrly/Fast-F1) for F1 data access
- scikit-learn and XGBoost for machine learning tools

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.
