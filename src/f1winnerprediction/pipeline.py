"""
F1 Winner Prediction Pipeline

This module orchestrates the complete machine learning pipeline for predicting F1 race winners.
It coordinates data loading, preprocessing, feature engineering, model training, and evaluation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from f1winnerprediction import config
from f1winnerprediction.io_fastf1 import load_sessions, fetch_race_sessions_cache
from f1winnerprediction.preprocessing import prepare_race_data
from f1winnerprediction.split import split_train_test
from f1winnerprediction.metrics import evaluate_model
from f1winnerprediction.visualization import plot_results

logger = logging.getLogger(__name__)


class F1PredictionPipeline:
    """Main pipeline class for F1 winner prediction."""
    
    def __init__(
        self,
        years_to_fetch: Optional[List[int]] = None,
        cache_dir: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
        model_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize the F1 prediction pipeline.
        
        Args:
            years_to_fetch: List of years to fetch data for
            cache_dir: Directory for FastF1 cache
            checkpoint_dir: Directory for checkpoints
            model_dir: Directory for saving models
            output_dir: Directory for outputs
        """
        self.years_to_fetch = years_to_fetch or config.YEARS_TO_FETCH
        self.cache_dir = cache_dir or config.FASTF1_RAW_CACHE_DIR
        self.checkpoint_dir = checkpoint_dir or config.FASTF1_CHECKPOINT_DIR
        self.model_dir = model_dir or config.FASTF1_MODELS_DIR
        self.output_dir = output_dir or config.FASTF1_OUTPUT_DIR
        
        self.sessions = None
        self.df_train = None
        self.df_test = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.results = {}
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.cache_dir, self.checkpoint_dir, self.model_dir, self.output_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, use_cache: bool = True) -> Dict:
        """
        Load race session data.
        
        Args:
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary of sessions by year
        """
        logger.info("Loading race session data...")
        
        if use_cache:
            self.sessions = load_sessions()
            if not self.sessions:
                logger.warning("No cached sessions found, fetching new data...")
                self.sessions = fetch_race_sessions_cache(self.years_to_fetch)
        else:
            self.sessions = fetch_race_sessions_cache(self.years_to_fetch)
        
        logger.info(f"Loaded {len(self.sessions)} years of data")
        return self.sessions
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the loaded race sessions into a structured DataFrame.
        
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing data...")
        
        if self.sessions is None:
            raise ValueError("No sessions loaded. Call load_data() first.")
        
        df = prepare_race_data(self.sessions)
        logger.info(f"Preprocessed data shape: {df.shape}")
        
        return df
    
    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.
        
        Args:
            df: Input DataFrame
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            stratify_column: Column to use for stratification
            
        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info("Splitting data into train/test sets...")
        
        stratify = df[stratify_column] if stratify_column else None
        
        self.df_train, self.df_test = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        logger.info(f"Train set size: {len(self.df_train)}, Test set size: {len(self.df_test)}")
        
        return self.df_train, self.df_test
    
    def prepare_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        target_column: str = "Position"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare feature matrices and target vectors.
        
        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame
            feature_columns: List of feature column names
            target_column: Name of target column
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Preparing features...")
        
        if feature_columns is None:
            # Exclude target and non-feature columns
            exclude_cols = {target_column, "Driver", "Team", "Date", "EventName"}
            feature_columns = [col for col in train_df.columns if col not in exclude_cols]
        
        self.X_train = train_df[feature_columns].values
        self.X_test = test_df[feature_columns].values
        self.y_train = train_df[target_column].values
        self.y_test = test_df[target_column].values
        
        logger.info(f"Feature matrix shapes - X_train: {self.X_train.shape}, X_test: {self.X_test.shape}")
        logger.info(f"Target vector shapes - y_train: {self.y_train.shape}, y_test: {self.y_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        model_type: str = "xgboost"
    ) -> Any:
        """
        Train the prediction model.
        
        Args:
            model_params: Dictionary of model hyperparameters
            model_type: Type of model to train
            
        Returns:
            Trained model
        """
        logger.info(f"Training {model_type} model...")
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("Features not prepared. Call prepare_features() first.")
        
        if model_type == "xgboost":
            default_params = {
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42
            }
            params = {**default_params, **(model_params or {})}
            self.model = XGBClassifier(**params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model.fit(self.X_train, self.y_train)
        logger.info("Model training completed")
        
        return self.model
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model...")
        
        if self.model is None:
            raise ValueError("No model trained. Call train_model() first.")
        
        if self.X_test is None or self.y_test is None:
            raise ValueError("Test data not prepared. Call prepare_features() first.")
        
        # Get predictions
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)
        
        # Calculate metrics
        self.results = evaluate_model(self.y_test, y_pred, y_proba)
        
        logger.info("Evaluation completed")
        logger.info(f"Test Accuracy: {self.results.get('accuracy', 0):.4f}")
        
        return self.results
    
    def visualize_results(self, save_plots: bool = True) -> None:
        """
        Generate visualizations of results.
        
        Args:
            save_plots: Whether to save plots to disk
        """
        logger.info("Generating visualizations...")
        
        if not self.results:
            raise ValueError("No results to visualize. Call evaluate() first.")
        
        plots_dir = config.FASTF1_PLOTS_DIR if save_plots else None
        
        plot_results(
            y_true=self.y_test,
            y_pred=self.results.get("predictions"),
            y_proba=self.results.get("probabilities"),
            output_dir=plots_dir
        )
        
        logger.info("Visualizations completed")
    
    def save_model(self, model_name: str = "f1_winner_model.pkl") -> Path:
        """
        Save the trained model to disk.
        
        Args:
            model_name: Name of the model file
            
        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("No model to save. Call train_model() first.")
        
        model_path = self.model_dir / model_name
        
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        
        logger.info(f"Model saved to {model_path}")
        return model_path
    
    def load_model(self, model_path: Path) -> Any:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded model
        """
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        
        logger.info(f"Model loaded from {model_path}")
        return self.model
    
    def run(
        self,
        use_cache: bool = True,
        test_size: float = 0.2,
        model_params: Optional[Dict[str, Any]] = None,
        save_model: bool = True,
        visualize: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Args:
            use_cache: Whether to use cached data
            test_size: Proportion of data for testing
            model_params: Model hyperparameters
            save_model: Whether to save the trained model
            visualize: Whether to generate visualizations
            
        Returns:
            Dictionary of results
        """
        logger.info("Starting F1 prediction pipeline...")
        
        # Load data
        self.load_data(use_cache=use_cache)
        
        # Preprocess
        df = self.preprocess_data()
        
        # Split data
        self.split_data(df, test_size=test_size)
        
        # Prepare features
        self.prepare_features(self.df_train, self.df_test)
        
        # Train model
        self.train_model(model_params=model_params)
        
        # Evaluate
        results = self.evaluate()
        
        # Save model
        if save_model:
            self.save_model()
        
        # Visualize
        if visualize:
            self.visualize_results()
        
        logger.info("Pipeline completed successfully")
        
        return results


def run_pipeline(**kwargs) -> Dict[str, Any]:
    """
    Convenience function to run the pipeline with default settings.
    
    Args:
        **kwargs: Arguments to pass to pipeline.run()
        
    Returns:
        Dictionary of results
    """
    pipeline = F1PredictionPipeline()
    return pipeline.run(**kwargs)
