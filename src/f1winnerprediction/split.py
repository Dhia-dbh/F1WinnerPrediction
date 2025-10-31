"""
Data Splitting Module

This module contains functions for splitting data into train/test sets.
"""

import logging
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_column: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into training and testing sets.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data for testing (0.0 to 1.0)
        random_state: Random seed for reproducibility
        stratify_column: Column name to use for stratified splitting
        
    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info(f"Splitting data: test_size={test_size}, stratify={stratify_column}")
    
    stratify = df[stratify_column] if stratify_column and stratify_column in df.columns else None
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    
    logger.info(f"Train set: {len(train_df)} samples, Test set: {len(test_df)} samples")
    
    return train_df, test_df


def split_features_target(
    df: pd.DataFrame,
    target_column: str = "Position",
    exclude_columns: Optional[list] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features and target.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        exclude_columns: List of columns to exclude from features
        
    Returns:
        Tuple of (features_df, target_series)
    """
    if exclude_columns is None:
        exclude_columns = []
    
    # Always exclude target and common non-feature columns
    exclude_set = set(exclude_columns + [target_column, "Driver", "Team", "EventName", "Date"])
    
    feature_columns = [col for col in df.columns if col not in exclude_set]
    
    X = df[feature_columns]
    y = df[target_column]
    
    logger.info(f"Features: {len(feature_columns)} columns, Target: {target_column}")
    
    return X, y
