"""
Data Preprocessing Module

This module contains functions for cleaning, transforming, and preparing
F1 race data for machine learning models.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from fastf1.core import Session as FastF1Session

logger = logging.getLogger(__name__)


def extract_race_results(session: FastF1Session) -> pd.DataFrame:
    """
    Extract race results from a FastF1 session.
    
    Args:
        session: FastF1 race session object
        
    Returns:
        DataFrame with race results
    """
    try:
        results = session.results
        
        # Add session metadata
        results["EventName"] = session.event.EventName
        results["Year"] = session.event.year
        results["RoundNumber"] = session.event.RoundNumber
        results["Date"] = session.date
        
        return results
    except Exception as e:
        logger.error(f"Error extracting results from session: {e}")
        return pd.DataFrame()


def prepare_race_data(sessions: Dict[int, List[FastF1Session]]) -> pd.DataFrame:
    """
    Prepare race data from sessions dictionary.
    
    Args:
        sessions: Dictionary mapping years to lists of FastF1 sessions
        
    Returns:
        Consolidated DataFrame with race data
    """
    all_results = []
    
    for year, year_sessions in sessions.items():
        logger.info(f"Processing {len(year_sessions)} sessions for year {year}")
        
        for session in year_sessions:
            results_df = extract_race_results(session)
            if not results_df.empty:
                all_results.append(results_df)
    
    if not all_results:
        logger.warning("No race results extracted")
        return pd.DataFrame()
    
    # Concatenate all results
    df = pd.concat(all_results, ignore_index=True)
    
    logger.info(f"Extracted {len(df)} race results from {len(all_results)} sessions")
    
    return df


def clean_race_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean race data by handling missing values and invalid entries.
    
    Args:
        df: Raw race data DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning race data...")
    
    initial_rows = len(df)
    
    # Remove rows with missing Position
    df = df.dropna(subset=["Position"])
    
    # Remove disqualified or non-finishing positions
    df = df[df["Position"] != "DQ"]
    df = df[df["Position"] != "DNS"]
    df = df[df["Position"] != "DNF"]
    
    # Convert Position to numeric
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
    df = df.dropna(subset=["Position"])
    df["Position"] = df["Position"].astype(int)
    
    # Remove invalid positions (e.g., > 20)
    df = df[df["Position"] <= 20]
    
    logger.info(f"Cleaned data: {initial_rows} -> {len(df)} rows ({initial_rows - len(df)} removed)")
    
    return df


def create_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create basic features from race data.
    
    Args:
        df: Race data DataFrame
        
    Returns:
        DataFrame with additional features
    """
    logger.info("Creating basic features...")
    
    # Grid position
    if "GridPosition" in df.columns:
        df["GridPosition"] = pd.to_numeric(df["GridPosition"], errors="coerce")
        df["GridPosition"] = df["GridPosition"].fillna(20)  # Assume back of grid if missing
    
    # Time features
    if "Time" in df.columns:
        # Convert time to seconds
        df["TimeSeconds"] = df["Time"].apply(lambda x: x.total_seconds() if pd.notna(x) else np.nan)
    
    # Points
    if "Points" in df.columns:
        df["Points"] = pd.to_numeric(df["Points"], errors="coerce").fillna(0)
    
    # Status (did they finish?)
    if "Status" in df.columns:
        df["DidFinish"] = df["Status"].apply(lambda x: 1 if "Finished" in str(x) else 0)
    
    # Calculate position change
    if "GridPosition" in df.columns and "Position" in df.columns:
        df["PositionChange"] = df["GridPosition"] - df["Position"]
    
    return df


def create_driver_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create driver-specific features.
    
    Args:
        df: Race data DataFrame
        
    Returns:
        DataFrame with driver features
    """
    logger.info("Creating driver features...")
    
    # Sort by date to ensure chronological order
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Driver statistics (rolling averages)
    driver_stats = []
    
    for driver in df["Driver"].unique():
        driver_df = df[df["Driver"] == driver].copy()
        
        # Calculate rolling averages
        driver_df["DriverAvgPosition"] = driver_df["Position"].expanding().mean().shift(1)
        driver_df["DriverAvgPoints"] = driver_df["Points"].expanding().mean().shift(1) if "Points" in driver_df.columns else np.nan
        driver_df["DriverRaceCount"] = range(1, len(driver_df) + 1)
        
        driver_stats.append(driver_df)
    
    df = pd.concat(driver_stats).sort_index()
    
    # Fill NaN for first races
    df["DriverAvgPosition"] = df["DriverAvgPosition"].fillna(10)  # Assume mid-field
    df["DriverAvgPoints"] = df["DriverAvgPoints"].fillna(0)
    
    return df


def create_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create team-specific features.
    
    Args:
        df: Race data DataFrame
        
    Returns:
        DataFrame with team features
    """
    logger.info("Creating team features...")
    
    # Sort by date to ensure chronological order
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Team statistics (rolling averages)
    team_stats = []
    
    for team in df["Team"].unique():
        team_df = df[df["Team"] == team].copy()
        
        # Calculate rolling averages
        team_df["TeamAvgPosition"] = team_df["Position"].expanding().mean().shift(1)
        team_df["TeamAvgPoints"] = team_df["Points"].expanding().mean().shift(1) if "Points" in team_df.columns else np.nan
        team_df["TeamRaceCount"] = range(1, len(team_df) + 1)
        
        team_stats.append(team_df)
    
    df = pd.concat(team_stats).sort_index()
    
    # Fill NaN for first races
    df["TeamAvgPosition"] = df["TeamAvgPosition"].fillna(10)
    df["TeamAvgPoints"] = df["TeamAvgPoints"].fillna(0)
    
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features.
    
    Args:
        df: Race data DataFrame
        
    Returns:
        DataFrame with encoded features
    """
    logger.info("Encoding categorical features...")
    
    # Label encode drivers
    if "Driver" in df.columns:
        df["DriverEncoded"] = pd.Categorical(df["Driver"]).codes
    
    # Label encode teams
    if "Team" in df.columns:
        df["TeamEncoded"] = pd.Categorical(df["Team"]).codes
    
    # Label encode tracks
    if "EventName" in df.columns:
        df["TrackEncoded"] = pd.Categorical(df["EventName"]).codes
    
    return df


def select_features(df: pd.DataFrame, feature_list: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Select and order features for modeling.
    
    Args:
        df: Race data DataFrame
        feature_list: Optional list of features to select
        
    Returns:
        DataFrame with selected features
    """
    if feature_list is None:
        # Default feature set
        potential_features = [
            "Position",  # Target variable
            "Driver",
            "Team",
            "EventName",
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
            "RoundNumber",
            "Date"
        ]
        
        # Only include features that exist in the DataFrame
        feature_list = [f for f in potential_features if f in df.columns]
    
    return df[feature_list]


def preprocess_pipeline(sessions: Dict[int, List[FastF1Session]]) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.
    
    Args:
        sessions: Dictionary mapping years to lists of FastF1 sessions
        
    Returns:
        Fully preprocessed DataFrame ready for modeling
    """
    logger.info("Starting preprocessing pipeline...")
    
    # Extract raw data
    df = prepare_race_data(sessions)
    
    if df.empty:
        logger.error("No data to preprocess")
        return df
    
    # Clean data
    df = clean_race_data(df)
    
    # Create features
    df = create_basic_features(df)
    df = create_driver_features(df)
    df = create_team_features(df)
    df = encode_categorical_features(df)
    
    # Select features
    df = select_features(df)
    
    logger.info(f"Preprocessing complete. Final shape: {df.shape}")
    logger.info(f"Features: {list(df.columns)}")
    
    return df
