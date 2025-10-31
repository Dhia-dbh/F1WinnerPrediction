"""
Main Entry Point for F1 Winner Prediction Pipeline

This script provides a command-line interface for running the F1 prediction pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path

import fastf1

from f1winnerprediction import config
from f1winnerprediction.pipeline import F1PredictionPipeline
from f1winnerprediction.metrics import print_evaluation_summary


def setup_logging(log_level: str = "INFO", log_file: Path = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
    """
    # Create logs directory if needed
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="F1 Winner Prediction Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data options
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=config.YEARS_TO_FETCH,
        help="Years to fetch data for"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use cached data, fetch fresh"
    )
    
    # Training options
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for testing"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Model options
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of estimators for XGBoost"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum tree depth for XGBoost"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for XGBoost"
    )
    
    # Output options
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the trained model"
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Don't generate visualizations"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.FASTF1_OUTPUT_DIR,
        help="Output directory for results"
    )
    
    # Logging options
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=config.FASTF1_LOGS_DIR / "pipeline.log",
        help="Log file path"
    )
    
    return parser.parse_args()


def main():
    """Main function to run the pipeline."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("F1 Winner Prediction Pipeline")
    logger.info("="*60)
    
    # Enable FastF1 cache
    logger.info(f"Enabling FastF1 cache at {config.FASTF1_RAW_CACHE_DIR}")
    fastf1.Cache.enable_cache(config.FASTF1_RAW_CACHE_DIR)
    
    # Configure pandas display
    import pandas as pd
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    try:
        # Initialize pipeline
        logger.info("Initializing pipeline...")
        pipeline = F1PredictionPipeline(
            years_to_fetch=args.years,
            output_dir=args.output_dir
        )
        
        # Prepare model parameters
        model_params = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "random_state": args.random_state
        }
        
        # Run pipeline
        logger.info("Starting pipeline execution...")
        results = pipeline.run(
            use_cache=not args.no_cache,
            test_size=args.test_size,
            model_params=model_params,
            save_model=not args.no_save,
            visualize=not args.no_viz
        )
        
        # Print evaluation summary
        print_evaluation_summary(results)
        
        logger.info("="*60)
        logger.info("Pipeline completed successfully!")
        logger.info("="*60)
        
        return 0
    
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
