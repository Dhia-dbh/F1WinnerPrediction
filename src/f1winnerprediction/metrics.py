"""
Model Evaluation Metrics Module

This module contains functions for evaluating machine learning models.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    top_k_accuracy_score,
    mean_absolute_error,
    r2_score
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for probabilistic metrics)
        num_classes: Number of classes (inferred if not provided)
        
    Returns:
        Dictionary containing various evaluation metrics
    """
    logger.info("Evaluating model performance...")
    
    if num_classes is None:
        num_classes = len(np.unique(y_true))
    
    results = {
        "predictions": y_pred,
        "probabilities": y_proba,
        "num_classes": num_classes
    }
    
    # Basic classification metrics
    results["accuracy"] = accuracy_score(y_true, y_pred)
    results["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    results["f1_micro"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
    results["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    results["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    results["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Confusion matrix
    results["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    
    # Classification report
    results["classification_report"] = classification_report(y_true, y_pred, zero_division=0)
    
    # Regression-like metrics for position prediction
    results["mae"] = mean_absolute_error(y_true, y_pred)
    results["r2"] = r2_score(y_true, y_pred)
    
    # Probabilistic metrics (if probabilities provided)
    if y_proba is not None:
        # Top-k accuracy
        try:
            results["top3_accuracy"] = top_k_accuracy_score(
                y_true, y_proba, k=3, labels=np.arange(num_classes)
            )
            results["top5_accuracy"] = top_k_accuracy_score(
                y_true, y_proba, k=5, labels=np.arange(num_classes)
            )
        except Exception as e:
            logger.warning(f"Could not calculate top-k accuracy: {e}")
        
        # ROC AUC (multiclass one-vs-rest)
        try:
            y_bin = label_binarize(y_true, classes=np.arange(num_classes))
            results["roc_auc_macro"] = roc_auc_score(
                y_bin, y_proba, average="macro", multi_class="ovr"
            )
            results["roc_auc_weighted"] = roc_auc_score(
                y_bin, y_proba, average="weighted", multi_class="ovr"
            )
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {e}")
    
    # Log key metrics
    logger.info(f"Accuracy: {results['accuracy']:.4f}")
    logger.info(f"F1 Score (Macro): {results['f1_macro']:.4f}")
    logger.info(f"MAE: {results['mae']:.4f}")
    
    if "top3_accuracy" in results:
        logger.info(f"Top-3 Accuracy: {results['top3_accuracy']:.4f}")
    
    return results


def print_evaluation_summary(results: Dict[str, Any]) -> None:
    """
    Print a summary of evaluation results.
    
    Args:
        results: Dictionary of evaluation results
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION SUMMARY")
    print("="*50)
    
    print(f"\nAccuracy: {results.get('accuracy', 0):.4f}")
    print(f"F1 Score (Macro): {results.get('f1_macro', 0):.4f}")
    print(f"F1 Score (Weighted): {results.get('f1_weighted', 0):.4f}")
    print(f"Precision (Macro): {results.get('precision_macro', 0):.4f}")
    print(f"Recall (Macro): {results.get('recall_macro', 0):.4f}")
    
    print(f"\nMean Absolute Error: {results.get('mae', 0):.4f}")
    print(f"R² Score: {results.get('r2', 0):.4f}")
    
    if "top3_accuracy" in results:
        print(f"\nTop-3 Accuracy: {results.get('top3_accuracy', 0):.4f}")
    if "top5_accuracy" in results:
        print(f"Top-5 Accuracy: {results.get('top5_accuracy', 0):.4f}")
    
    if "roc_auc_macro" in results:
        print(f"\nROC AUC (Macro): {results.get('roc_auc_macro', 0):.4f}")
        print(f"ROC AUC (Weighted): {results.get('roc_auc_weighted', 0):.4f}")
    
    print("\n" + "-"*50)
    print("Classification Report:")
    print("-"*50)
    print(results.get("classification_report", "Not available"))
    print("="*50 + "\n")
