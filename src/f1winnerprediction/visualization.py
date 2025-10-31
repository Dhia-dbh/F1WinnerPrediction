"""
Visualization Module

This module contains functions for creating plots and visualizations.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        figsize: Figure size
        save_path: Path to save the plot
    """
    logger.info("Plotting confusion matrix...")
    
    if labels is None:
        labels = np.unique(y_true)
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Position')
    plt.ylabel('True Position')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    num_classes: Optional[int] = None,
    max_classes_to_plot: int = 8,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None
) -> None:
    """
    Plot ROC curves for multiclass classification.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        num_classes: Number of classes
        max_classes_to_plot: Maximum number of classes to plot
        figsize: Figure size
        save_path: Path to save the plot
    """
    logger.info("Plotting ROC curves...")
    
    if num_classes is None:
        num_classes = len(np.unique(y_true))
    
    # Binarize labels for multiclass ROC
    y_bin = label_binarize(y_true, classes=np.arange(num_classes))
    
    plt.figure(figsize=figsize)
    
    # Plot ROC curves for each class (limited to max_classes_to_plot)
    for i in range(min(max_classes_to_plot, num_classes)):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Position {i+1} (AUC = {roc_auc:.2f})')
        except Exception as e:
            logger.warning(f"Could not plot ROC for class {i}: {e}")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves by Position')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curves saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_feature_importance(
    feature_names: list,
    importance_values: np.ndarray,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None
) -> None:
    """
    Plot feature importance.
    
    Args:
        feature_names: List of feature names
        importance_values: Feature importance values
        top_n: Number of top features to display
        figsize: Figure size
        save_path: Path to save the plot
    """
    logger.info("Plotting feature importance...")
    
    # Sort features by importance
    indices = np.argsort(importance_values)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importance = importance_values[indices]
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(top_features)), top_importance)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_prediction_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[Path] = None
) -> None:
    """
    Plot distribution of true vs predicted positions.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        figsize: Figure size
        save_path: Path to save the plot
    """
    logger.info("Plotting prediction distribution...")
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # True position distribution
    axes[0].hist(y_true, bins=20, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('True Position Distribution')
    axes[0].grid(alpha=0.3)
    
    # Predicted position distribution
    axes[1].hist(y_pred, bins=20, edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Predicted Position Distribution')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Distribution plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    output_dir: Optional[Path] = None
) -> None:
    """
    Generate all result plots.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        output_dir: Directory to save plots
    """
    logger.info("Generating result visualizations...")
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Confusion matrix
    cm_path = output_dir / "confusion_matrix.png" if output_dir else None
    plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
    
    # ROC curves (if probabilities available)
    if y_proba is not None:
        roc_path = output_dir / "roc_curves.png" if output_dir else None
        plot_roc_curves(y_true, y_proba, save_path=roc_path)
    
    # Prediction distribution
    dist_path = output_dir / "prediction_distribution.png" if output_dir else None
    plot_prediction_distribution(y_true, y_pred, save_path=dist_path)
    
    logger.info("Visualization complete")
