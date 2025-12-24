"""
Model evaluation utilities.
"""

import numpy as np


def calculate_accuracy(y_true, y_pred):
    """
    Calculate accuracy score.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        
    Returns:
        float: Accuracy score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    return np.mean(y_true == y_pred)


def calculate_f1_score(y_true, y_pred):
    """
    Calculate F1 score for binary classification.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        
    Returns:
        float: F1 score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate precision and recall
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calculate F1
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
        
    return f1
