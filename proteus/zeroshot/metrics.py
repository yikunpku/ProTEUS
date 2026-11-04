#!/usr/bin/env python3
"""
Evaluation metrics for protein fitness prediction.
Includes Spearman, MSE, AUC, MCC, NDCG, and Top K% Recall.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, roc_auc_score, matthews_corrcoef


def calculate_ndcg(y_true, y_pred, k=None):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG).

    Args:
        y_true: True labels/scores
        y_pred: Predicted scores
        k: Top k items to consider (None for all)

    Returns:
        NDCG score
    """
    def dcg_score(y_true, y_pred, k=None):
        order = np.argsort(y_pred)[::-1]
        y_true = np.take(y_true, order)

        if k is not None:
            y_true = y_true[:k]

        gains = 2 ** y_true - 1
        discounts = np.log2(np.arange(len(y_true)) + 2)
        return np.sum(gains / discounts)

    actual_dcg = dcg_score(y_true, y_pred, k)
    ideal_dcg = dcg_score(y_true, y_true, k)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def calculate_top_k_recall(y_true, y_pred, k_percent=0.1):
    """
    Calculate Top K% Recall.

    Args:
        y_true: True labels/scores
        y_pred: Predicted scores
        k_percent: Percentage of top items to consider (default: 0.1 for top 10%)

    Returns:
        Recall score
    """
    n = len(y_true)
    k = max(1, int(n * k_percent))

    true_top_k_indices = np.argsort(y_true)[-k:]
    pred_top_k_indices = np.argsort(y_pred)[-k:]

    intersection = len(set(true_top_k_indices) & set(pred_top_k_indices))
    recall = intersection / k

    return recall


def calculate_all_metrics(y_true, y_pred, binary_threshold=0.0):
    """
    Calculate all metrics for predictions.

    Args:
        y_true: True labels/scores
        y_pred: Predicted scores
        binary_threshold: Threshold for converting scores to binary labels

    Returns:
        Dictionary with all metrics
    """
    results = {}

    try:
        # Spearman correlation
        spearman_corr, p_value = spearmanr(y_pred, y_true)
        results['spearman'] = spearman_corr
        results['p_value'] = p_value

        # Mean Squared Error
        mse = mean_squared_error(y_true, y_pred)
        results['mse'] = mse

        # AUC - using binary_threshold as cutoff for positive class
        binary_labels = (y_true > binary_threshold).astype(int)
        unique_classes = np.unique(binary_labels)

        if len(unique_classes) == 2:
            try:
                auc = roc_auc_score(binary_labels, y_pred)
                results['auc'] = auc
            except Exception:
                results['auc'] = np.nan
        else:
            results['auc'] = np.nan

        # MCC - using binary_threshold for both true and predicted
        pred_binary = (y_pred > binary_threshold).astype(int)

        if len(np.unique(binary_labels)) > 1 and len(np.unique(pred_binary)) > 1:
            try:
                mcc = matthews_corrcoef(binary_labels, pred_binary)
                results['mcc'] = mcc
            except Exception:
                results['mcc'] = np.nan
        else:
            results['mcc'] = np.nan

        # NDCG - normalize scores to [0, 1]
        true_min, true_max = y_true.min(), y_true.max()
        pred_min, pred_max = y_pred.min(), y_pred.max()

        if true_max > true_min and pred_max > pred_min:
            true_normalized = (y_true - true_min) / (true_max - true_min)
            pred_normalized = (y_pred - pred_min) / (pred_max - pred_min)

            ndcg_all = calculate_ndcg(true_normalized, pred_normalized)
            results['ndcg'] = ndcg_all

            # NDCG@10
            if len(y_true) >= 10:
                ndcg_10 = calculate_ndcg(true_normalized, pred_normalized, k=10)
                results['ndcg@10'] = ndcg_10
            else:
                results['ndcg@10'] = ndcg_all  # Use all if less than 10 samples
        else:
            results['ndcg'] = np.nan
            results['ndcg@10'] = np.nan

        # Top 10% Recall
        if len(y_true) >= 10:
            top10_recall = calculate_top_k_recall(y_true, y_pred, k_percent=0.1)
            results['top10_recall'] = top10_recall
        else:
            results['top10_recall'] = np.nan

    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        results = {
            'spearman': np.nan, 'p_value': np.nan, 'mse': np.nan,
            'auc': np.nan, 'mcc': np.nan, 'ndcg': np.nan,
            'ndcg@10': np.nan, 'top10_recall': np.nan
        }

    return results


def print_metrics(metrics, title="Evaluation Metrics"):
    """
    Pretty print evaluation metrics.

    Args:
        metrics: Dictionary of metrics
        title: Title to display
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    # Primary metrics
    print(f"Spearman Correlation: {metrics['spearman']:.4f}", end="")
    if 'p_value' in metrics and not np.isnan(metrics['p_value']):
        print(f" (p-value: {metrics['p_value']:.4e})")
    else:
        print()

    print(f"MSE: {metrics['mse']:.4f}")

    # Binary classification metrics
    if not np.isnan(metrics.get('auc', np.nan)):
        print(f"AUC: {metrics['auc']:.4f}")

    if not np.isnan(metrics.get('mcc', np.nan)):
        print(f"MCC: {metrics['mcc']:.4f}")

    # Ranking metrics
    if not np.isnan(metrics.get('ndcg', np.nan)):
        print(f"NDCG: {metrics['ndcg']:.4f}")

    if not np.isnan(metrics.get('ndcg@10', np.nan)):
        print(f"NDCG@10: {metrics['ndcg@10']:.4f}")

    if not np.isnan(metrics.get('top10_recall', np.nan)):
        print(f"Top 10% Recall: {metrics['top10_recall']:.4f}")

    print(f"{'='*60}\n")
