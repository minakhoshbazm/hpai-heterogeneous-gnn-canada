import os
import json
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    f1_score,
    accuracy_score
)

def calculate_metrics(y_true, y_pred, y_prob):
    """
    Computes standard classification evaluation metrics for HPAI risk prediction.
    Handles potential edge cases where y_true contains only one class.
    
    Parameters:
    - y_true: Ground truth binary labels (0 or 1)
    - y_pred: Predicted binary labels (0 or 1)
    - y_prob: Predicted probability of positive class [0, 1]
    
    Returns:
    - Dict of evaluation metrics
    """
    # Ensure numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    # Calculate confusion matrix
    # Format is: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Sensitivity (Recall or True Positive Rate)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # F1-score
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # ROC-AUC (Requires probability)
    try:
        # Check if there are both positive and negative classes in y_true
        if len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, y_prob)
        else:
            roc_auc = 0.5  # default if only one class is present
    except Exception:
        roc_auc = 0.5
        
    # PR-AUC (Precision-Recall Area Under Curve, highly recommended for imbalanced datasets)
    try:
        if len(np.unique(y_true)) > 1:
            precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(recalls, precisions)
        else:
            pr_auc = 0.0
    except Exception:
        pr_auc = 0.0
        
    return {
        'total_samples': int(len(y_true)),
        'pos_samples': int(np.sum(y_true)),
        'neg_samples': int(len(y_true) - np.sum(y_true)),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc)
    }

def print_metrics_summary(metrics, title="Evaluation Summary"):
    """Prints a formatted console summary of the metrics."""
    print("=" * 50)
    print(f" {title} ")
    print("=" * 50)
    print(f"Total Samples:   {metrics['total_samples']} (Pos: {metrics['pos_samples']}, Neg: {metrics['neg_samples']})")
    print(f"Confusion Matrix: TP={metrics['true_positives']}, FP={metrics['false_positives']}, TN={metrics['true_negatives']}, FN={metrics['false_negatives']}")
    print("-" * 50)
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Precision:       {metrics['precision']:.4f}")
    print(f"Sensitivity:     {metrics['sensitivity']:.4f} (Recall / TPR)")
    print(f"Specificity:     {metrics['specificity']:.4f} (TNR)")
    print(f"F1-score:        {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:         {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:          {metrics['pr_auc']:.4f}")
    print("=" * 50)

def save_metrics_report(metrics, file_path):
    """Saves the metrics dictionary as a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics report to: {file_path}")
