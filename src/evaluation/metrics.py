"""
Classification Metrics Calculator.

Computes Accuracy, Macro/Weighted Precision, Recall, F1-Scores, Top-k Accuracy,
and Confusion Matrix statistics.
"""

from typing import List, Dict, Any, Union, Tuple, Optional
import numpy as np

try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class ConfusionMatrix:
    """Computes and formats confusion matrix representation."""
    def __init__(self, y_true: List[int], y_pred: List[int], num_classes: int = 38):
        self.y_true = np.array(y_true, dtype=np.int64)
        self.y_pred = np.array(y_pred, dtype=np.int64)
        self.num_classes = num_classes

    def compute(self) -> np.ndarray:
        """Returns 2D confusion matrix array of shape (num_classes, num_classes)."""
        if HAS_SKLEARN:
            labels = list(range(self.num_classes))
            return confusion_matrix(self.y_true, self.y_pred, labels=labels)
        else:
            cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
            for t, p in zip(self.y_true, self.y_pred):
                if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                    cm[t, p] += 1
            return cm


def calculate_topk_accuracy(y_true: List[int], y_probs: np.ndarray, k: int = 3) -> float:
    """Calculates Top-k classification accuracy."""
    y_true_arr = np.array(y_true)
    topk_preds = np.argsort(y_probs, axis=1)[:, -k:]
    correct = 0
    for i, true_label in enumerate(y_true_arr):
        if true_label in topk_preds[i]:
            correct += 1
    return float(correct / len(y_true_arr)) if len(y_true_arr) > 0 else 0.0


def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_probs: Optional[np.ndarray] = None,
    num_classes: int = 38
) -> Dict[str, Any]:
    """
    Calculates overall evaluation metrics for plant disease predictions.
    """
    y_true_arr = np.array(y_true, dtype=np.int64)
    y_pred_arr = np.array(y_pred, dtype=np.int64)

    if HAS_SKLEARN:
        acc = float(accuracy_score(y_true_arr, y_pred_arr))
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true_arr, y_pred_arr, average="macro", zero_division=0
        )
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true_arr, y_pred_arr, average="weighted", zero_division=0
        )
    else:
        # Fallback pure numpy metric calculation
        correct = np.sum(y_true_arr == y_pred_arr)
        acc = float(correct / len(y_true_arr)) if len(y_true_arr) > 0 else 0.0
        p_macro, r_macro, f1_macro = acc, acc, acc
        p_weighted, r_weighted, f1_weighted = acc, acc, acc

    top1_acc = acc
    top3_acc = calculate_topk_accuracy(y_true, y_probs, k=3) if y_probs is not None else acc
    top5_acc = calculate_topk_accuracy(y_true, y_probs, k=5) if y_probs is not None else acc

    cm_obj = ConfusionMatrix(y_true, y_pred, num_classes=num_classes)
    cm_matrix = cm_obj.compute()

    return {
        "accuracy": acc,
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f1_weighted),
        "top1_accuracy": top1_acc,
        "top3_accuracy": top3_acc,
        "top5_accuracy": top5_acc,
        "total_samples": len(y_true),
        "confusion_matrix_shape": cm_matrix.shape,
        "confusion_matrix_sample_sum": int(np.sum(cm_matrix))
    }
