"""Classification metric utilities."""

from typing import Dict, Optional

import torch


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return a safe division result, defaulting to 0.0 for zero denominator."""
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy from predicted class indices and target labels."""
    if targets.numel() == 0:
        return 0.0
    correct = (predictions == targets).sum().item()
    return _safe_divide(float(correct), float(targets.numel()))


def compute_classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None,
    include_optional: bool = True,
) -> Dict[str, float]:
    """
    Compute accuracy and optional macro precision/recall/F1 from class indices.

    This function is dependency-free (no sklearn) to keep the framework lightweight.
    """
    if targets.numel() == 0:
        metrics: Dict[str, float] = {'accuracy': 0.0}
        if include_optional:
            metrics.update({'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0})
        return metrics

    accuracy = compute_accuracy(predictions, targets)
    metrics = {'accuracy': accuracy}

    if not include_optional:
        return metrics

    if num_classes is None:
        max_label = max(int(predictions.max().item()), int(targets.max().item()))
        num_classes = max_label + 1

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64)
    for true_label, pred_label in zip(targets.view(-1), predictions.view(-1)):
        confusion[int(true_label.item()), int(pred_label.item())] += 1.0

    precision_values = []
    recall_values = []
    f1_values = []

    for class_idx in range(num_classes):
        tp = float(confusion[class_idx, class_idx].item())
        fp = float(confusion[:, class_idx].sum().item() - tp)
        fn = float(confusion[class_idx, :].sum().item() - tp)

        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2.0 * precision * recall, precision + recall)

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

    metrics['precision'] = sum(precision_values) / len(precision_values)
    metrics['recall'] = sum(recall_values) / len(recall_values)
    metrics['f1_score'] = sum(f1_values) / len(f1_values)

    return metrics
