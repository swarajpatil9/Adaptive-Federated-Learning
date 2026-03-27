"""Loss-related metric helpers."""


def compute_average_loss(total_loss: float, total_samples: int) -> float:
    """Compute average loss with a safe zero-sample guard."""
    if total_samples <= 0:
        return 0.0
    return total_loss / float(total_samples)
