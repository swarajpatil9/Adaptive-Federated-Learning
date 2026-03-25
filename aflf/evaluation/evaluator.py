"""
Global model evaluation utilities.

Provides centralized evaluation of the global model on the server-side
held-out dataset.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..client.client_utils import get_criterion, get_device


class GlobalEvaluator:
    """
    Evaluate a global model on a centralized test loader.

    The evaluator is intentionally independent from orchestration and aggregation
    so it can be reused by different training controllers.
    """

    def __init__(self, test_loader: DataLoader, device: Optional[str] = None):
        """
        Initialize evaluator.

        Args:
            test_loader: Centralized evaluation dataloader
            device: Evaluation device (auto-detect if None)
        """
        self.test_loader = test_loader
        self.device = get_device(device)

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        criterion_name: str = 'cross_entropy',
    ) -> Dict[str, float]:
        """
        Evaluate global model on centralized test set.

        Args:
            model: Global model to evaluate
            criterion_name: Loss function name

        Returns:
            Dictionary with global_loss, global_accuracy, and num_eval_samples
        """
        criterion = get_criterion(criterion_name)

        was_training = model.training
        model = model.to(self.device)
        model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for data, target in self.test_loader:
            data = data.to(self.device)
            target = target.to(self.device)

            output = model(data)
            loss = criterion(output, target)

            batch_size = target.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = output.argmax(dim=1)
            total_correct += (predictions == target).sum().item()

        if was_training:
            model.train()

        if total_samples == 0:
            return {
                'global_loss': 0.0,
                'global_accuracy': 0.0,
                'num_eval_samples': 0,
            }

        return {
            'global_loss': total_loss / total_samples,
            'global_accuracy': total_correct / total_samples,
            'num_eval_samples': total_samples,
        }
