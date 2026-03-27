"""Communication cost estimation helpers for federated learning."""

from typing import Tuple

import torch.nn as nn


BYTES_PER_FP32 = 4
BYTES_PER_MB = 1024 * 1024


def estimate_model_size_bytes(model: nn.Module, bytes_per_parameter: int = BYTES_PER_FP32) -> int:
    """Estimate model payload size in bytes from parameter count."""
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return int(parameter_count * bytes_per_parameter)


def estimate_round_communication_bytes(
    model_size_bytes: int,
    num_participating_clients: int,
    include_downlink: bool = True,
    include_uplink: bool = True,
) -> Tuple[int, float]:
    """
    Estimate round communication from model size and participating clients.

    Downlink approximates server-to-client broadcast of the global model.
    Uplink approximates client-to-server upload of updated model parameters.
    """
    multipliers = int(include_downlink) + int(include_uplink)
    total_bytes = model_size_bytes * num_participating_clients * multipliers
    total_mb = total_bytes / float(BYTES_PER_MB)
    return int(total_bytes), total_mb
