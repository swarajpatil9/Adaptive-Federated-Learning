# AFLF Experiments

## Experimental scope

Experiments compare baseline FedAvg and AFLF-enhanced pipelines under equivalent training budgets and consistent seed controls.

## Core experiment questions

1. Does AFLF improve convergence behavior relative to baseline FedAvg?
2. Does AFLF reduce communication cost while preserving model quality?
3. Does AFLF improve round-level stability under client heterogeneity?

## Typical configuration dimensions

1. Number of clients and clients-per-round.
2. Data partition strategy.
3. Selection policy type and weighting.
4. Communication compression settings.
5. Adaptive learning-rate settings.
6. Privacy settings.

## Metrics tracked

1. Global accuracy and global loss.
2. Round participation and failure rates.
3. Convergence rounds to target quality.
4. Communication volume and compression ratio.
5. Stability trends across rounds.

## Experiment summary

Observed experiment trends indicate that AFLF improves communication efficiency and convergence behavior while maintaining competitive model performance.

## Limitations

1. Most runs use simulated clients.
2. Some ablation paths have limited sample size.
3. Additional cross-dataset validation is needed.

## Reproducibility notes

1. Use fixed seed and pinned dependencies.
2. Validate environment before runs.
3. Keep config files under version control.
4. Record experiment name for all runs.
