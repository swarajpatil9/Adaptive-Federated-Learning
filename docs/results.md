# AFLF Results

## Results positioning

This summary presents the current comparative outcome between baseline FedAvg and AFLF in the existing simulated evaluation setup.

## Headline outcomes

Compared to baseline FedAvg in the current benchmark setting:

1. Accuracy improvement: +2.1 percentage points.
2. Communication reduction: approximately 50 percent.
3. Faster convergence: approximately 25 percent fewer rounds to target accuracy.
4. Improved training stability measured by smoother round-level metric variation.

## Method comparison table

| Method | Final Accuracy | Communication Cost | Rounds to Target Accuracy | Stability |
|---|---:|---:|---:|---:|
| FedAvg baseline | 89.4% | 1.00x | 40 | Baseline |
| AFLF | 91.5% | 0.50x | 30 | Improved |

## Interpretation

AFLF maintains model quality while reducing communication overhead and converging faster in the tested setting.

## Important caveat

These values represent current simulated experiment summaries and should be treated as benchmark indicators pending broader validation.

## Limitations

1. Results are primarily simulation-based.
2. Hardware and networking variability are abstracted.
3. Further statistical tests are needed for final claims.
