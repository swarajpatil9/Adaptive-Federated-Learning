# AFLF Architecture

## Objective

This document describes the architecture of the Adaptive Federated Learning Framework and clarifies component responsibilities for research reproducibility and extension.

## System Components

## Client layer

The client layer performs local model training on private datasets, computes local metrics, and returns model updates with metadata such as sample count, local performance, and communication statistics.

## Server layer

The server layer coordinates global rounds, selects clients, dispatches global model parameters, aggregates client updates, and tracks round-level outcomes.

## Selection module

The selection module supports both random and dynamic policies. The dynamic strategy scores clients based on multiple factors and improves participation balance and convergence behavior under heterogeneous client conditions.

## Privacy module

The privacy module provides clipping and noise injection paths for differential privacy research. It records privacy metadata and integrates with client update boundaries.

## Optimization module

The optimization module controls adaptive learning-rate behavior across rounds and records adjustment context for analysis.

## Communication module

The communication module supports compression-oriented transport behavior, including update-level size tracking and compression metadata.

## Experiment pipeline

The experiment pipeline covers configuration loading, validation, deterministic seeding, federated training execution, metric export, and summary logging.

## Visualization pipeline

The visualization pipeline consumes experiment metrics and generates comparison, training, and communication plots.

## Data and control flow summary

1. Load and validate configuration.
2. Initialize deterministic runtime state.
3. Construct data module, clients, model, and server orchestration.
4. Execute federated rounds.
5. Aggregate updates and evaluate global state.
6. Export metrics and logs.
7. Generate visual artifacts from exported metrics.

## Limitations

1. Primary evaluations are based on simulated federated clients.
2. Cross-device systems behavior has limited empirical coverage.
3. Production-scale secure aggregation is not yet fully integrated.
