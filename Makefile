PYTHON ?= .venv/bin/python

.PHONY: run quickstart experiment plots check

run:
	$(PYTHON) main.py --config configs/baseline.yaml --rounds 2

quickstart:
	$(PYTHON) main.py --rounds 2 --experiment quickstart

experiment:
	$(PYTHON) -m experiments.experiment_runner --mode baseline_vs_aflf --num-runs 1

plots:
	$(PYTHON) -m visualization.plotter

check:
	$(PYTHON) system_check.py
