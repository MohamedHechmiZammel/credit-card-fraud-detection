# Runtimes below are measured on 4 CPU cores. Use PYTHON=.venv/bin/python make ... if the venv is not active.
PYTHON ?= python

.PHONY: all data reproduce improve cv cost report test lint space

all: data reproduce improve cv cost report test space  ## about 40 minutes, almost all of it in `reproduce`

data:       ## download and verify the dataset (about 1 minute, 150 MB)
	$(PYTHON) scripts/download_data.py

reproduce:  ## original 2024 grid searches, scoring=accuracy (about 12 min for dt plus 21 min for xgb)
	$(PYTHON) scripts/reproduce.py dt
	$(PYTHON) scripts/reproduce.py xgb

improve:    ## stratified split, baselines, class weighting, calibration, figures (about 1 minute)
	$(PYTHON) scripts/train_improved.py

cv:         ## stratified 5-fold cross-validation, writes reports/cv.json (about 1.5 minutes)
	$(PYTHON) scripts/cross_validate.py

cost:       ## cost-sensitive thresholds and amount-weighted training, writes reports/cost.json (about 15 seconds)
	$(PYTHON) scripts/cost_analysis.py

report:     ## rebuild reports/results.md from the JSON files (instant)
	$(PYTHON) scripts/make_report.py

test:       ## pytest, uses bundled and synthetic data only (a few seconds)
	$(PYTHON) -m pytest

lint:       ## ruff check
	$(PYTHON) -m ruff check

space:      ## rebuild the Hugging Face Space folder from app/ and models/ (instant, uploads nothing)
	$(PYTHON) scripts/build_space.py
