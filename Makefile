PY := .venv/Scripts/python.exe
ifeq ($(OS),)
PY := .venv/bin/python
endif

.DEFAULT_GOAL := help
.PHONY: help venv install lint fmt type test test-all parse index ablation probe universe up down seed demo clean

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv:  ## Create the development virtualenv (Python 3.11)
	py -3.11 -m venv .venv || python3.11 -m venv .venv

install:  ## Install the package with dev extras, editable
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

lint:  ## ruff check + format check
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

fmt:  ## Apply ruff fixes and formatting
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

type:  ## mypy, strict
	$(PY) -m mypy src

test:  ## Unit tests. Network-marked tests are deselected.
	$(PY) -m pytest -m "not network"

test-all:  ## Everything, including the live SEC smoke test
	$(PY) -m pytest

universe:  ## Build config/universe.json from EDGAR company_tickers.json
	$(PY) -m scripts.build_universe

parse:  ## Parse + chunk an archived filing and verify invariants (ACCESSION=...)
	$(PY) -m scripts.parse_filing --accession $(ACCESSION)

index:  ## Parse, chunk, embed and index every landed filing
	$(PY) -m scripts.index_corpus

ablation:  ## Measure BM25 vs dense vs hybrid on the labelled query set
	$(PY) -m scripts.run_ablation

probe:  ## Block 1.3: can XBRL ground truth actually be resolved?
	$(PY) -m scripts.probe_xbrl_resolution

up:  ## Start the local stack
	docker compose -f deploy/docker-compose.yml up -d

down:  ## Stop the local stack
	docker compose -f deploy/docker-compose.yml down

seed:  ## Load the committed sample corpus
	$(PY) -m scripts.seed_sample

demo:  ## One search, one extraction, one scored evaluation
	$(PY) -m scripts.demo

clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
