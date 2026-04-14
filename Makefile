.PHONY: help install dev test lint format typecheck security smoke clean docker docker-up docker-down index

PYTHON  ?= python3.11
VENV    ?= venv
PIP     := $(VENV)/bin/pip
PY      := $(VENV)/bin/python
PYTEST  := $(VENV)/bin/pytest
RUFF    := $(VENV)/bin/ruff
MYPY    := $(VENV)/bin/mypy
BANDIT  := $(VENV)/bin/bandit

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ── Setup ─────────────────────────────────────

install: ## Create venv and install production deps
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r verifactai/requirements.txt
	$(PY) -m spacy download en_core_web_sm

dev: install ## Install dev dependencies (linting, testing, etc.)
	$(PIP) install pytest ruff mypy pre-commit bandit
	$(VENV)/bin/pre-commit install

# ── Quality ───────────────────────────────────

test: ## Run unit tests
	$(PYTEST) verifactai/tests/ -v

smoke: ## Run full smoke test (requires Ollama + index)
	cd verifactai && ../$(PY) smoke_test.py

lint: ## Lint with ruff
	$(RUFF) check verifactai/

format: ## Auto-format with ruff
	$(RUFF) format verifactai/
	$(RUFF) check --fix verifactai/

typecheck: ## Static type check with mypy
	$(MYPY) verifactai/core/ verifactai/config.py

security: ## Security scan with bandit
	$(BANDIT) -r verifactai/core/ verifactai/utils/ -q

check: lint typecheck test ## Run all quality checks

# ── Data ──────────────────────────────────────

index: ## Build FAISS knowledge index (full Wikipedia)
	cd verifactai && ../$(PY) data/build_index.py --wiki-only

index-dev: ## Build small dev index (5K articles)
	cd verifactai && ../$(PY) data/build_index.py --wiki-only --max-articles 5000

# ── Run ───────────────────────────────────────

dashboard: ## Launch Streamlit dashboard
	cd verifactai && ../$(PY) -m streamlit run app.py

server: ## Launch overlay API server (for browser extension)
	cd verifactai && ../$(PY) overlay_server.py

# ── Eval ──────────────────────────────────────

benchmark: ## Run full benchmark suite
	cd verifactai && PYTHONPATH=. ../$(PY) evaluation/evaluate.py --benchmark all --max-samples 30

benchmark-quick: ## Run sanity check only
	cd verifactai && PYTHONPATH=. ../$(PY) evaluation/evaluate.py --benchmark sanity

# ── Docker ────────────────────────────────────

docker: ## Build Docker image
	docker build -t verifact-ai .

docker-up: ## Start with docker-compose
	docker-compose up -d

docker-down: ## Stop docker-compose
	docker-compose down

# ── Cleanup ───────────────────────────────────

clean: ## Remove caches, logs, build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.log" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/
