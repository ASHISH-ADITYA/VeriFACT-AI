#!/usr/bin/env bash
# VeriFACT AI — One-command cross-platform setup
# Usage: bash scripts/setup.sh
set -euo pipefail

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-venv}"

echo "================================================"
echo "  VeriFACT AI Setup"
echo "================================================"

# ── Check Python version ──────────────────────
echo ""
echo "[1/6] Checking Python..."
PY_VER=$($PYTHON --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "  ERROR: Python 3.10+ required (found $PY_VER)"
    echo "  Install: brew install python@3.11  OR  apt install python3.11"
    exit 1
fi
echo "  Python $PY_VER OK"

# ── Create virtual environment ────────────────
echo ""
echo "[2/6] Creating virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    $PYTHON -m venv "$VENV_DIR"
    echo "  Created $VENV_DIR"
else
    echo "  $VENV_DIR already exists, reusing"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel -q

# ── Install dependencies ──────────────────────
echo ""
echo "[3/6] Installing dependencies..."
pip install -r verifactai/requirements.txt -q
pip install pytest ruff -q
python -m spacy download en_core_web_sm -q
echo "  Dependencies installed"

# ── Create .env if missing ────────────────────
echo ""
echo "[4/6] Checking environment config..."
if [ ! -f verifactai/.env ]; then
    cp verifactai/.env.example verifactai/.env
    echo "  Created verifactai/.env from template"
else
    echo "  verifactai/.env already exists"
fi

# ── Check Ollama ──────────────────────────────
echo ""
echo "[5/6] Checking Ollama..."
if command -v ollama &>/dev/null; then
    echo "  Ollama found"
    if curl -sf http://localhost:11434/api/tags &>/dev/null; then
        echo "  Ollama server running"
    else
        echo "  WARNING: Ollama installed but not running. Start with: ollama serve"
    fi
else
    echo "  WARNING: Ollama not installed."
    echo "  Install: https://ollama.com"
    echo "  Then: ollama pull llama3.1:8b"
fi

# ── Run tests ─────────────────────────────────
echo ""
echo "[6/6] Running tests..."
python -m pytest verifactai/tests/ -q --tb=line 2>&1 | tail -3

echo ""
echo "================================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    make index-dev     # Build knowledge index (5 min)"
echo "    make smoke         # Verify everything works"
echo "    make dashboard     # Launch at http://localhost:8501"
echo "================================================"
