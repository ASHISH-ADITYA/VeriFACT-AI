#!/bin/zsh
set -euo pipefail

ROOT="/Users/adityaashish/Desktop/ENGINEERING_PROJECT_II/verifactai"
PY1="/Users/adityaashish/Desktop/ENGINEERING_PROJECT_II/.venv/bin/python"
PY2="/Users/adityaashish/Desktop/ENGINEERING_PROJECT_II/venv/bin/python"
NATIVE_LOG_OUT="/tmp/verifact_native_app.out.log"
NATIVE_LOG_ERR="/tmp/verifact_native_app.err.log"

cd "$ROOT"

if [[ -x "$PY1" ]]; then
	PY="$PY1"
elif [[ -x "$PY2" ]]; then
	PY="$PY2"
else
	osascript -e 'display notification "Python environment not found." with title "VeriFact Desktop"'
	exit 1
fi

# Start a native window shell that boots services and renders dashboard as desktop app.
nohup "$PY" desktop/verifact_native_app.py > "$NATIVE_LOG_OUT" 2> "$NATIVE_LOG_ERR" &
