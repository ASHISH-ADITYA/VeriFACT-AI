#!/bin/zsh
set -euo pipefail

pkill -f 'overlay_server.py' || true
pkill -f 'streamlit run app.py --server.port 8501' || true
pkill -f 'desktop/verifact_native_app.py' || true
osascript -e 'display notification "Analyzer and dashboard stopped." with title "VeriFact Desktop"'
