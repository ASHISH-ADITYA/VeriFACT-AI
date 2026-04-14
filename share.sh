#!/usr/bin/env bash
# VeriFACT AI — One-command share script
# Starts backend + creates public tunnel + shows shareable URL
set -euo pipefail

echo "================================================"
echo "  VeriFACT AI — Share Mode"
echo "================================================"
echo ""

cd "$(dirname "$0")"

# ── Check backend is running ──────────────────────
echo "[1/3] Checking backend..."
if curl -sf http://localhost:8765/health > /dev/null 2>&1; then
    echo "  API server already running on :8765"
else
    echo "  Starting API server..."
    cd verifactai
    ../venv/bin/python overlay_server.py &
    API_PID=$!
    cd ..
    sleep 5
    if ! curl -sf http://localhost:8765/health > /dev/null 2>&1; then
        echo "  ERROR: API server failed to start"
        exit 1
    fi
    echo "  API server started (PID $API_PID)"
fi

# ── Create public tunnel ─────────────────────────
echo ""
echo "[2/3] Creating public tunnel (Cloudflare, free, no account needed)..."
echo "  This gives you a public URL that anyone can access."
echo ""

# Start tunnel in background, capture the URL
cloudflared tunnel --url http://localhost:8765 2>&1 &
TUNNEL_PID=$!

# Wait for tunnel to produce URL
sleep 5
echo ""
echo "[3/3] Getting your shareable URL..."
sleep 3

echo ""
echo "================================================"
echo "  READY TO SHARE"
echo "================================================"
echo ""
echo "  Look above for a URL like:"
echo "    https://xxxxx-xxxx-xxxx.trycloudflare.com"
echo ""
echo "  Anyone can open that URL and use the API."
echo "  Send them the URL — it works on any device."
echo ""
echo "  To test: open in browser:"
echo "    https://YOUR-URL/health"
echo ""
echo "  To stop: press Ctrl+C"
echo "================================================"

# Wait for Ctrl+C
trap "echo ''; echo 'Stopping tunnel...'; kill $TUNNEL_PID 2>/dev/null; echo 'Done.'; exit 0" INT TERM
wait
