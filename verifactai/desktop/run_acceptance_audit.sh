#!/bin/zsh
set -euo pipefail

ROOT="/Users/adityaashish/Desktop/ENGINEERING_PROJECT_II/verifactai"
PY="/Users/adityaashish/Desktop/ENGINEERING_PROJECT_II/venv/bin/python"
OUT="$ROOT/assets/acceptance_audit_report.txt"

mkdir -p "$ROOT/assets"

echo "VeriFact Acceptance Audit" > "$OUT"
echo "Generated: $(date)" >> "$OUT"
echo "" >> "$OUT"

pass() { echo "[PASS] $1" | tee -a "$OUT"; }
fail() { echo "[FAIL] $1" | tee -a "$OUT"; }
info() { echo "[INFO] $1" | tee -a "$OUT"; }

# 1. Core files
[[ -f "$ROOT/overlay_server.py" ]] && pass "Overlay server present" || fail "Overlay server missing"
[[ -f "$ROOT/desktop/verifact_desktop.py" ]] && pass "Desktop app present" || fail "Desktop app missing"
[[ -f "$ROOT/integrations/web_beacon_extension/manifest.json" ]] && pass "Extension manifest present" || fail "Extension manifest missing"

# 2. Python compile
if "$PY" -m py_compile "$ROOT/overlay_server.py" "$ROOT/desktop/verifact_desktop.py"; then
  pass "Python modules compile"
else
  fail "Python compile errors"
fi

# 3. Extension host coverage
if grep -q "gemini.google.com" "$ROOT/integrations/web_beacon_extension/manifest.json"; then
  pass "Gemini host support configured"
else
  fail "Gemini host support missing"
fi
if grep -q "claude.ai" "$ROOT/integrations/web_beacon_extension/manifest.json" && grep -q "chatgpt.com" "$ROOT/integrations/web_beacon_extension/manifest.json"; then
  pass "ChatGPT + Claude host support configured"
else
  fail "ChatGPT/Claude host support incomplete"
fi

# 4. Mid-right beacon style
if grep -q "top: 50%" "$ROOT/integrations/web_beacon_extension/styles.css" && grep -q "right: 18px" "$ROOT/integrations/web_beacon_extension/styles.css"; then
  pass "Beacon mid-right positioning configured"
else
  fail "Beacon positioning not mid-right"
fi

# 5. Start overlay server and run health + sample analyze
SERVER_LOG="$ROOT/assets/overlay_server_test.log"
"$PY" "$ROOT/overlay_server.py" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in {1..20}; do
  if curl -sS -m 2 http://127.0.0.1:8765/health >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if curl -sS -m 3 http://127.0.0.1:8765/health >/dev/null; then
  pass "Overlay health endpoint reachable"
else
  fail "Overlay health endpoint unreachable"
fi

ANALYZE_PAYLOAD='{"text":"Albert Einstein invented the telephone in 1920.","top_claims":3}'
if curl -sS -m 60 -H "Content-Type: application/json" -d "$ANALYZE_PAYLOAD" http://127.0.0.1:8765/analyze > "$ROOT/assets/analyze_sample_response.json"; then
  pass "Overlay analyze endpoint returns response"
else
  fail "Overlay analyze endpoint failed"
fi

kill "$SERVER_PID" >/dev/null 2>&1 || true

# 6. Report consistency quick checks
if grep -q "\[N\]" "$ROOT/assets/CAPSTONE_REPORT.md" || grep -q "\[M\]" "$ROOT/assets/CAPSTONE_REPORT.md" || grep -q "\[V\]" "$ROOT/assets/CAPSTONE_REPORT.md"; then
  fail "CAPSTONE_REPORT has unresolved placeholders"
else
  pass "CAPSTONE_REPORT has no [N]/[M]/[V] placeholders"
fi

# 7. Index build status snapshot
if pgrep -f "data/build_index.py --wiki-only" >/dev/null; then
  info "Index build process is currently running"
else
  info "Index build process is not running"
fi
if [[ -f "$ROOT/data/index/knowledge.index" ]]; then
  info "Index file: $(ls -lh "$ROOT/data/index/knowledge.index" | awk '{print $5, $6, $7, $8}')"
fi
if [[ -f "$ROOT/data/metadata/chunks.jsonl" ]]; then
  info "Metadata lines: $(wc -l < "$ROOT/data/metadata/chunks.jsonl")"
fi

echo "" >> "$OUT"
echo "Audit complete." | tee -a "$OUT"
