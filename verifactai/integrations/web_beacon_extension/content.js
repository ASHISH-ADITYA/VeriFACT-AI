/**
 * VeriFACT AI — Browser Extension Content Script
 *
 * Color spec:
 *   Blue   = safe (90-100% correctness)
 *   Purple = minor hallucinations (70-89%)
 *   Orange = moderate / needs fact check (50-69%)
 *   Red    = hallucinations detected (<50%)
 *
 * Scanning: MutationObserver + debounced fallback polling.
 */

const PLATFORM = location.hostname.includes("claude")
  ? "claude"
  : location.hostname.includes("gemini")
    ? "gemini"
    : "chatgpt";

const DEBOUNCE_MS = 2000;
const POLL_FALLBACK_MS = 5000;
const CONVERSATION_WINDOW = 6;

let lastFingerprint = "";
let panelVisible = false;
let panelEl = null;
let latestTargetNode = null;
let debounceTimer = null;
let pulseTimer = null;

// ── Beacon creation ──────────────────────────────────

const beacon = document.createElement("button");
beacon.className = "verifact-beacon idle";
beacon.title = "VeriFACT AI";
beacon.innerHTML = '<span class="verifact-beacon-label">VF</span><span class="verifact-beacon-status">Ready</span>';
document.body.appendChild(beacon);

const ticker = document.createElement("div");
ticker.className = "verifact-ticker";
document.body.appendChild(ticker);

beacon.addEventListener("click", () => {
  panelVisible = !panelVisible;
  if (!panelVisible && panelEl) {
    panelEl.remove();
    panelEl = null;
  }
});

// ── Beacon state machine ─────────────────────────────

function setBeacon(state, statusText) {
  beacon.className = "verifact-beacon " + state;
  const statusEl = beacon.querySelector(".verifact-beacon-status");
  if (statusEl && statusText) {
    statusEl.textContent = statusText;
  }
}

function scoreToBeaconState(score) {
  if (score >= 90) return { state: "safe", text: "Safe" };
  if (score >= 70) return { state: "minor", text: "Minor" };
  if (score >= 50) return { state: "moderate", text: "Check" };
  return { state: "hallucinated", text: "Alert" };
}

// ── Alert ticker ─────────────────────────────────────

function alertClass(category) {
  const c = String(category || "").toLowerCase();
  if (c.includes("halluc")) return "hallucination";
  if (c.includes("bias")) return "bias";
  if (c.includes("red")) return "red-flag";
  return "neutral";
}

function pulseTicker(alerts) {
  if (pulseTimer) { clearTimeout(pulseTimer); pulseTimer = null; }

  if (!alerts || !alerts.length) {
    ticker.innerHTML = '<div class="verifact-chip neutral">No issues detected.</div>';
    ticker.classList.add("show");
    pulseTimer = setTimeout(() => ticker.classList.remove("show"), 3500);
    return;
  }

  ticker.innerHTML = alerts.slice(0, 2).map((a) => {
    const cls = alertClass(a.category);
    const msg = String(a.message || "Issue detected").replace(/</g, "&lt;");
    return `<div class="verifact-chip ${cls}">${msg}</div>`;
  }).join("");
  ticker.classList.add("show");
  pulseTimer = setTimeout(() => ticker.classList.remove("show"), 5000);
}

// ── DOM scanning ─────────────────────────────────────

function getAssistantMessages() {
  if (PLATFORM === "chatgpt") {
    const nodes = Array.from(document.querySelectorAll("[data-message-author-role='assistant']"));
    const pairs = nodes.map((n) => ({ text: n.innerText?.trim(), node: n })).filter((x) => x.text);
    latestTargetNode = pairs.length ? pairs[pairs.length - 1].node : null;
    return pairs.map((p) => p.text);
  }

  if (PLATFORM === "gemini") {
    const nodes = Array.from(
      document.querySelectorAll("message-content .markdown, model-response .markdown, model-response")
    );
    const pairs = nodes.map((n) => ({ text: n.innerText?.trim(), node: n })).filter((x) => x.text && x.text.length > 40);
    latestTargetNode = pairs.length ? pairs[pairs.length - 1].node : null;
    return pairs.map((p) => p.text).slice(-8);
  }

  // Claude
  const nodes = Array.from(
    document.querySelectorAll("div[data-is-streaming], div.prose, div.font-claude-message")
  );
  const pairs = nodes.map((n) => ({ text: n.innerText?.trim(), node: n })).filter((x) => x.text && x.text.length > 40).slice(-8);
  latestTargetNode = pairs.length ? pairs[pairs.length - 1].node : null;
  return pairs.map((p) => p.text);
}

function fingerprint(text) {
  return `${text.length}:${text.slice(0, 80)}:${text.slice(-80)}`;
}

// ── Backend call ─────────────────────────────────────

async function callAnalyzer(text) {
  const store = await chrome.storage.local.get(["verifactApiUrl", "verifactEnabled", "verifactApiToken"]);
  if (store.verifactEnabled === false) return null;

  const apiUrl = store.verifactApiUrl || "https://adiashish-verifact-ai.hf.space/analyze";
  const apiToken = store.verifactApiToken || "";

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 45000); // 45s timeout

  try {
    const response = await fetch(apiUrl, {
      method: "POST",
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...(apiToken ? { "X-VeriFact-Token": apiToken } : {}),
      },
      body: JSON.stringify({
        text,
        source: PLATFORM,
        mode: "fast",
        include_claims: true,
        top_claims: 6,
      }),
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  } finally {
    clearTimeout(timeout);
  }
}

// ── Claim rendering ──────────────────────────────────

function claimClass(v) {
  const x = (v || "").toLowerCase();
  if (x.includes("contrad")) return "contradicted";
  if (x.includes("support")) return "supported";
  if (x.includes("unver")) return "unverifiable";
  return "";
}

function renderPanel(result) {
  const panel = ensurePanel();
  if (!panel) return;

  const claims = (result.flags || []).map((c) => {
    const confPct = Math.round((c.confidence || 0) * 100);
    return `
      <div class="verifact-claim ${claimClass(c.verdict)}">
        <div><strong>${c.verdict}</strong> (${confPct}%)</div>
        <div>${(c.claim || "").replace(/</g, "&lt;")}</div>
        ${c.evidence ? `<div style="margin-top:4px;opacity:0.8;font-size:11px">${String(c.evidence).substring(0, 150).replace(/</g, "&lt;")}...</div>` : ""}
      </div>`;
  }).join("");

  panel.innerHTML = `
    <h3>VeriFACT AI (${PLATFORM})</h3>
    <div class="verifact-kpi">
      <div class="verifact-kpi-item">
        <div class="verifact-kpi-value">${Math.round(result.factuality_score || 0)}</div>
        <div class="verifact-kpi-label">Score</div>
      </div>
      <div class="verifact-kpi-item">
        <div class="verifact-kpi-value">${Math.round((result.overall_confidence || 0) * 100)}%</div>
        <div class="verifact-kpi-label">Confidence</div>
      </div>
      <div class="verifact-kpi-item">
        <div class="verifact-kpi-value">${result.total_claims || 0}</div>
        <div class="verifact-kpi-label">Claims</div>
      </div>
    </div>
    <div style="margin-bottom:8px"><strong>Summary:</strong> ${result.summary || "—"}</div>
    <hr style="border-color:rgba(255,255,255,0.14)"/>
    ${claims || "<div>No claims extracted</div>"}
  `;
}

function ensurePanel() {
  if (!panelVisible) return null;
  if (!panelEl) {
    panelEl = document.createElement("div");
    panelEl.className = "verifact-panel";
    panelEl.innerHTML = "<h3>VeriFACT AI</h3><div>Waiting for response...</div>";
    document.body.appendChild(panelEl);
  }
  return panelEl;
}

// ── Inline highlighting ──────────────────────────────

function highlightClaims(node, flags) {
  if (!node || !flags || node.dataset.verifactHighlighted === "1") return;
  const flagged = flags.filter((f) => {
    const v = (f.verdict || "").toLowerCase();
    return v.includes("contrad") || v.includes("unver");
  });
  if (!flagged.length) return;

  const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
  const textNodes = [];
  let current;
  while ((current = walker.nextNode())) {
    if (current.nodeValue?.trim()) textNodes.push(current);
  }

  for (const f of flagged) {
    const claim = (f.claim || "").trim();
    if (claim.length < 16) continue;
    const probe = claim.slice(0, Math.min(48, claim.length));
    for (const tn of textNodes) {
      const idx = tn.nodeValue.toLowerCase().indexOf(probe.toLowerCase());
      if (idx === -1) continue;
      const span = document.createElement("span");
      span.className = "verifact-inline-flag";
      span.title = `${f.verdict} (${Math.round((f.confidence || 0) * 100)}%)`;
      const full = tn.nodeValue;
      const frag = document.createDocumentFragment();
      if (full.slice(0, idx)) frag.appendChild(document.createTextNode(full.slice(0, idx)));
      span.textContent = full.slice(idx, idx + probe.length);
      frag.appendChild(span);
      if (full.slice(idx + probe.length)) frag.appendChild(document.createTextNode(full.slice(idx + probe.length)));
      tn.parentNode.replaceChild(frag, tn);
      node.dataset.verifactHighlighted = "1";
      break;
    }
  }
}

// ── Main scan loop ───────────────────────────────────

async function scan() {
  const messages = getAssistantMessages();
  if (!messages.length) return;

  const latest = messages[messages.length - 1];
  if (!latest || latest.length < 40) return;

  const convoText = messages.slice(-CONVERSATION_WINDOW).join("\n\n");
  const fp = fingerprint(convoText);
  if (fp === lastFingerprint) return;

  lastFingerprint = fp;
  setBeacon("analyzing", "Checking...");

  try {
    const result = await callAnalyzer(convoText);
    if (!result) { setBeacon("idle", "Ready"); return; }

    // No claims = nothing to flag = safe
    if (result.total_claims === 0) { setBeacon("safe", "Safe"); return; }

    const { state, text } = scoreToBeaconState(result.factuality_score || 0);
    setBeacon(state, text);
    pulseTicker(result.alerts || []);
    renderPanel(result);
    highlightClaims(latestTargetNode, result.flags || []);
  } catch (err) {
    const isTimeout = err.name === "AbortError";
    setBeacon("error", isTimeout ? "Timeout" : "Offline");
    pulseTicker([{ category: "red_flag", message: isTimeout ? "Analysis timed out. Try a shorter response." : "Backend unavailable." }]);
  }
}

function debouncedScan() {
  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => scan().catch(() => setBeacon("error", "Offline")), DEBOUNCE_MS);
}

// ── MutationObserver (primary) + polling (fallback) ──

const chatContainer = document.querySelector("main") || document.body;
const observer = new MutationObserver((mutations) => {
  // Only trigger if text content changed in the chat area
  for (const m of mutations) {
    if (m.type === "childList" || m.type === "characterData") {
      debouncedScan();
      return;
    }
  }
});

observer.observe(chatContainer, {
  childList: true,
  subtree: true,
  characterData: true,
});

// Fallback polling in case MutationObserver misses something
setInterval(() => {
  scan().catch(() => setBeacon("error", "Offline"));
}, POLL_FALLBACK_MS);
