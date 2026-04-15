/**
 * VeriFACT AI — Browser Extension Content Script v2
 *
 * Full-conversation scanning with per-message analysis.
 * Glassmorphism squircle dashboard with claim-level detail.
 *
 * Color spec:
 *   Blue   = safe (90-100% correctness)
 *   Purple = minor hallucinations (70-89%)
 *   Orange = moderate / needs fact check (50-69%)
 *   Red    = hallucinations detected (<50%)
 */

const PLATFORM = location.hostname.includes("claude")
  ? "claude"
  : location.hostname.includes("gemini")
    ? "gemini"
    : "chatgpt";

const DEBOUNCE_MS = 2500;
const POLL_FALLBACK_MS = 6000;
const MAX_TEXT_PER_MSG = 2000;

let panelVisible = false;
let panelEl = null;
let debounceTimer = null;
let isAnalyzing = false;

// Store results per message node for full-conversation tracking
const analyzedMessages = new Map(); // node → result
let latestResults = []; // ordered list of all results

// ── Beacon creation ──────────────────────────────────

const beacon = document.createElement("button");
beacon.className = "verifact-beacon idle";
beacon.title = "VeriFACT AI — Drag to move, click to open dashboard";
beacon.innerHTML = '<span class="verifact-beacon-label">VF</span><span class="verifact-beacon-status">Ready</span>';
document.body.appendChild(beacon);

// ── Draggable beacon ────────────────────────────────
let isDragging = false;
let dragStartX = 0, dragStartY = 0;
let beaconStartX = 0, beaconStartY = 0;
let hasMoved = false;

beacon.addEventListener("mousedown", (e) => {
  isDragging = true;
  hasMoved = false;
  dragStartX = e.clientX;
  dragStartY = e.clientY;
  const rect = beacon.getBoundingClientRect();
  beaconStartX = rect.left;
  beaconStartY = rect.top;
  beacon.style.transition = "none";
  e.preventDefault();
});

document.addEventListener("mousemove", (e) => {
  if (!isDragging) return;
  const dx = e.clientX - dragStartX;
  const dy = e.clientY - dragStartY;
  if (Math.abs(dx) > 4 || Math.abs(dy) > 4) hasMoved = true;
  if (!hasMoved) return;

  const newX = Math.max(0, Math.min(window.innerWidth - 60, beaconStartX + dx));
  const newY = Math.max(0, Math.min(window.innerHeight - 60, beaconStartY + dy));
  beacon.style.left = newX + "px";
  beacon.style.top = newY + "px";
  beacon.style.right = "auto";
  beacon.style.bottom = "auto";
});

document.addEventListener("mouseup", () => {
  if (!isDragging) return;
  isDragging = false;
  beacon.style.transition = "";
});

beacon.addEventListener("click", (e) => {
  if (hasMoved) { e.preventDefault(); return; } // was a drag, not a click
  panelVisible = !panelVisible;
  if (panelVisible) {
    renderDashboard();
  } else if (panelEl) {
    panelEl.classList.remove("show");
    setTimeout(() => { if (panelEl && !panelVisible) { panelEl.remove(); panelEl = null; } }, 300);
  }
});

// ── Beacon state machine ─────────────────────────────

function setBeacon(state, statusText) {
  beacon.className = "verifact-beacon " + state;
  const statusEl = beacon.querySelector(".verifact-beacon-status");
  if (statusEl && statusText) statusEl.textContent = statusText;
}

function overallBeaconState() {
  if (!latestResults.length) return { state: "idle", text: "Ready" };

  const totalClaims = latestResults.reduce((s, r) => s + (r.total_claims || 0), 0);
  if (totalClaims === 0) return { state: "safe", text: "Safe" };

  const contradicted = latestResults.reduce((s, r) => s + (r.contradicted || 0), 0);
  const supported = latestResults.reduce((s, r) => s + (r.supported || 0), 0);

  if (contradicted === 0) return { state: "safe", text: "Safe" };

  const ratio = supported / Math.max(totalClaims, 1) * 100;
  if (ratio >= 90) return { state: "safe", text: "Safe" };
  if (ratio >= 70) return { state: "minor", text: "Minor" };
  if (ratio >= 50) return { state: "moderate", text: "Check" };
  return { state: "hallucinated", text: "Alert" };
}

// ── DOM scanning ─────────────────────────────────────

function getAssistantMessageNodes() {
  let nodes = [];
  if (PLATFORM === "chatgpt") {
    nodes = Array.from(document.querySelectorAll("[data-message-author-role='assistant']"));
  } else if (PLATFORM === "gemini") {
    nodes = Array.from(document.querySelectorAll("message-content .markdown, model-response .markdown, model-response"));
  } else {
    nodes = Array.from(document.querySelectorAll("div[data-is-streaming], div.prose, div.font-claude-message"));
  }
  return nodes.filter((n) => n.innerText?.trim().length > 30);
}

// ── Backend call ─────────────────────────────────────

async function callAnalyzer(text) {
  const store = await chrome.storage.local.get(["verifactApiUrl", "verifactEnabled", "verifactApiToken"]);
  if (store.verifactEnabled === false) return null;

  const apiUrl = store.verifactApiUrl || "https://adiashish-verifact-ai.hf.space/analyze";
  const apiToken = store.verifactApiToken || "";

  // Truncate very long messages
  const truncated = text.length > MAX_TEXT_PER_MSG ? text.slice(0, MAX_TEXT_PER_MSG) : text;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 90000);

  try {
    const response = await fetch(apiUrl, {
      method: "POST",
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...(apiToken ? { "X-VeriFact-Token": apiToken } : {}),
      },
      body: JSON.stringify({
        text: truncated,
        source: PLATFORM,
        mode: "fast",
        include_claims: true,
        top_claims: 8,
      }),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  } finally {
    clearTimeout(timeout);
  }
}

// ── Inline highlighting ──────────────────────────────

function highlightInNode(node, flags) {
  if (!node || !flags || node.dataset.verifactDone === "1") return;
  const flagged = flags.filter((f) => {
    const v = (f.verdict || "").toLowerCase();
    return v.includes("contrad") || v.includes("unver");
  });
  if (!flagged.length) { node.dataset.verifactDone = "1"; return; }

  const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
  const textNodes = [];
  let cur;
  while ((cur = walker.nextNode())) {
    if (cur.nodeValue?.trim()) textNodes.push(cur);
  }

  for (const f of flagged) {
    const claim = (f.claim || "").trim();
    if (claim.length < 12) continue;
    const probe = claim.slice(0, Math.min(50, claim.length)).toLowerCase();
    for (const tn of textNodes) {
      const idx = tn.nodeValue.toLowerCase().indexOf(probe);
      if (idx === -1) continue;
      const span = document.createElement("span");
      const isContradicted = (f.verdict || "").toLowerCase().includes("contrad");
      span.className = isContradicted ? "verifact-hl-red" : "verifact-hl-yellow";
      span.dataset.verifactClaim = claim;
      span.dataset.verifactVerdict = f.verdict;
      span.dataset.verifactConf = Math.round((f.confidence || 0) * 100);
      span.dataset.verifactEvidence = (f.evidence || "").substring(0, 200);
      span.dataset.verifactSource = f.source || "";
      span.title = `${f.verdict} (${span.dataset.verifactConf}%) — Click VF for details`;

      const full = tn.nodeValue;
      const frag = document.createDocumentFragment();
      if (idx > 0) frag.appendChild(document.createTextNode(full.slice(0, idx)));
      span.textContent = full.slice(idx, idx + probe.length);
      frag.appendChild(span);
      if (idx + probe.length < full.length) frag.appendChild(document.createTextNode(full.slice(idx + probe.length)));
      tn.parentNode.replaceChild(frag, tn);
      break;
    }
  }
  node.dataset.verifactDone = "1";
}

// ── Dashboard rendering (glassmorphism squircle) ─────

function verdictIcon(v) {
  const l = (v || "").toLowerCase();
  if (l.includes("contrad")) return '<span class="vf-icon vf-icon-red">!</span>';
  if (l.includes("support")) return '<span class="vf-icon vf-icon-green">&#10003;</span>';
  if (l.includes("unver")) return '<span class="vf-icon vf-icon-yellow">?</span>';
  return '<span class="vf-icon vf-icon-gray">-</span>';
}

function renderDashboard() {
  if (!panelEl) {
    panelEl = document.createElement("div");
    panelEl.className = "vf-dashboard";
    document.body.appendChild(panelEl);
    panelEl.addEventListener("click", (e) => {
      if (e.target.classList.contains("vf-close")) {
        panelVisible = false;
        panelEl.classList.remove("show");
        setTimeout(() => { if (panelEl && !panelVisible) { panelEl.remove(); panelEl = null; } }, 300);
      }
    });
  }

  // Position dashboard near beacon
  const br = beacon.getBoundingClientRect();
  const dw = 380, maxH = window.innerHeight * 0.75;
  let left = br.left - dw - 12;
  if (left < 8) left = br.right + 12;
  if (left + dw > window.innerWidth - 8) left = window.innerWidth - dw - 8;
  let top = br.top - maxH / 2 + 28;
  if (top < 8) top = 8;
  if (top + maxH > window.innerHeight - 8) top = window.innerHeight - maxH - 8;
  panelEl.style.left = left + "px";
  panelEl.style.top = top + "px";
  panelEl.style.right = "auto";
  panelEl.style.bottom = "auto";

  const totalClaims = latestResults.reduce((s, r) => s + (r.total_claims || 0), 0);
  const supported = latestResults.reduce((s, r) => s + (r.supported || 0), 0);
  const contradicted = latestResults.reduce((s, r) => s + (r.contradicted || 0), 0);
  const unverifiable = latestResults.reduce((s, r) => s + (r.unverifiable || 0), 0);
  const score = totalClaims > 0 ? Math.round(supported / totalClaims * 100) : 100;
  const allFlags = latestResults.flatMap((r) => r.flags || []);

  const flaggedHtml = allFlags.length ? allFlags.map((f) => {
    const conf = Math.round((f.confidence || 0) * 100);
    const ev = (f.evidence || "No evidence available").replace(/</g, "&lt;").substring(0, 250);
    const src = f.source || "knowledge base";
    const vClass = (f.verdict || "").toLowerCase().includes("contrad") ? "vf-card-red"
      : (f.verdict || "").toLowerCase().includes("unver") ? "vf-card-yellow" : "vf-card-green";
    return `
      <div class="vf-card ${vClass}">
        <div class="vf-card-header">
          ${verdictIcon(f.verdict)}
          <span class="vf-card-verdict">${f.verdict}</span>
          <span class="vf-card-conf">${conf}%</span>
        </div>
        <div class="vf-card-claim">${(f.claim || "").replace(/</g, "&lt;")}</div>
        <div class="vf-card-evidence">
          <span class="vf-card-ev-label">Evidence:</span> ${ev}
        </div>
        <div class="vf-card-source">Source: ${src.replace(/</g, "&lt;")}</div>
      </div>`;
  }).join("") : '<div class="vf-empty">No issues detected. All claims verified.</div>';

  panelEl.innerHTML = `
    <div class="vf-dash-inner">
      <button class="vf-close" title="Close">&times;</button>
      <div class="vf-dash-header">
        <div class="vf-logo">VF</div>
        <div class="vf-title">VeriFACT AI</div>
        <div class="vf-subtitle">${PLATFORM.charAt(0).toUpperCase() + PLATFORM.slice(1)} Analysis</div>
      </div>
      <div class="vf-kpi-row">
        <div class="vf-kpi">
          <div class="vf-kpi-val ${score >= 80 ? 'vf-green' : score >= 50 ? 'vf-yellow' : 'vf-red'}">${score}</div>
          <div class="vf-kpi-lbl">Score</div>
        </div>
        <div class="vf-kpi">
          <div class="vf-kpi-val">${totalClaims}</div>
          <div class="vf-kpi-lbl">Claims</div>
        </div>
        <div class="vf-kpi">
          <div class="vf-kpi-val vf-green">${supported}</div>
          <div class="vf-kpi-lbl">Verified</div>
        </div>
        <div class="vf-kpi">
          <div class="vf-kpi-val vf-red">${contradicted}</div>
          <div class="vf-kpi-lbl">False</div>
        </div>
        <div class="vf-kpi">
          <div class="vf-kpi-val vf-yellow">${unverifiable}</div>
          <div class="vf-kpi-lbl">Unclear</div>
        </div>
      </div>
      <div class="vf-divider"></div>
      <div class="vf-section-title">${allFlags.length ? 'Flagged Claims' : 'Conversation Clean'}</div>
      <div class="vf-cards">${flaggedHtml}</div>
      <div class="vf-footer">Powered by DeBERTa-v3 NLI + Wikipedia + BM25</div>
    </div>
  `;

  requestAnimationFrame(() => panelEl.classList.add("show"));
}

// ── Main scan loop (per-message) ─────────────────────

async function scan() {
  if (isAnalyzing) return;
  const msgNodes = getAssistantMessageNodes();
  if (!msgNodes.length) return;

  // Find unanalyzed messages
  const newNodes = msgNodes.filter((n) => !analyzedMessages.has(n));
  // Also re-check last message if its content changed
  const lastNode = msgNodes[msgNodes.length - 1];
  const lastText = lastNode?.innerText?.trim() || "";
  const lastCached = analyzedMessages.get(lastNode);
  const lastChanged = lastCached && lastCached._textLen !== lastText.length;

  if (!newNodes.length && !lastChanged) return;

  isAnalyzing = true;
  setBeacon("analyzing", "Checking...");

  try {
    // Analyze each new message independently
    const toAnalyze = lastChanged ? [...newNodes.filter((n) => n !== lastNode), lastNode] : newNodes;

    for (const node of toAnalyze) {
      const text = node.innerText?.trim();
      if (!text || text.length < 30) continue;

      try {
        const result = await callAnalyzer(text);
        if (!result) continue;
        result._textLen = text.length;
        analyzedMessages.set(node, result);
        highlightInNode(node, result.flags || []);
      } catch (err) {
        if (err.name === "AbortError") {
          // On timeout, store partial result so we don't re-try immediately
          analyzedMessages.set(node, { total_claims: 0, flags: [], _textLen: text.length, _timeout: true });
        }
      }
    }

    // Rebuild results list in DOM order
    latestResults = msgNodes.map((n) => analyzedMessages.get(n)).filter(Boolean);

    // Update beacon state
    const { state, text } = overallBeaconState();
    setBeacon(state, text);

    // Update dashboard if open
    if (panelVisible) renderDashboard();

  } catch (err) {
    setBeacon("error", "Error");
  } finally {
    isAnalyzing = false;
  }
}

function debouncedScan() {
  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => scan().catch(() => setBeacon("error", "Offline")), DEBOUNCE_MS);
}

// ── MutationObserver + polling ───────────────────────

const chatContainer = document.querySelector("main") || document.body;
const observer = new MutationObserver(() => debouncedScan());
observer.observe(chatContainer, { childList: true, subtree: true, characterData: true });
setInterval(() => scan().catch(() => setBeacon("error", "Offline")), POLL_FALLBACK_MS);
