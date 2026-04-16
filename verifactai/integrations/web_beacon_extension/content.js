/**
 * VeriFACT AI — Browser Extension v3
 *
 * Parallel per-message analysis with full-conversation monitoring.
 * Glassmorphism squircle dashboard. Draggable beacon.
 *
 * Architecture:
 *   - MutationObserver detects new/changed messages
 *   - Up to 3 messages analyzed in parallel (concurrency pool)
 *   - Client-side pre-filter skips non-factual text before sending
 *   - Results cached per DOM node, only re-analyzed on content change
 *   - Beacon shows live progress (2/5, 3/5, ...)
 *   - Dashboard aggregates all results across full conversation
 */

const PLATFORM = location.hostname.includes("claude")
  ? "claude"
  : location.hostname.includes("gemini")
    ? "gemini"
    : "chatgpt";

const DEBOUNCE_MS = 1500;
const POLL_FALLBACK_MS = 8000;
const MAX_TEXT_PER_MSG = 2000;
const MAX_CONCURRENT = 3;

let panelVisible = false;
let panelEl = null;
let debounceTimer = null;
let scanRunning = false;

// Per-message result cache
const cache = new Map(); // node → { result, textHash }
let allResults = [];

// ── Utility ──────────────────────────────────────────

function textHash(s) { let h = 0; for (let i = 0; i < s.length; i++) h = ((h << 5) - h + s.charCodeAt(i)) | 0; return h; }

// Client-side pre-filter: strip non-factual lines before sending
function extractFactualText(text) {
  return text.split(/\n+/).filter((line) => {
    const l = line.trim().toLowerCase();
    if (l.length < 20) return false;
    if (l.endsWith("?")) return false;
    if (/^(here |if you|let me|are you|do you|would you|feel free|note:|tip:|in summary|to summarize|i hope|i can|sure|of course|great question|absolutely)/i.test(l)) return false;
    if (/^[\d]+\.\s*$/.test(l.trim())) return false; // bare numbered list
    return true;
  }).join("\n");
}

// ── Beacon ───────────────────────────────────────────

const LOGO_URL = chrome.runtime.getURL("logo.png");

const beacon = document.createElement("button");
beacon.className = "verifact-beacon idle";
beacon.title = "VeriFACT AI — Drag to move, click for dashboard";
beacon.innerHTML = `<img src="${LOGO_URL}" class="verifact-beacon-logo" alt="VF" /><span class="verifact-beacon-status">Ready</span>`;
document.body.appendChild(beacon);

function setBeacon(state, statusText) {
  beacon.className = "verifact-beacon " + state;
  beacon.querySelector(".verifact-beacon-status").textContent = statusText;
}

function computeOverallState() {
  if (!allResults.length) return { state: "idle", text: "Ready" };
  const tc = allResults.reduce((s, r) => s + (r.total_claims || 0), 0);
  if (tc === 0) return { state: "safe", text: "Safe" };
  const con = allResults.reduce((s, r) => s + (r.contradicted || 0), 0);
  const sup = allResults.reduce((s, r) => s + (r.supported || 0), 0);
  if (con === 0) return { state: "safe", text: "Safe" };
  const pct = sup / Math.max(tc, 1) * 100;
  if (pct >= 90) return { state: "safe", text: "Safe" };
  if (pct >= 70) return { state: "minor", text: "Minor" };
  if (pct >= 50) return { state: "moderate", text: "Check" };
  return { state: "hallucinated", text: `${con} False` };
}

// ── Draggable beacon ─────────────────────────────────

let isDragging = false, dragSX = 0, dragSY = 0, beaconSX = 0, beaconSY = 0, hasMoved = false;

beacon.addEventListener("mousedown", (e) => {
  isDragging = true; hasMoved = false;
  dragSX = e.clientX; dragSY = e.clientY;
  const r = beacon.getBoundingClientRect();
  beaconSX = r.left; beaconSY = r.top;
  beacon.style.transition = "none";
  e.preventDefault();
});
document.addEventListener("mousemove", (e) => {
  if (!isDragging) return;
  const dx = e.clientX - dragSX, dy = e.clientY - dragSY;
  if (Math.abs(dx) > 4 || Math.abs(dy) > 4) hasMoved = true;
  if (!hasMoved) return;
  beacon.style.left = Math.max(0, Math.min(innerWidth - 60, beaconSX + dx)) + "px";
  beacon.style.top = Math.max(0, Math.min(innerHeight - 60, beaconSY + dy)) + "px";
  beacon.style.right = "auto"; beacon.style.bottom = "auto";
});
document.addEventListener("mouseup", () => { if (isDragging) { isDragging = false; beacon.style.transition = ""; } });

beacon.addEventListener("click", (e) => {
  if (hasMoved) return;
  panelVisible = !panelVisible;
  if (panelVisible) renderDashboard();
  else if (panelEl) { panelEl.classList.remove("show"); setTimeout(() => { if (panelEl && !panelVisible) { panelEl.remove(); panelEl = null; } }, 300); }
});

// ── DOM scanning ─────────────────────────────────────

function getAssistantNodes() {
  let nodes;
  if (PLATFORM === "chatgpt") nodes = document.querySelectorAll("[data-message-author-role='assistant']");
  else if (PLATFORM === "gemini") nodes = document.querySelectorAll("message-content .markdown, model-response .markdown, model-response");
  else nodes = document.querySelectorAll("div[data-is-streaming], div.prose, div.font-claude-message");
  return Array.from(nodes).filter((n) => (n.innerText?.trim().length || 0) > 30);
}

// ── Backend call ─────────────────────────────────────

async function callAPI(text) {
  const store = await chrome.storage.local.get(["verifactApiUrl", "verifactEnabled", "verifactApiToken"]);
  if (store.verifactEnabled === false) return null;

  const url = store.verifactApiUrl || "https://adiashish-verifact-ai.hf.space/analyze/fast";
  const token = store.verifactApiToken || "";
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 90000);

  try {
    const res = await fetch(url, {
      method: "POST", signal: ctrl.signal,
      headers: { "Content-Type": "application/json", ...(token ? { "X-VeriFact-Token": token } : {}) },
      body: JSON.stringify({ text: text.slice(0, MAX_TEXT_PER_MSG), source: PLATFORM, mode: "fast", top_claims: 8 }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  } finally { clearTimeout(timer); }
}

// ── Inline highlighting ──────────────────────────────

function highlightNode(node, flags) {
  if (!node || !flags || node.dataset.vfDone === "1") return;
  const bad = flags.filter((f) => /(contrad|unver)/i.test(f.verdict || ""));
  if (!bad.length) { node.dataset.vfDone = "1"; return; }

  const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
  const tns = []; let c; while ((c = walker.nextNode())) if (c.nodeValue?.trim()) tns.push(c);

  for (const f of bad) {
    const claim = (f.claim || "").trim();
    if (claim.length < 12) continue;
    const probe = claim.slice(0, Math.min(50, claim.length)).toLowerCase();
    for (const tn of tns) {
      const idx = tn.nodeValue.toLowerCase().indexOf(probe);
      if (idx === -1) continue;
      const span = document.createElement("span");
      span.className = /(contrad)/i.test(f.verdict) ? "verifact-hl-red" : "verifact-hl-yellow";
      span.title = `${f.verdict} (${Math.round((f.confidence || 0) * 100)}%)`;
      const full = tn.nodeValue, frag = document.createDocumentFragment();
      if (idx > 0) frag.appendChild(document.createTextNode(full.slice(0, idx)));
      span.textContent = full.slice(idx, idx + probe.length);
      frag.appendChild(span);
      if (idx + probe.length < full.length) frag.appendChild(document.createTextNode(full.slice(idx + probe.length)));
      tn.parentNode.replaceChild(frag, tn);
      break;
    }
  }
  node.dataset.vfDone = "1";
}

// ── Parallel scan engine ─────────────────────────────

async function analyzeNode(node) {
  const raw = node.innerText?.trim() || "";
  const h = textHash(raw);
  const cached = cache.get(node);
  if (cached && cached.textHash === h) return cached.result;

  // Pre-filter non-factual text client-side
  const filtered = extractFactualText(raw);
  if (filtered.length < 30) {
    const empty = { total_claims: 0, supported: 0, contradicted: 0, unverifiable: 0, flags: [], factuality_score: 100 };
    cache.set(node, { result: empty, textHash: h });
    return empty;
  }

  try {
    const result = await callAPI(filtered);
    if (!result) return null;
    cache.set(node, { result, textHash: h });

    // Reset highlights for re-analysis
    node.dataset.vfDone = "";
    highlightNode(node, result.flags || []);
    return result;
  } catch (err) {
    if (err.name === "AbortError") {
      const timeout = { total_claims: 0, supported: 0, contradicted: 0, unverifiable: 0, flags: [], _timeout: true };
      cache.set(node, { result: timeout, textHash: h });
      return timeout;
    }
    return null;
  }
}

async function runPool(nodes) {
  // Process nodes in batches of MAX_CONCURRENT
  const results = new Array(nodes.length);
  let nextIdx = 0;

  async function worker() {
    while (nextIdx < nodes.length) {
      const i = nextIdx++;
      results[i] = await analyzeNode(nodes[i]);
    }
  }

  const workers = Array.from({ length: Math.min(MAX_CONCURRENT, nodes.length) }, () => worker());
  await Promise.all(workers);
  return results;
}

async function scan() {
  if (scanRunning) return;
  scanRunning = true;

  try {
    const nodes = getAssistantNodes();
    if (!nodes.length) { scanRunning = false; return; }

    // Find nodes that need analysis (new or changed content)
    const needsWork = nodes.filter((n) => {
      const raw = n.innerText?.trim() || "";
      const h = textHash(raw);
      const cached = cache.get(n);
      return !cached || cached.textHash !== h;
    });

    if (!needsWork.length) {
      // Rebuild results from cache
      allResults = nodes.map((n) => cache.get(n)?.result).filter(Boolean);
      const s = computeOverallState();
      setBeacon(s.state, s.text);
      scanRunning = false;
      return;
    }

    // Show progress
    const total = needsWork.length;
    let done = 0;
    setBeacon("analyzing", `0/${total}`);

    // Override analyzeNode to update progress
    const origAnalyze = analyzeNode;
    const wrappedNodes = needsWork.map((node) => {
      return (async () => {
        const result = await origAnalyze(node);
        done++;
        setBeacon("analyzing", `${done}/${total}`);
        // Live-update dashboard if open
        allResults = nodes.map((n) => cache.get(n)?.result).filter(Boolean);
        if (panelVisible) renderDashboard();
        return result;
      })();
    });

    // Run with concurrency pool
    const pool = [];
    for (const p of wrappedNodes) {
      pool.push(p);
      if (pool.length >= MAX_CONCURRENT) {
        await Promise.race(pool);
        // Remove resolved promises
        for (let i = pool.length - 1; i >= 0; i--) {
          const status = await Promise.race([pool[i].then(() => "done"), Promise.resolve("pending")]);
          if (status === "done") pool.splice(i, 1);
        }
      }
    }
    await Promise.all(pool);

    // Final state
    allResults = nodes.map((n) => cache.get(n)?.result).filter(Boolean);
    const s = computeOverallState();
    setBeacon(s.state, s.text);
    if (panelVisible) renderDashboard();

  } catch (err) {
    setBeacon("error", "Error");
  } finally {
    scanRunning = false;
  }
}

// ── Dashboard ────────────────────────────────────────

function verdictIcon(v) {
  const l = (v || "").toLowerCase();
  if (l.includes("contrad")) return '<span class="vf-icon vf-icon-red">!</span>';
  if (l.includes("support")) return '<span class="vf-icon vf-icon-green">&#10003;</span>';
  if (l.includes("unver")) return '<span class="vf-icon vf-icon-yellow">?</span>';
  return '<span class="vf-icon vf-icon-gray">-</span>';
}

let dashDragging = false, dashDragSX = 0, dashDragSY = 0, dashStartX = 0, dashStartY = 0, dashPositioned = false;

function renderDashboard() {
  if (!panelEl) {
    panelEl = document.createElement("div");
    panelEl.className = "vf-dashboard";
    document.body.appendChild(panelEl);
    dashPositioned = false;

    panelEl.addEventListener("click", (e) => {
      if (e.target.classList.contains("vf-close")) {
        panelVisible = false;
        panelEl.classList.remove("show");
        setTimeout(() => { if (panelEl && !panelVisible) { panelEl.remove(); panelEl = null; dashPositioned = false; } }, 300);
      }
    });

    // Make dashboard draggable via header area
    panelEl.addEventListener("mousedown", (e) => {
      const header = panelEl.querySelector(".vf-dash-header");
      if (!header || !header.contains(e.target)) return;
      if (e.target.tagName === "A") return; // don't block logo click
      dashDragging = true;
      dashDragSX = e.clientX; dashDragSY = e.clientY;
      const r = panelEl.getBoundingClientRect();
      dashStartX = r.left; dashStartY = r.top;
      panelEl.style.transition = "none";
      e.preventDefault();
    });
    document.addEventListener("mousemove", (e) => {
      if (!dashDragging) return;
      panelEl.style.left = Math.max(0, dashStartX + e.clientX - dashDragSX) + "px";
      panelEl.style.top = Math.max(0, dashStartY + e.clientY - dashDragSY) + "px";
      panelEl.style.right = "auto"; panelEl.style.bottom = "auto";
    });
    document.addEventListener("mouseup", () => { if (dashDragging) { dashDragging = false; panelEl.style.transition = ""; } });
  }

  // Position near beacon only on first open
  if (!dashPositioned) {
    const br = beacon.getBoundingClientRect();
    const dw = 400, maxH = innerHeight * 0.8;
    let left = br.left - dw - 14;
    if (left < 8) left = br.right + 14;
    if (left + dw > innerWidth - 8) left = innerWidth - dw - 8;
    let top = br.top - maxH / 2 + 28;
    top = Math.max(8, Math.min(top, innerHeight - maxH - 8));
    panelEl.style.left = left + "px";
    panelEl.style.top = top + "px";
    panelEl.style.right = "auto";
    panelEl.style.bottom = "auto";
    dashPositioned = true;
  }

  const tc = allResults.reduce((s, r) => s + (r.total_claims || 0), 0);
  const sup = allResults.reduce((s, r) => s + (r.supported || 0), 0);
  const con = allResults.reduce((s, r) => s + (r.contradicted || 0), 0);
  const unv = allResults.reduce((s, r) => s + (r.unverifiable || 0), 0);
  const score = tc > 0 ? Math.round(sup / tc * 100) : 100;
  const msgs = allResults.length;
  const allFlags = allResults.flatMap((r) => r.flags || []);

  // Only show non-supported flags in dashboard
  const issues = allFlags.filter((f) => !/(support)/i.test(f.verdict || ""));

  const cardsHtml = issues.length ? issues.map((f) => {
    const conf = Math.round((f.confidence || 0) * 100);
    const ev = (f.evidence || "No evidence available").replace(/</g, "&lt;").substring(0, 250);
    const src = f.source || "knowledge base";
    const cls = /(contrad)/i.test(f.verdict) ? "vf-card-red" : "vf-card-yellow";
    return `<div class="vf-card ${cls}">
      <div class="vf-card-header">${verdictIcon(f.verdict)}<span class="vf-card-verdict">${f.verdict}</span><span class="vf-card-conf">${conf}%</span></div>
      <div class="vf-card-claim">${(f.claim || "").replace(/</g, "&lt;")}</div>
      <div class="vf-card-evidence"><span class="vf-card-ev-label">Evidence:</span> ${ev}</div>
      <div class="vf-card-source">Source: ${src.replace(/</g, "&lt;")}</div>
    </div>`;
  }).join("") : '<div class="vf-empty">No issues detected. All claims verified.</div>';

  panelEl.innerHTML = `<div class="vf-dash-inner">
    <button class="vf-close" title="Close">&times;</button>
    <div class="vf-dash-header">
      <a href="https://web-five-mocha-51.vercel.app" target="_blank" rel="noreferrer" class="vf-logo" title="Open VeriFACT Dashboard"><img src="${LOGO_URL}" class="vf-logo-img" alt="VF" /></a>
      <div class="vf-title">VeriFACT AI</div>
      <div class="vf-subtitle">${PLATFORM.charAt(0).toUpperCase() + PLATFORM.slice(1)} &middot; ${msgs} messages scanned</div>
    </div>
    <div class="vf-kpi-row">
      <div class="vf-kpi"><div class="vf-kpi-val ${score >= 80 ? 'vf-green' : score >= 50 ? 'vf-yellow' : 'vf-red'}">${score}</div><div class="vf-kpi-lbl">Score</div></div>
      <div class="vf-kpi"><div class="vf-kpi-val">${tc}</div><div class="vf-kpi-lbl">Claims</div></div>
      <div class="vf-kpi"><div class="vf-kpi-val vf-green">${sup}</div><div class="vf-kpi-lbl">Verified</div></div>
      <div class="vf-kpi"><div class="vf-kpi-val vf-red">${con}</div><div class="vf-kpi-lbl">False</div></div>
      <div class="vf-kpi"><div class="vf-kpi-val vf-yellow">${unv}</div><div class="vf-kpi-lbl">Unclear</div></div>
    </div>
    <div class="vf-divider"></div>
    <div class="vf-section-title">${issues.length ? issues.length + ' Issue' + (issues.length > 1 ? 's' : '') + ' Found' : 'Conversation Clean'}</div>
    <div class="vf-cards">${cardsHtml}</div>
    <div class="vf-footer">DeBERTa-v3 NLI &middot; Wikipedia + BM25 &middot; Rule Engine</div>
  </div>`;

  requestAnimationFrame(() => panelEl.classList.add("show"));
}

// ── Prompt Optimizer (real-time suggestions) ─────────

let promptTip = null;
let promptTimer = null;
let lastPromptText = "";

function getInputElement() {
  if (PLATFORM === "chatgpt") return document.querySelector("#prompt-textarea, textarea[data-id]");
  if (PLATFORM === "gemini") return document.querySelector("rich-textarea .ql-editor, textarea, div[contenteditable='true']");
  return document.querySelector("div[contenteditable='true'], textarea");
}

async function optimizePrompt(text) {
  const store = await chrome.storage.local.get(["verifactApiUrl"]);
  const base = (store.verifactApiUrl || "https://adiashish-verifact-ai.hf.space").replace("/analyze", "");
  try {
    const res = await fetch(base + "/optimize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: text }),
    });
    if (!res.ok) return null;
    return res.json();
  } catch { return null; }
}

let lastSuggestedPrompt = "";

function showPromptTip(data) {
  if (!data || !data.improvements || !data.improvements.length) { hidePromptTip(); return; }
  if (!promptTip) {
    promptTip = document.createElement("div");
    promptTip.className = "vf-prompt-tip";
    document.body.appendChild(promptTip);
  }

  lastSuggestedPrompt = data.suggested || "";

  const inputEl = getInputElement();
  if (inputEl) {
    const r = inputEl.getBoundingClientRect();
    promptTip.style.left = r.left + "px";
    promptTip.style.bottom = (innerHeight - r.top + 8) + "px";
    promptTip.style.width = Math.min(r.width, 480) + "px";
  }

  const improvHtml = data.improvements.slice(0, 3).map((i) => `<div class="vf-pt-item">${i.replace(/</g, "&lt;")}</div>`).join("");
  promptTip.innerHTML = `
    <div class="vf-pt-drag-handle">
      <span class="vf-pt-logo">VF</span>
      <span class="vf-pt-title">Prompt Suggestion</span>
      <span class="vf-pt-score">${data.score || 0}/100</span>
      <button class="vf-pt-close" title="Close">&times;</button>
    </div>
    ${improvHtml}
    ${lastSuggestedPrompt ? `<div class="vf-pt-suggested">${lastSuggestedPrompt.replace(/</g, "&lt;").substring(0, 300)}</div>` : ""}
    <div class="vf-pt-buttons">
      <button class="vf-pt-accept" data-action="accept">Accept Suggested</button>
      <button class="vf-pt-dismiss" data-action="dismiss">Use Original</button>
    </div>
  `;

  // Close X button
  promptTip.querySelector(".vf-pt-close")?.addEventListener("click", (e) => {
    e.stopPropagation();
    hidePromptTip();
  });

  // Make prompt tip draggable via header
  const dragHandle = promptTip.querySelector(".vf-pt-drag-handle");
  let ptDragging = false, ptDragSX = 0, ptDragSY = 0, ptStartX = 0, ptStartY = 0;
  dragHandle.addEventListener("mousedown", (e) => {
    if (e.target.classList.contains("vf-pt-close")) return;
    ptDragging = true;
    ptDragSX = e.clientX; ptDragSY = e.clientY;
    const r = promptTip.getBoundingClientRect();
    ptStartX = r.left; ptStartY = r.top;
    promptTip.style.transition = "none";
    e.preventDefault();
  });
  document.addEventListener("mousemove", (e) => {
    if (!ptDragging) return;
    promptTip.style.left = Math.max(0, ptStartX + e.clientX - ptDragSX) + "px";
    promptTip.style.top = Math.max(0, ptStartY + e.clientY - ptDragSY) + "px";
    promptTip.style.bottom = "auto";
    promptTip.style.right = "auto";
  });
  document.addEventListener("mouseup", () => { if (ptDragging) { ptDragging = false; promptTip.style.transition = ""; } });

  // Button handlers
  promptTip.querySelector(".vf-pt-accept")?.addEventListener("click", () => {
    const el = getInputElement();
    if (el && lastSuggestedPrompt) {
      if (el.innerText !== undefined && el.contentEditable === "true") {
        el.innerText = lastSuggestedPrompt;
      } else {
        el.value = lastSuggestedPrompt;
      }
      el.dispatchEvent(new Event("input", { bubbles: true }));
    }
    hidePromptTip();
  });

  promptTip.querySelector(".vf-pt-dismiss")?.addEventListener("click", () => {
    hidePromptTip();
  });

  promptTip.classList.add("show");
}

function hidePromptTip() {
  if (promptTip) promptTip.classList.remove("show");
}

function watchPromptInput() {
  const inputEl = getInputElement();
  if (!inputEl) return;

  const handler = () => {
    const text = (inputEl.innerText || inputEl.value || "").trim();
    if (text === lastPromptText || text.length < 15) { hidePromptTip(); return; }
    lastPromptText = text;

    if (promptTimer) clearTimeout(promptTimer);
    promptTimer = setTimeout(async () => {
      const data = await optimizePrompt(text);
      if (data) showPromptTip(data);
    }, 3000); // 3s debounce after user stops typing
  };

  inputEl.addEventListener("input", handler);
  inputEl.addEventListener("keyup", handler);
  // Re-watch when DOM changes (new chat, page navigation)
  new MutationObserver(() => {
    const newInput = getInputElement();
    if (newInput && newInput !== inputEl) {
      newInput.addEventListener("input", handler);
      newInput.addEventListener("keyup", handler);
    }
  }).observe(document.body, { childList: true, subtree: true });
}

// Start prompt watching after page loads
setTimeout(watchPromptInput, 2000);

// ── Observer + polling ───────────────────────────────

function debouncedScan() {
  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => scan().catch(() => setBeacon("error", "Offline")), DEBOUNCE_MS);
}

const chatRoot = document.querySelector("main") || document.body;
new MutationObserver(() => debouncedScan()).observe(chatRoot, { childList: true, subtree: true, characterData: true });
setInterval(() => scan().catch(() => setBeacon("error", "Offline")), POLL_FALLBACK_MS);
