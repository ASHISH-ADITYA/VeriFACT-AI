const PLATFORM = location.hostname.includes("claude")
  ? "claude"
  : (location.hostname.includes("gemini") ? "gemini" : "chatgpt");
const POLL_MS = 3500;
const CONVERSATION_WINDOW = 6;

let lastFingerprint = "";
let panelVisible = false;
let panelEl = null;
let latestTargetNode = null;
let pulseTimer = null;

const beacon = document.createElement("button");
beacon.className = "verifact-beacon idle";
beacon.title = "VeriFact Beacon";
beacon.textContent = "VF";
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

function setBeacon(state) {
  beacon.className = "verifact-beacon " + state;
}

function alertClass(category) {
  const c = String(category || "").toLowerCase();
  if (c.includes("halluc")) return "hallucination";
  if (c.includes("bias")) return "bias";
  if (c.includes("red")) return "red-flag";
  return "neutral";
}

function pulseTicker(alerts = []) {
  if (pulseTimer) {
    clearTimeout(pulseTimer);
    pulseTimer = null;
  }

  if (!alerts.length) {
    ticker.innerHTML = '<div class="verifact-chip neutral">No immediate red flags in latest pass.</div>';
    ticker.classList.add("show");
    pulseTimer = setTimeout(() => ticker.classList.remove("show"), 3800);
    return;
  }

  const chips = alerts.slice(0, 2).map((a) => {
    const cls = alertClass(a.category);
    const msg = String(a.message || "Potential issue detected").replace(/</g, "&lt;");
    return `<div class="verifact-chip ${cls}">${msg}</div>`;
  }).join("");

  ticker.innerHTML = chips;
  ticker.classList.add("show");
  pulseTimer = setTimeout(() => ticker.classList.remove("show"), 5200);
}

function ensurePanel() {
  if (!panelVisible) {
    return null;
  }
  if (!panelEl) {
    panelEl = document.createElement("div");
    panelEl.className = "verifact-panel";
    panelEl.innerHTML = "<h3>VeriFact Live</h3><div>Waiting for assistant response...</div>";
    document.body.appendChild(panelEl);
  }
  return panelEl;
}

function getAssistantMessages() {
  if (PLATFORM === "chatgpt") {
    const nodes = Array.from(document.querySelectorAll("[data-message-author-role='assistant']"));
    const pairs = nodes
      .map((n) => ({ text: n.innerText?.trim(), node: n }))
      .filter((x) => x.text);
    latestTargetNode = pairs.length ? pairs[pairs.length - 1].node : null;
    return pairs.map((p) => p.text);
  }

  if (PLATFORM === "gemini") {
    const nodes = Array.from(
      document.querySelectorAll("message-content .markdown, model-response .markdown, model-response")
    );
    const pairs = nodes
      .map((n) => ({ text: n.innerText?.trim(), node: n }))
      .filter((x) => x.text && x.text.length > 40);
    latestTargetNode = pairs.length ? pairs[pairs.length - 1].node : null;
    return pairs.map((p) => p.text).slice(-8);
  }

  const claudeNodes = Array.from(
    document.querySelectorAll("div[data-is-streaming], div.prose, div.font-claude-message")
  );
  const pairs = claudeNodes
    .map((n) => ({ text: n.innerText?.trim(), node: n }))
    .filter((x) => x.text && x.text.length > 40)
    .slice(-8);
  latestTargetNode = pairs.length ? pairs[pairs.length - 1].node : null;
  return pairs.map((p) => p.text);
}

function fingerprint(text) {
  return `${text.length}:${text.slice(0, 80)}:${text.slice(-80)}`;
}

async function callAnalyzer(text) {
  const store = await chrome.storage.local.get(["verifactApiUrl", "verifactEnabled", "verifactApiToken"]);
  if (store.verifactEnabled === false) {
    return null;
  }
  const apiUrl = store.verifactApiUrl || "http://127.0.0.1:8765/analyze";
  const apiToken = store.verifactApiToken || "";

  const response = await fetch(apiUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-VeriFact-Token": apiToken
    },
    body: JSON.stringify({
      text,
      source: PLATFORM,
      mode: "fast",
      include_claims: true,
      top_claims: 6
    })
  });

  if (!response.ok) {
    throw new Error(`Analyzer failed: HTTP ${response.status}`);
  }
  return response.json();
}

function verdictClass(score) {
  if (score >= 80) return "good";
  if (score >= 55) return "warn";
  return "bad";
}

function claimClass(v) {
  const x = (v || "").toLowerCase();
  if (x.includes("contrad")) return "contradicted";
  if (x.includes("support")) return "supported";
  if (x.includes("unver")) return "unverifiable";
  return "";
}

function reasonForClaim(claim) {
  if (claim.reason) {
    return claim.reason;
  }
  const v = (claim.verdict || "").toLowerCase();
  if (v.includes("contrad")) {
    return "Evidence contradicts the claim details.";
  }
  if (v.includes("unver")) {
    return "Evidence is not specific enough to verify the claim.";
  }
  return "Claim is supported by retrieved evidence.";
}

function highlightClaimsInNode(node, flags) {
  if (!node || !flags || !flags.length) {
    return;
  }
  const flagged = flags.filter((f) => {
    const v = (f.verdict || "").toLowerCase();
    return v.includes("contrad") || v.includes("unver");
  });
  if (!flagged.length) {
    return;
  }

  // Skip if already highlighted this message block.
  if (node.dataset.verifactHighlighted === "1") {
    return;
  }

  const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
  const textNodes = [];
  let current;
  while ((current = walker.nextNode())) {
    if (current.nodeValue && current.nodeValue.trim()) {
      textNodes.push(current);
    }
  }

  for (const f of flagged) {
    const claim = (f.claim || "").trim();
    if (claim.length < 16) {
      continue;
    }
    const probe = claim.slice(0, Math.min(48, claim.length));
    for (const tn of textNodes) {
      const idx = tn.nodeValue.toLowerCase().indexOf(probe.toLowerCase());
      if (idx === -1) {
        continue;
      }
      const span = document.createElement("span");
      span.className = "verifact-inline-flag";
      span.title = `${f.verdict} (${Math.round((f.confidence || 0) * 100)}%)`;
      const full = tn.nodeValue;
      const before = full.slice(0, idx);
      const mid = full.slice(idx, idx + probe.length);
      const after = full.slice(idx + probe.length);
      const frag = document.createDocumentFragment();
      if (before) frag.appendChild(document.createTextNode(before));
      span.textContent = mid;
      frag.appendChild(span);
      if (after) frag.appendChild(document.createTextNode(after));
      tn.parentNode.replaceChild(frag, tn);
      node.dataset.verifactHighlighted = "1";
      break;
    }
  }

  // Fallback: if no claim substring matched, mark the full node as risky context.
  if (node.dataset.verifactHighlighted !== "1") {
    node.classList.add("verifact-inline-flag");
    node.dataset.verifactHighlighted = "1";
  }
}

function renderPanel(result) {
  const panel = ensurePanel();
  if (!panel) {
    return;
  }
  const alerts = (result.alerts || []).map((a) => {
    const cat = String(a.category || "red_flag").replace(/</g, "&lt;");
    const sev = String(a.severity || "medium").replace(/</g, "&lt;");
    const msg = String(a.message || "Potential issue detected").replace(/</g, "&lt;");
    return `<div class="verifact-claim ${alertClass(cat)}"><div><strong>${cat.toUpperCase()}</strong> (${sev})</div><div>${msg}</div></div>`;
  }).join("");

  const claims = (result.flags || []).map((c) => {
    const confPct = Math.round((c.confidence || 0) * 100);
    const uncPct = Math.round((c.uncertainty || 0) * 100);
    const stabPct = Math.round((c.stability || 0) * 100);
    const sourceLine = c.url
      ? `<div style="margin-top:4px;"><strong>Source:</strong> <a href="${String(c.url).replace(/"/g, "&quot;")}" target="_blank" rel="noopener noreferrer" style="color:#93c5fd;">${String(c.source || "reference").replace(/</g, "&lt;")}</a></div>`
      : (c.source ? `<div style="margin-top:4px;"><strong>Source:</strong> ${String(c.source).replace(/</g, "&lt;")}</div>` : "");
    return `
      <div class="verifact-claim ${claimClass(c.verdict)}">
        <div><strong>${c.verdict}</strong> (${confPct}%)</div>
        <div>${(c.claim || "").replace(/</g, "&lt;")}</div>
        <div style="margin-top:4px;opacity:0.9;"><strong>Reason:</strong> ${reasonForClaim(c)}</div>
        <div style="margin-top:4px;opacity:0.9;"><strong>Uncertainty:</strong> ${uncPct}% &nbsp; <strong>Stability:</strong> ${stabPct}%</div>
        ${c.correction ? `<div style="margin-top:4px;"><strong>Suggested correction:</strong> ${String(c.correction).replace(/</g, "&lt;")}</div>` : ""}
        ${c.evidence ? `<div style="margin-top:4px;"><strong>Support:</strong> ${String(c.evidence).replace(/</g, "&lt;")}</div>` : ""}
        ${sourceLine}
      </div>
    `;
  }).join("");

  panel.innerHTML = `
    <h3>VeriFact Live (${PLATFORM})</h3>
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
    <div><strong>Summary:</strong> ${result.summary || "No summary"}</div>
    ${alerts ? `<hr style="border-color: rgba(255,255,255,0.14);"/><div style="margin-bottom:8px;"><strong>Risk Alerts</strong></div>${alerts}` : ""}
    <hr style="border-color: rgba(255,255,255,0.14);"/>
    ${claims || "<div>No claims extracted</div>"}
  `;
}

async function tick() {
  const messages = getAssistantMessages();
  if (!messages.length) {
    return;
  }

  const latest = messages[messages.length - 1];
  if (!latest || latest.length < 40) {
    return;
  }

  const convoText = messages.slice(-CONVERSATION_WINDOW).join("\n\n");
  if (convoText.length < 40) {
    return;
  }

  const fp = fingerprint(convoText);
  if (fp === lastFingerprint) {
    return;
  }

  lastFingerprint = fp;
  setBeacon("analyzing");

  try {
    const result = await callAnalyzer(convoText);
    if (!result) {
      setBeacon("idle");
      return;
    }
    setBeacon(verdictClass(result.factuality_score || 0));
    pulseTicker(result.alerts || []);
    renderPanel(result);
    highlightClaimsInNode(latestTargetNode, result.flags || []);
  } catch (err) {
    setBeacon("error");
    pulseTicker([{ category: "red_flag", message: "Analyzer unavailable on this machine." }]);
    const panel = ensurePanel();
    if (panel) {
      panel.innerHTML = `<h3>VeriFact Live</h3><div>Analyzer offline: ${String(err)}</div>`;
    }
  }
}

setInterval(() => {
  tick().catch(() => {
    setBeacon("error");
  });
}, POLL_MS);
