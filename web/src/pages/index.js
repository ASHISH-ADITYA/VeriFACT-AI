import { useMemo, useState, useRef, useEffect } from "react";
import Head from "next/head";

const API =
  process.env.NEXT_PUBLIC_API_URL ||
  process.env.NEXT_PUBLIC_VERIFACT_API_BASE ||
  "https://adiashish-verifact-ai.hf.space";

const CHATBOTS = [
  { name: "ChatGPT", url: "https://chatgpt.com", icon: "🤖", color: "#10a37f" },
  { name: "Gemini", url: "https://gemini.google.com", icon: "✨", color: "#4285f4" },
  { name: "Claude", url: "https://claude.ai", icon: "🧠", color: "#d97706" },
  { name: "Grok", url: "https://grok.x.ai", icon: "⚡", color: "#1da1f2" },
  { name: "Copilot", url: "https://copilot.microsoft.com", icon: "💎", color: "#7c3aed" },
  { name: "Perplexity", url: "https://perplexity.ai", icon: "🔍", color: "#20b2aa" },
];

const EXAMPLES = [
  "Albert Einstein invented the telephone in 1876. He was born in Germany and won the Nobel Prize in Physics in 1921.",
  "The Great Wall of China is in South America and is 50 meters long.",
  "Napoleon won the Battle of Waterloo in 1815. The Earth revolves around the Sun.",
];

const VS = {
  SUPPORTED: { bg: "rgba(52,211,153,0.12)", border: "#34d399", label: "Verified", icon: "✓" },
  CONTRADICTED: { bg: "rgba(248,113,113,0.12)", border: "#f87171", label: "False", icon: "✗" },
  UNVERIFIABLE: { bg: "rgba(251,191,36,0.12)", border: "#fbbf24", label: "Unclear", icon: "?" },
  NO_EVIDENCE: { bg: "rgba(156,163,175,0.1)", border: "#9ca3af", label: "No Evidence", icon: "—" },
};

// ── Crystal/Ice CSS-in-JS ──────────────────────────────
const glass = (opacity = 0.08) => ({
  background: `rgba(255,255,255,${opacity})`,
  backdropFilter: "blur(24px) saturate(180%)",
  WebkitBackdropFilter: "blur(24px) saturate(180%)",
  border: "1px solid rgba(255,255,255,0.18)",
  borderRadius: 20,
});

const crystalBtn = (active = false) => ({
  ...glass(active ? 0.18 : 0.08),
  padding: "10px 20px",
  borderRadius: 14,
  cursor: "pointer",
  fontWeight: 700,
  fontSize: "0.85rem",
  color: active ? "#fff" : "rgba(255,255,255,0.8)",
  transition: "all 0.2s ease",
  border: active ? "1px solid rgba(255,255,255,0.35)" : "1px solid rgba(255,255,255,0.12)",
});

export default function Home() {
  const [tab, setTab] = useState("dashboard");
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [chatMsgs, setChatMsgs] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [minConf, setMinConf] = useState(0);
  const chatEndRef = useRef(null);

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [chatMsgs]);

  const analyze = async () => {
    if (!text.trim()) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await fetch(`${API}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.trim(), top_claims: 12 }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setResult(await res.json());
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const sendChat = async () => {
    if (!chatInput.trim()) return;
    const userMsg = chatInput.trim();
    setChatInput("");
    setChatMsgs((m) => [...m, { role: "user", text: userMsg }]);
    setChatLoading(true);
    try {
      const res = await fetch(`${API}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: userMsg, top_claims: 6 }),
      });
      const data = await res.json();
      const flags = (data.flags || []).filter((f) => f.verdict !== "SUPPORTED");
      let reply = `Analyzed ${data.total_claims || 0} claims (Score: ${Math.round(data.factuality_score || 0)}/100).\n\n`;
      if (flags.length) {
        reply += flags.map((f) =>
          `${VS[f.verdict]?.icon || "•"} **${f.verdict}** (${Math.round((f.confidence||0)*100)}%): ${f.claim}\n  Evidence: ${(f.evidence||"").substring(0,150)}`
        ).join("\n\n");
      } else {
        reply += "All claims verified. No hallucinations detected.";
      }
      setChatMsgs((m) => [...m, { role: "assistant", text: reply, data }]);
    } catch (e) {
      setChatMsgs((m) => [...m, { role: "assistant", text: `Error: ${e.message}` }]);
    } finally { setChatLoading(false); }
  };

  const scoreColor = (s) => s >= 75 ? "#34d399" : s >= 40 ? "#fbbf24" : "#f87171";

  const filtered = useMemo(() => {
    if (!result?.flags) return [];
    return result.flags.filter((c) => Math.round((c.confidence || 0) * 100) >= minConf);
  }, [result, minConf]);

  return (
    <>
      <Head>
        <title>VeriFACT AI — Real-Time Hallucination Detection</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <style jsx global>{`
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html, body { height: 100%; }
        body {
          font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif;
          background: linear-gradient(135deg, #0a1628 0%, #0d2137 25%, #0a2a4a 50%, #0d1f3c 75%, #0a1628 100%);
          color: #e8ecf2;
          min-height: 100vh;
          overflow-x: hidden;
        }
        body::before {
          content: '';
          position: fixed;
          inset: 0;
          background:
            radial-gradient(ellipse at 20% 20%, rgba(56,189,248,0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 80%, rgba(139,92,246,0.06) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 50%, rgba(52,211,153,0.04) 0%, transparent 60%);
          pointer-events: none;
          z-index: 0;
        }
        ::selection { background: rgba(56,189,248,0.3); }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 6px; }
        textarea:focus, input:focus { outline: none; border-color: rgba(56,189,248,0.5) !important; }
        @keyframes shimmer {
          0% { background-position: -200% 0; }
          100% { background-position: 200% 0; }
        }
        @keyframes float {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-6px); }
        }
      `}</style>

      <div style={{ position: "relative", zIndex: 1, maxWidth: 960, margin: "0 auto", padding: "1.5rem 1rem" }}>

        {/* ── Header ────────────────────────────────────── */}
        <header style={{ textAlign: "center", marginBottom: 28 }}>
          <div style={{
            display: "inline-flex", width: 56, height: 56, borderRadius: 16,
            background: "linear-gradient(135deg, rgba(56,189,248,0.3), rgba(139,92,246,0.3))",
            ...glass(0.15), alignItems: "center", justifyContent: "center",
            fontSize: 24, marginBottom: 10, animation: "float 3s ease-in-out infinite",
          }}>💎</div>
          <h1 style={{ fontSize: "2.2rem", fontWeight: 800, letterSpacing: "-0.03em",
            background: "linear-gradient(135deg, #38bdf8, #a78bfa, #34d399)",
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          }}>VeriFACT AI</h1>
          <p style={{ color: "rgba(255,255,255,0.5)", fontSize: "0.9rem", marginTop: 4 }}>
            Real-Time Hallucination Detection & Prompt Intelligence
          </p>
        </header>

        {/* ── Chatbot Quick Links ───────────────────────── */}
        <div style={{ display: "flex", justifyContent: "center", gap: 10, marginBottom: 24, flexWrap: "wrap" }}>
          {CHATBOTS.map((bot) => (
            <a key={bot.name} href={bot.url} target="_blank" rel="noreferrer"
              style={{
                ...glass(0.06), padding: "8px 14px", borderRadius: 12,
                display: "flex", alignItems: "center", gap: 6,
                textDecoration: "none", color: "rgba(255,255,255,0.8)",
                fontSize: "0.82rem", fontWeight: 600,
                transition: "all 0.2s ease",
              }}
              onMouseOver={(e) => { e.currentTarget.style.background = `rgba(255,255,255,0.14)`; e.currentTarget.style.borderColor = bot.color; }}
              onMouseOut={(e) => { e.currentTarget.style.background = "rgba(255,255,255,0.06)"; e.currentTarget.style.borderColor = "rgba(255,255,255,0.18)"; }}
            >
              <span style={{ fontSize: "1.1rem" }}>{bot.icon}</span>
              {bot.name}
            </a>
          ))}
        </div>

        {/* ── Tab Navigation ────────────────────────────── */}
        <div style={{ display: "flex", gap: 6, marginBottom: 20, justifyContent: "center" }}>
          {[["dashboard", "🏠 Dashboard"], ["verify", "🔍 Verify"], ["chat", "💬 Chat"], ["about", "ℹ️ About"]].map(([id, label]) => (
            <button key={id} onClick={() => setTab(id)} style={crystalBtn(tab === id)}>{label}</button>
          ))}
        </div>

        {/* ── Dashboard Tab ─────────────────────────────── */}
        {tab === "dashboard" && (
          <div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 14, marginBottom: 20 }}>
              {[
                { label: "Pipeline", value: "Active", sub: "DeBERTa-v3 + FAISS + BM25", color: "#34d399" },
                { label: "Rule Engine", value: "400+", sub: "Hard fact rules", color: "#38bdf8" },
                { label: "NLI Model", value: "DeBERTa", sub: "cross-encoder/nli-deberta-v3-base", color: "#a78bfa" },
                { label: "Knowledge", value: "11K+", sub: "Wikipedia chunks indexed", color: "#fbbf24" },
              ].map((kpi) => (
                <div key={kpi.label} style={{ ...glass(0.06), padding: 18, textAlign: "center" }}>
                  <div style={{ fontSize: "1.8rem", fontWeight: 800, color: kpi.color }}>{kpi.value}</div>
                  <div style={{ fontSize: "0.85rem", fontWeight: 700, marginTop: 4 }}>{kpi.label}</div>
                  <div style={{ fontSize: "0.72rem", color: "rgba(255,255,255,0.4)", marginTop: 2 }}>{kpi.sub}</div>
                </div>
              ))}
            </div>

            <div style={{ ...glass(0.05), padding: 20, marginBottom: 16 }}>
              <h3 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: 10, color: "#38bdf8" }}>Pipeline Architecture</h3>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                {["Claim Decomposition", "FAISS + BM25 Retrieval", "DeBERTa NLI", "Rule Engine", "SelfCheck", "Semantic Entropy", "Confidence Fusion"].map((s) => (
                  <span key={s} style={{
                    ...glass(0.08), padding: "6px 12px", borderRadius: 10,
                    fontSize: "0.78rem", fontWeight: 600, color: "rgba(255,255,255,0.7)",
                  }}>{s}</span>
                ))}
              </div>
            </div>

            <div style={{ ...glass(0.05), padding: 20 }}>
              <h3 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: 10, color: "#a78bfa" }}>Quick Verify</h3>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                {EXAMPLES.map((ex, i) => (
                  <button key={i} onClick={() => { setText(ex); setTab("verify"); }}
                    style={{ ...crystalBtn(false), fontSize: "0.76rem", padding: "8px 14px" }}>
                    Example {i + 1}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ── Verify Tab ────────────────────────────────── */}
        {tab === "verify" && (
          <div>
            <textarea value={text} onChange={(e) => setText(e.target.value)}
              placeholder="Paste AI-generated text here to verify..."
              rows={6}
              style={{
                width: "100%", padding: 16, ...glass(0.06),
                fontSize: "0.95rem", color: "#e8ecf2", resize: "vertical",
              }}
            />
            <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
              <button onClick={analyze} disabled={loading || !text.trim()}
                style={{
                  flex: 1, padding: 14, borderRadius: 14, border: "none",
                  background: loading
                    ? "rgba(255,255,255,0.1)"
                    : "linear-gradient(135deg, rgba(56,189,248,0.3), rgba(139,92,246,0.3))",
                  backdropFilter: "blur(12px)", color: "#fff", fontWeight: 700,
                  fontSize: "1rem", cursor: loading ? "wait" : "pointer",
                  transition: "all 0.2s ease",
                }}>
                {loading ? "⏳ Analyzing..." : "🔍 Verify Claims"}
              </button>
              <button onClick={() => { setText(""); setResult(null); setError(null); }}
                style={crystalBtn(false)}>Clear</button>
            </div>

            {error && (
              <div style={{ marginTop: 14, padding: 14, ...glass(0.06), borderColor: "#f87171", color: "#f87171" }}>
                <strong>Error:</strong> {error}
              </div>
            )}

            {result && (
              <div style={{ marginTop: 20 }}>
                {/* Score card */}
                <div style={{ ...glass(0.08), padding: 20, display: "flex", alignItems: "center", gap: 20, marginBottom: 16 }}>
                  <div style={{ textAlign: "center", minWidth: 80 }}>
                    <div style={{ fontSize: "2.6rem", fontWeight: 800, color: scoreColor(result.factuality_score) }}>
                      {Math.round(result.factuality_score)}
                    </div>
                    <div style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.5)" }}>Score</div>
                  </div>
                  <div style={{ display: "flex", gap: 12 }}>
                    {[
                      { n: result.supported, l: "Verified", c: "#34d399" },
                      { n: result.contradicted, l: "False", c: "#f87171" },
                      { n: (result.unverifiable || 0) + (result.no_evidence || 0), l: "Unclear", c: "#fbbf24" },
                    ].map((x) => (
                      <div key={x.l} style={{ ...glass(0.06), padding: "10px 16px", borderRadius: 12, textAlign: "center" }}>
                        <div style={{ fontSize: "1.4rem", fontWeight: 800, color: x.c }}>{x.n}</div>
                        <div style={{ fontSize: "0.7rem", color: "rgba(255,255,255,0.5)" }}>{x.l}</div>
                      </div>
                    ))}
                  </div>
                  <div style={{ fontSize: "0.8rem", color: "rgba(255,255,255,0.4)", marginLeft: "auto" }}>
                    {result.processing_time?.toFixed(1)}s
                  </div>
                </div>

                {/* Confidence filter */}
                <div style={{ marginBottom: 12, fontSize: "0.82rem", color: "rgba(255,255,255,0.5)" }}>
                  Min confidence: {minConf}%
                  <input type="range" min={0} max={100} step={5} value={minConf}
                    onChange={(e) => setMinConf(Number(e.target.value))}
                    style={{ width: "100%", marginTop: 4 }} />
                </div>

                {/* Claim cards */}
                {filtered.map((c, i) => {
                  const v = VS[c.verdict] || VS.NO_EVIDENCE;
                  return (
                    <div key={i} style={{
                      ...glass(0.05), padding: 16, marginBottom: 10,
                      borderLeft: `3px solid ${v.border}`,
                    }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                        <span style={{ fontWeight: 700, color: v.border }}>
                          {v.icon} {v.label}
                        </span>
                        <span style={{ color: "rgba(255,255,255,0.5)", fontSize: "0.85rem" }}>
                          {Math.round((c.confidence || 0) * 100)}%
                        </span>
                      </div>
                      <div style={{ fontSize: "0.9rem", lineHeight: 1.5, marginBottom: 8 }}>{c.claim}</div>
                      {c.evidence && (
                        <div style={{
                          ...glass(0.04), padding: 10, borderRadius: 12,
                          fontSize: "0.78rem", color: "rgba(255,255,255,0.5)", marginBottom: 4,
                        }}>
                          <strong style={{ color: "rgba(255,255,255,0.7)" }}>Evidence:</strong> {c.evidence.substring(0, 200)}
                        </div>
                      )}
                      {c.source && (
                        <div style={{ fontSize: "0.72rem", color: "rgba(255,255,255,0.3)", fontStyle: "italic" }}>
                          Source: {c.source}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* ── Chat Tab (RAG-powered) ────────────────────── */}
        {tab === "chat" && (
          <div style={{ ...glass(0.05), padding: 20, minHeight: 400, display: "flex", flexDirection: "column" }}>
            <h3 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: 12, color: "#38bdf8" }}>
              💬 VeriFACT Chat — Paste any text to fact-check
            </h3>

            <div style={{ flex: 1, overflowY: "auto", marginBottom: 14, maxHeight: "50vh" }}>
              {chatMsgs.length === 0 && (
                <div style={{ textAlign: "center", padding: 40, color: "rgba(255,255,255,0.3)", fontSize: "0.9rem" }}>
                  Paste any AI response or claim here. VeriFACT will analyze it using RAG + NLI and highlight hallucinations.
                </div>
              )}
              {chatMsgs.map((m, i) => (
                <div key={i} style={{
                  display: "flex", justifyContent: m.role === "user" ? "flex-end" : "flex-start",
                  marginBottom: 10,
                }}>
                  <div style={{
                    ...glass(m.role === "user" ? 0.12 : 0.06),
                    padding: "12px 16px", maxWidth: "85%",
                    borderRadius: m.role === "user" ? "16px 16px 4px 16px" : "16px 16px 16px 4px",
                    fontSize: "0.88rem", lineHeight: 1.55, whiteSpace: "pre-wrap",
                  }}>
                    {m.text}
                  </div>
                </div>
              ))}
              {chatLoading && (
                <div style={{ display: "flex", justifyContent: "flex-start", marginBottom: 10 }}>
                  <div style={{
                    ...glass(0.06), padding: "12px 20px", borderRadius: "16px 16px 16px 4px",
                    fontSize: "0.88rem", color: "rgba(255,255,255,0.5)",
                    background: "linear-gradient(90deg, rgba(255,255,255,0.06) 25%, rgba(255,255,255,0.12) 50%, rgba(255,255,255,0.06) 75%)",
                    backgroundSize: "200% 100%", animation: "shimmer 1.5s infinite",
                  }}>Analyzing claims...</div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            <div style={{ display: "flex", gap: 8 }}>
              <textarea value={chatInput} onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); } }}
                placeholder="Paste text to fact-check..."
                rows={2}
                style={{
                  flex: 1, padding: 12, ...glass(0.08),
                  fontSize: "0.9rem", color: "#e8ecf2", resize: "none",
                }}
              />
              <button onClick={sendChat} disabled={chatLoading || !chatInput.trim()}
                style={{
                  ...crystalBtn(true), padding: "12px 20px",
                  background: "linear-gradient(135deg, rgba(56,189,248,0.25), rgba(139,92,246,0.25))",
                }}>
                {chatLoading ? "⏳" : "🔍"}
              </button>
            </div>
          </div>
        )}

        {/* ── About Tab ─────────────────────────────────── */}
        {tab === "about" && (
          <div style={{ ...glass(0.05), padding: 24 }}>
            <h3 style={{ fontSize: "1.2rem", fontWeight: 700, marginBottom: 14,
              background: "linear-gradient(135deg, #38bdf8, #a78bfa)",
              WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
            }}>About VeriFACT AI</h3>
            <div style={{ fontSize: "0.9rem", lineHeight: 1.7, color: "rgba(255,255,255,0.7)" }}>
              <p style={{ marginBottom: 12 }}>
                VeriFACT AI is a real-time hallucination detection system for AI chatbots.
                It monitors conversations on ChatGPT, Claude, Gemini, and more — flagging
                factual errors with evidence-backed verdicts.
              </p>
              <h4 style={{ color: "#38bdf8", marginBottom: 8 }}>Pipeline</h4>
              <ul style={{ paddingLeft: 20, marginBottom: 14 }}>
                <li>Claim Decomposition (spaCy + LLM)</li>
                <li>Hybrid Retrieval (FAISS dense + BM25 sparse + RRF)</li>
                <li>DeBERTa-v3 NLI (entailment/contradiction/neutral)</li>
                <li>Rule-based validator (400+ hard facts)</li>
                <li>SelfCheckGPT consistency scoring</li>
                <li>Semantic entropy uncertainty estimation</li>
                <li>Bayesian confidence fusion (5 weighted signals)</li>
              </ul>
              <h4 style={{ color: "#a78bfa", marginBottom: 8 }}>Tech Stack</h4>
              <ul style={{ paddingLeft: 20 }}>
                <li>Backend: Python, HuggingFace Spaces (free CPU)</li>
                <li>Models: DeBERTa-v3, MiniLM-L6, Sentence-Transformers</li>
                <li>Frontend: Next.js, Vercel</li>
                <li>Extension: Chrome (ChatGPT, Claude, Gemini)</li>
                <li>LLM: Groq Llama 3.1 (free), Ollama (local)</li>
              </ul>
            </div>
          </div>
        )}

        {/* ── Footer ────────────────────────────────────── */}
        <footer style={{ textAlign: "center", marginTop: 32, fontSize: "0.72rem", color: "rgba(255,255,255,0.2)" }}>
          VeriFACT AI · Built by Aditya Ashish · DeBERTa NLI + FAISS + Wikipedia · Free & Open Source
        </footer>
      </div>
    </>
  );
}
