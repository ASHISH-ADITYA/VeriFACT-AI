import { useMemo, useState, useRef, useEffect } from "react";
import Head from "next/head";

const API = "https://adiashish-verifact-ai.hf.space";

const CHATBOTS = [
  { name: "ChatGPT", url: "https://chatgpt.com", logo: "/ai-logos/chatgpt.png", c: "#10a37f" },
  { name: "Gemini", url: "https://gemini.google.com", logo: "/ai-logos/gemini.webp", c: "#4285f4" },
  { name: "Claude", url: "https://claude.ai", logo: "/ai-logos/claude.webp", c: "#d97706" },
  { name: "Grok", url: "https://grok.x.ai", logo: "/ai-logos/grok.webp", c: "#1da1f2" },
  { name: "Copilot", url: "https://copilot.microsoft.com", logo: "/ai-logos/copilot.webp", c: "#7c3aed" },
  { name: "Perplexity", url: "https://perplexity.ai", logo: "/ai-logos/perplexity.webp", c: "#20b2aa" },
];

const EXAMPLES = [
  "Albert Einstein invented the telephone in 1876. He won the Nobel Prize in Physics in 1921.",
  "The Great Wall of China is in South America and is 50 meters long.",
  "Napoleon won the Battle of Waterloo. The Earth revolves around the Sun.",
];

const VS = {
  SUPPORTED: { c: "#059669", bg: "rgba(5,150,105,0.1)", icon: "\u2713", label: "Verified" },
  CONTRADICTED: { c: "#dc2626", bg: "rgba(220,38,38,0.1)", icon: "\u2717", label: "False" },
  UNVERIFIABLE: { c: "#d97706", bg: "rgba(217,119,6,0.1)", icon: "?", label: "Unclear" },
  NO_EVIDENCE: { c: "#6b7280", bg: "rgba(107,114,128,0.1)", icon: "\u2014", label: "No Evidence" },
};

export default function Home() {
  const [tab, setTab] = useState("dashboard");
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [chatMsgs, setChatMsgs] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const chatEnd = useRef(null);

  useEffect(() => { chatEnd.current?.scrollIntoView({ behavior: "smooth" }); }, [chatMsgs]);

  const analyze = async () => {
    if (!text.trim()) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const r = await fetch(`${API}/analyze`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: text.trim(), top_claims: 12 }) });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setResult(await r.json());
    } catch (e) { setError(e.message); } finally { setLoading(false); }
  };

  const sendChat = async () => {
    if (!chatInput.trim()) return;
    const msg = chatInput.trim(); setChatInput("");
    setChatMsgs(m => [...m, { role: "user", text: msg }]); setChatLoading(true);
    try {
      const r = await fetch(`${API}/analyze`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: msg, top_claims: 6 }) });
      const d = await r.json();
      const issues = (d.flags || []).filter(f => f.verdict !== "SUPPORTED");
      let reply = `Analyzed ${d.total_claims || 0} claims (Score: ${Math.round(d.factuality_score || 0)}/100)\n\n`;
      reply += issues.length ? issues.map(f => `${VS[f.verdict]?.icon||"*"} ${f.verdict} (${Math.round((f.confidence||0)*100)}%): ${f.claim}\n   Evidence: ${(f.evidence||"").substring(0,120)}`).join("\n\n") : "All claims verified. No hallucinations detected.";
      setChatMsgs(m => [...m, { role: "ai", text: reply }]);
    } catch (e) { setChatMsgs(m => [...m, { role: "ai", text: `Error: ${e.message}` }]); } finally { setChatLoading(false); }
  };

  const sc = s => s >= 75 ? "#059669" : s >= 40 ? "#d97706" : "#dc2626";

  const filtered = useMemo(() => result?.flags || [], [result]);

  return (
    <>
      <Head>
        <title>VeriFACT AI</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet" />
      </Head>

      <style jsx global>{`
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html, body { height: 100%; }
        body {
          font-family: 'Inter', -apple-system, sans-serif;
          background: linear-gradient(160deg, #e8f5e9 0%, #e0f7fa 20%, #e1f5fe 40%, #f1f8e9 60%, #fff8e1 80%, #e0f2f1 100%);
          color: #1a2e1a;
          min-height: 100vh;
        }
        body::before {
          content: '';
          position: fixed; inset: 0; z-index: 0; pointer-events: none;
          background:
            radial-gradient(ellipse at 25% 15%, rgba(0,210,211,0.12) 0%, transparent 50%),
            radial-gradient(ellipse at 75% 85%, rgba(255,193,7,0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 50%, rgba(76,175,80,0.06) 0%, transparent 60%);
        }
        ::selection { background: rgba(0,210,211,0.25); }
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-thumb { background: rgba(0,150,136,0.2); border-radius: 5px; }
        textarea:focus, input:focus { outline: none; box-shadow: 0 0 0 2px rgba(0,210,211,0.3); }
        @keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-5px)} }
        @keyframes shimmer { 0%{background-position:-200% 0} 100%{background-position:200% 0} }
        @keyframes glow { 0%,100%{filter:brightness(1)} 50%{filter:brightness(1.15)} }
      `}</style>

      <div style={{ position: "relative", zIndex: 1, maxWidth: 880, margin: "0 auto", padding: "2rem 1rem" }}>

        {/* Header */}
        <header style={{ textAlign: "center", marginBottom: 24 }}>
          <img src="/logo.png" alt="VeriFACT AI" style={{
            width: 72, height: 72, objectFit: "contain",
            marginBottom: 10, animation: "float 3s ease-in-out infinite",
            filter: "drop-shadow(0 6px 20px rgba(0,100,180,0.3))",
          }} />
          <h1 style={{
            fontSize: "2.4rem", fontWeight: 800, letterSpacing: "-0.03em",
            background: "linear-gradient(135deg, #00897b, #00bcd4, #4caf50, #ffc107)",
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          }}>VeriFACT AI</h1>
          <p style={{ color: "#5d7a5d", fontSize: "0.88rem", marginTop: 4 }}>
            Real-Time Hallucination Detection & Verification
          </p>
        </header>

        {/* Chatbot Icons — round liquid glass buttons with SVG logos */}
        <div style={{ display: "flex", justifyContent: "center", gap: 12, marginBottom: 20, flexWrap: "wrap" }}>
          {CHATBOTS.map(b => (
            <a key={b.name} href={b.url} target="_blank" rel="noreferrer" title={b.name} style={{
              display: "flex", alignItems: "center", justifyContent: "center",
              width: 50, height: 50, borderRadius: "50%",
              textDecoration: "none", color: b.c,
              background: "rgba(255,255,255,0.65)",
              border: "2px solid rgba(255,255,255,0.8)",
              backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)",
              transition: "all 0.25s ease",
              boxShadow: `0 4px 14px rgba(0,0,0,0.06), inset 0 2px 0 rgba(255,255,255,0.7), inset 0 -1px 3px rgba(0,0,0,0.04)`,
            }}
            onMouseOver={e => { e.currentTarget.style.background = "rgba(255,255,255,0.9)"; e.currentTarget.style.borderColor = b.c; e.currentTarget.style.boxShadow = `0 6px 20px ${b.c}30, inset 0 2px 0 rgba(255,255,255,0.8)`; e.currentTarget.style.transform = "scale(1.1)"; }}
            onMouseOut={e => { e.currentTarget.style.background = "rgba(255,255,255,0.65)"; e.currentTarget.style.borderColor = "rgba(255,255,255,0.8)"; e.currentTarget.style.boxShadow = "0 4px 14px rgba(0,0,0,0.06), inset 0 2px 0 rgba(255,255,255,0.7), inset 0 -1px 3px rgba(0,0,0,0.04)"; e.currentTarget.style.transform = "scale(1)"; }}
            ><img src={b.logo} alt={b.name} style={{ width: 30, height: 30, objectFit: "contain", borderRadius: "50%", pointerEvents: "none" }} /></a>
          ))}
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 6, marginBottom: 20, justifyContent: "center" }}>
          {[["dashboard","Dashboard"],["verify","Verify"],["chat","Chat"],["about","About"]].map(([id,label]) => (
            <button key={id} onClick={() => setTab(id)} style={{
              padding: "9px 20px", borderRadius: 12, fontSize: "0.85rem", fontWeight: 700,
              cursor: "pointer", transition: "all 0.2s ease",
              background: tab === id ? "rgba(0,210,211,0.15)" : "rgba(255,255,255,0.5)",
              border: tab === id ? "1.5px solid rgba(0,210,211,0.4)" : "1px solid rgba(0,150,136,0.12)",
              color: tab === id ? "#00695c" : "#5d7a5d",
              backdropFilter: "blur(8px)", WebkitBackdropFilter: "blur(8px)",
              boxShadow: tab === id ? "0 4px 16px rgba(0,210,211,0.12)" : "0 2px 8px rgba(0,0,0,0.03)",
            }}>{label}</button>
          ))}
        </div>

        {/* ── Dashboard ────────────────────────────────── */}
        {tab === "dashboard" && (
          <div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12, marginBottom: 18 }}>
              {[
                { label: "Pipeline", value: "Active", sub: "DeBERTa + FAISS + BM25", color: "#00897b" },
                { label: "Rules", value: "400+", sub: "Hard fact checks", color: "#00bcd4" },
                { label: "Knowledge", value: "11K+", sub: "Wikipedia chunks", color: "#4caf50" },
                { label: "Models", value: "3", sub: "NLI + Embed + Rerank", color: "#ffc107" },
              ].map(k => (
                <div key={k.label} style={{
                  padding: 18, textAlign: "center", borderRadius: 16,
                  background: "rgba(255,255,255,0.6)", border: "1px solid rgba(0,150,136,0.1)",
                  backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)",
                  boxShadow: "0 4px 20px rgba(0,0,0,0.04)",
                }}>
                  <div style={{ fontSize: "1.8rem", fontWeight: 800, color: k.color }}>{k.value}</div>
                  <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "#2e5a3e", marginTop: 2 }}>{k.label}</div>
                  <div style={{ fontSize: "0.72rem", color: "#7a9a7a", marginTop: 2 }}>{k.sub}</div>
                </div>
              ))}
            </div>

            <div style={{
              padding: 20, borderRadius: 18,
              background: "rgba(255,255,255,0.55)", border: "1px solid rgba(0,150,136,0.1)",
              backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)",
              marginBottom: 14, boxShadow: "0 4px 24px rgba(0,0,0,0.04)",
            }}>
              <h3 style={{ fontSize: "0.95rem", fontWeight: 700, color: "#00695c", marginBottom: 10 }}>Pipeline Stages</h3>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                {["Claim Decomposition","Hybrid Retrieval","DeBERTa NLI","Rule Engine","SelfCheck","Entropy","Confidence Fusion"].map(s => (
                  <span key={s} style={{
                    padding: "5px 12px", borderRadius: 10, fontSize: "0.76rem", fontWeight: 600,
                    background: "rgba(0,210,211,0.08)", border: "1px solid rgba(0,210,211,0.15)",
                    color: "#2e5a3e",
                  }}>{s}</span>
                ))}
              </div>
            </div>

            <div style={{
              padding: 20, borderRadius: 18,
              background: "rgba(255,255,255,0.55)", border: "1px solid rgba(0,150,136,0.1)",
              backdropFilter: "blur(12px)", boxShadow: "0 4px 24px rgba(0,0,0,0.04)",
            }}>
              <h3 style={{ fontSize: "0.95rem", fontWeight: 700, color: "#00695c", marginBottom: 10 }}>Quick Verify</h3>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                {EXAMPLES.map((ex, i) => (
                  <button key={i} onClick={() => { setText(ex); setTab("verify"); }} style={{
                    padding: "8px 14px", borderRadius: 10, fontSize: "0.78rem", fontWeight: 600,
                    background: "rgba(255,255,255,0.7)", border: "1px solid rgba(0,150,136,0.12)",
                    color: "#2e5a3e", cursor: "pointer", transition: "all 0.2s",
                  }}>Example {i + 1}</button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ── Verify ───────────────────────────────────── */}
        {tab === "verify" && (
          <div>
            <textarea value={text} onChange={e => setText(e.target.value)}
              placeholder="Paste AI-generated text here to verify..." rows={5}
              style={{
                width: "100%", padding: 16, borderRadius: 16, fontSize: "0.92rem",
                background: "rgba(255,255,255,0.65)", border: "1px solid rgba(0,150,136,0.15)",
                backdropFilter: "blur(12px)", color: "#1a2e1a", resize: "vertical",
              }} />
            <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
              <button onClick={analyze} disabled={loading || !text.trim()} style={{
                flex: 1, padding: 14, borderRadius: 14, border: "none", fontWeight: 700, fontSize: "0.95rem",
                background: loading ? "rgba(0,150,136,0.15)" : "linear-gradient(135deg, rgba(0,210,211,0.25), rgba(76,175,80,0.2))",
                border: "1.5px solid rgba(0,210,211,0.3)",
                color: "#00695c", cursor: loading ? "wait" : "pointer",
                backdropFilter: "blur(8px)", transition: "all 0.2s",
              }}>{loading ? "Analyzing..." : "Verify Claims"}</button>
              <button onClick={() => { setText(""); setResult(null); setError(null); }} style={{
                padding: "14px 20px", borderRadius: 14, fontWeight: 600,
                background: "rgba(255,255,255,0.6)", border: "1px solid rgba(0,150,136,0.12)",
                color: "#5d7a5d", cursor: "pointer",
              }}>Clear</button>
            </div>

            {error && <div style={{ marginTop: 12, padding: 14, borderRadius: 14, background: "rgba(220,38,38,0.08)", border: "1px solid rgba(220,38,38,0.2)", color: "#dc2626", fontSize: "0.88rem" }}><strong>Error:</strong> {error}</div>}

            {result && (
              <div style={{ marginTop: 18 }}>
                <div style={{
                  display: "flex", alignItems: "center", gap: 18, padding: 20, borderRadius: 18,
                  background: "rgba(255,255,255,0.6)", border: "1px solid rgba(0,150,136,0.1)",
                  backdropFilter: "blur(12px)", marginBottom: 14, boxShadow: "0 4px 20px rgba(0,0,0,0.04)",
                }}>
                  <div style={{ textAlign: "center", minWidth: 70 }}>
                    <div style={{ fontSize: "2.4rem", fontWeight: 800, color: sc(result.factuality_score) }}>{Math.round(result.factuality_score)}</div>
                    <div style={{ fontSize: "0.72rem", color: "#7a9a7a" }}>Score</div>
                  </div>
                  <div style={{ display: "flex", gap: 10 }}>
                    {[{ n: result.supported, l: "Verified", c: "#059669" }, { n: result.contradicted, l: "False", c: "#dc2626" }, { n: (result.unverifiable||0)+(result.no_evidence||0), l: "Unclear", c: "#d97706" }].map(x => (
                      <div key={x.l} style={{
                        padding: "10px 16px", borderRadius: 12, textAlign: "center",
                        background: "rgba(255,255,255,0.5)", border: "1px solid rgba(0,0,0,0.05)",
                      }}>
                        <div style={{ fontSize: "1.3rem", fontWeight: 800, color: x.c }}>{x.n}</div>
                        <div style={{ fontSize: "0.7rem", color: "#7a9a7a" }}>{x.l}</div>
                      </div>
                    ))}
                  </div>
                  <div style={{ fontSize: "0.78rem", color: "#7a9a7a", marginLeft: "auto" }}>{result.processing_time?.toFixed(1)}s</div>
                </div>

                {filtered.map((c, i) => {
                  const v = VS[c.verdict] || VS.NO_EVIDENCE;
                  return (
                    <div key={i} style={{
                      padding: 16, marginBottom: 8, borderRadius: 16,
                      background: "rgba(255,255,255,0.55)", borderLeft: `3px solid ${v.c}`,
                      border: `1px solid rgba(0,0,0,0.05)`, borderLeftWidth: 3, borderLeftColor: v.c,
                      backdropFilter: "blur(8px)", boxShadow: "0 2px 12px rgba(0,0,0,0.03)",
                    }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                        <span style={{ fontWeight: 700, color: v.c }}>{v.icon} {v.label}</span>
                        <span style={{ color: "#7a9a7a", fontSize: "0.85rem" }}>{Math.round((c.confidence||0)*100)}%</span>
                      </div>
                      <div style={{ fontSize: "0.88rem", lineHeight: 1.5, marginBottom: 6 }}>{c.claim}</div>
                      {c.evidence && <div style={{
                        padding: 10, borderRadius: 12, fontSize: "0.78rem", color: "#5d7a5d",
                        background: "rgba(0,150,136,0.04)", border: "1px solid rgba(0,150,136,0.08)",
                      }}><strong>Evidence:</strong> {c.evidence.substring(0, 200)}</div>}
                      {c.source && <div style={{ fontSize: "0.7rem", color: "#9ab09a", fontStyle: "italic", marginTop: 4 }}>Source: {c.source}</div>}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* ── Chat ─────────────────────────────────────── */}
        {tab === "chat" && (
          <div style={{
            padding: 20, borderRadius: 18, minHeight: 420, display: "flex", flexDirection: "column",
            background: "rgba(255,255,255,0.55)", border: "1px solid rgba(0,150,136,0.1)",
            backdropFilter: "blur(12px)", boxShadow: "0 4px 24px rgba(0,0,0,0.04)",
          }}>
            <h3 style={{ fontSize: "0.95rem", fontWeight: 700, color: "#00695c", marginBottom: 12 }}>VeriFACT Chat — Paste any text to fact-check</h3>
            <div style={{ flex: 1, overflowY: "auto", marginBottom: 12, maxHeight: "45vh" }}>
              {chatMsgs.length === 0 && <div style={{ textAlign: "center", padding: 40, color: "#9ab09a", fontSize: "0.88rem" }}>Paste any AI response here. VeriFACT will analyze it and highlight hallucinations.</div>}
              {chatMsgs.map((m, i) => (
                <div key={i} style={{ display: "flex", justifyContent: m.role === "user" ? "flex-end" : "flex-start", marginBottom: 8 }}>
                  <div style={{
                    padding: "10px 14px", maxWidth: "82%",
                    borderRadius: m.role === "user" ? "14px 14px 4px 14px" : "14px 14px 14px 4px",
                    fontSize: "0.86rem", lineHeight: 1.5, whiteSpace: "pre-wrap",
                    background: m.role === "user" ? "rgba(0,210,211,0.1)" : "rgba(255,255,255,0.7)",
                    border: `1px solid ${m.role === "user" ? "rgba(0,210,211,0.2)" : "rgba(0,0,0,0.05)"}`,
                  }}>{m.text}</div>
                </div>
              ))}
              {chatLoading && <div style={{ display: "flex", marginBottom: 8 }}><div style={{
                padding: "10px 18px", borderRadius: "14px 14px 14px 4px", fontSize: "0.86rem", color: "#7a9a7a",
                background: "linear-gradient(90deg, rgba(0,210,211,0.06) 25%, rgba(0,210,211,0.12) 50%, rgba(0,210,211,0.06) 75%)",
                backgroundSize: "200% 100%", animation: "shimmer 1.5s infinite",
              }}>Analyzing...</div></div>}
              <div ref={chatEnd} />
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <textarea value={chatInput} onChange={e => setChatInput(e.target.value)}
                onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); } }}
                placeholder="Paste text to fact-check..." rows={2}
                style={{
                  flex: 1, padding: 10, borderRadius: 12, fontSize: "0.88rem",
                  background: "rgba(255,255,255,0.7)", border: "1px solid rgba(0,150,136,0.12)",
                  color: "#1a2e1a", resize: "none",
                }} />
              <button onClick={sendChat} disabled={chatLoading || !chatInput.trim()} style={{
                padding: "10px 18px", borderRadius: 12, fontWeight: 700, border: "none",
                background: "linear-gradient(135deg, rgba(0,210,211,0.2), rgba(76,175,80,0.15))",
                border: "1px solid rgba(0,210,211,0.3)", color: "#00695c", cursor: "pointer",
              }}>{chatLoading ? "..." : "Verify"}</button>
            </div>
          </div>
        )}

        {/* ── About ────────────────────────────────────── */}
        {tab === "about" && (
          <div style={{
            padding: 24, borderRadius: 18,
            background: "rgba(255,255,255,0.55)", border: "1px solid rgba(0,150,136,0.1)",
            backdropFilter: "blur(12px)", boxShadow: "0 4px 24px rgba(0,0,0,0.04)",
          }}>
            <h3 style={{
              fontSize: "1.2rem", fontWeight: 700, marginBottom: 14,
              background: "linear-gradient(135deg, #00897b, #00bcd4, #4caf50)",
              WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
            }}>About VeriFACT AI</h3>
            <div style={{ fontSize: "0.88rem", lineHeight: 1.7, color: "#3a5a3a" }}>
              <p style={{ marginBottom: 12 }}>VeriFACT AI is a real-time hallucination detection system for AI chatbots. It monitors conversations on ChatGPT, Claude, Gemini, and more.</p>
              <h4 style={{ color: "#00695c", marginBottom: 6 }}>Pipeline</h4>
              <ul style={{ paddingLeft: 18, marginBottom: 12 }}>
                <li>Claim Decomposition (spaCy + Groq LLM)</li>
                <li>Hybrid Retrieval (FAISS + BM25 + Reciprocal Rank Fusion)</li>
                <li>DeBERTa-v3 NLI (entailment / contradiction)</li>
                <li>400+ hardcoded fact rules</li>
                <li>SelfCheckGPT + Semantic Entropy</li>
                <li>Bayesian confidence fusion</li>
              </ul>
              <h4 style={{ color: "#00695c", marginBottom: 6 }}>Stack</h4>
              <ul style={{ paddingLeft: 18 }}>
                <li>Backend: Python, HuggingFace Spaces</li>
                <li>Frontend: Next.js, Vercel</li>
                <li>Extension: Chrome (ChatGPT, Claude, Gemini)</li>
                <li>Models: DeBERTa-v3, MiniLM-L6, Groq Llama 3.1</li>
              </ul>
            </div>
          </div>
        )}

        <footer style={{ textAlign: "center", marginTop: 28, fontSize: "0.7rem", color: "#9ab09a" }}>
          VeriFACT AI &middot; Built by Aditya Ashish &middot; Free & Open Source
        </footer>
      </div>
    </>
  );
}
