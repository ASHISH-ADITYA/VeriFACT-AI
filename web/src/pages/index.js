import { useMemo, useState } from "react";
import Head from "next/head";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  process.env.NEXT_PUBLIC_VERIFACT_API_BASE ||
  "http://127.0.0.1:8765";

const EXAMPLES = {
  "Einstein mixed":
    "Albert Einstein invented the telephone in 1876. He was born in Germany and won the Nobel Prize in Physics in 1921.",
  "Medical":
    "Aspirin is commonly used as a blood thinner and was first synthesized in 1897 by Felix Hoffmann at Bayer.",
  "Deliberate hallucination":
    "The Great Wall of China is in South America and is 50 meters long.",
};

const VERDICT_STYLE = {
  SUPPORTED: { bg: "rgba(26, 122, 82, 0.15)", border: "#1a7a52", label: "Supported" },
  CONTRADICTED: { bg: "rgba(176, 48, 48, 0.15)", border: "#b03030", label: "Contradicted" },
  UNVERIFIABLE: { bg: "rgba(160, 106, 16, 0.15)", border: "#a06a10", label: "Unverifiable" },
  NO_EVIDENCE: { bg: "rgba(90, 100, 110, 0.15)", border: "#5a646e", label: "No Evidence" },
};

export default function Home() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("overview");
  const [minConfidence, setMinConfidence] = useState(0);
  const [verdictFilter, setVerdictFilter] = useState([
    "SUPPORTED",
    "CONTRADICTED",
    "UNVERIFIABLE",
    "NO_EVIDENCE",
  ]);

  const analyze = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.trim(), top_claims: 10 }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || `HTTP ${res.status}`);
      }
      setResult(await res.json());
    } catch (e) {
      setError(e.message || "Failed to reach backend");
    } finally {
      setLoading(false);
    }
  };

  const scoreColor = (s) => (s >= 75 ? "#1a7a52" : s >= 40 ? "#a06a10" : "#b03030");

  const filteredFlags = useMemo(() => {
    if (!result?.flags) return [];
    return result.flags.filter((c) => {
      const confPct = Math.round((c.confidence || 0) * 100);
      return verdictFilter.includes(c.verdict) && confPct >= minConfidence;
    });
  }, [result, verdictFilter, minConfidence]);

  const toggleVerdict = (v) => {
    setVerdictFilter((current) =>
      current.includes(v) ? current.filter((x) => x !== v) : [...current, v]
    );
  };

  return (
    <>
      <Head>
        <title>VeriFACT AI</title>
        <meta name="description" content="Real-time LLM hallucination detection" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div style={{ maxWidth: 720, margin: "0 auto", padding: "2rem 1rem", fontFamily: "system-ui, -apple-system, sans-serif", color: "#0f1a24" }}>

        <h1 style={{ fontSize: "2rem", fontWeight: 800, marginBottom: 4 }}>
          VeriFACT AI
        </h1>
        <p style={{ color: "#3a4a58", marginBottom: "1.5rem" }}>
          Paste any AI-generated text to verify its factual accuracy.
        </p>

        <div style={{ display: "flex", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
          {Object.entries(EXAMPLES).map(([k, v]) => (
            <button
              key={k}
              onClick={() => {
                setText(v);
                setError(null);
              }}
              style={{
                border: "1px solid #c0cad4",
                background: "#ffffff",
                color: "#0f1a24",
                borderRadius: 999,
                padding: "0.35rem 0.7rem",
                fontSize: "0.78rem",
                cursor: "pointer",
              }}
            >
              {k}
            </button>
          ))}
        </div>

        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste LLM output here..."
          rows={6}
          style={{
            width: "100%", padding: "0.8rem", borderRadius: 12,
            border: "1px solid #c0cad4", fontSize: "0.95rem",
            fontFamily: "inherit", resize: "vertical", background: "#fff",
          }}
        />

        <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
          <button
            onClick={analyze}
            disabled={loading || !text.trim()}
            style={{
              flex: 1, padding: "0.7rem", borderRadius: 10,
              border: "none", background: loading ? "#8aa" : "#1a6e88",
              color: "#fff", fontWeight: 700, fontSize: "1rem",
              cursor: loading ? "wait" : "pointer",
            }}
          >
            {loading ? "Analyzing..." : "Verify"}
          </button>
          <button
            onClick={() => { setText(""); setResult(null); setError(null); }}
            style={{
              padding: "0.7rem 1.2rem", borderRadius: 10,
              border: "1px solid #c0cad4", background: "#f5f7f9",
              fontWeight: 600, cursor: "pointer",
            }}
          >
            Clear
          </button>
        </div>

        {error && (
          <div style={{
            marginTop: 16, padding: "0.8rem", borderRadius: 10,
            background: "rgba(176, 48, 48, 0.1)", border: "1px solid #b03030",
            color: "#b03030", fontSize: "0.9rem",
          }}>
            <strong>Error:</strong> {error}
            <br />
            <small>Ensure the VeriFACT backend is running at {API_URL}</small>
          </div>
        )}

        {result && (
          <div style={{ marginTop: 24 }}>
            <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
              {[
                ["overview", "Overview"],
                ["claims", "Claims"],
                ["alerts", "Alerts"],
              ].map(([id, label]) => (
                <button
                  key={id}
                  onClick={() => setActiveTab(id)}
                  style={{
                    border: `1px solid ${activeTab === id ? "#1a6e88" : "#c0cad4"}`,
                    background: activeTab === id ? "#e5f2f7" : "#fff",
                    color: "#0f1a24",
                    borderRadius: 10,
                    padding: "0.45rem 0.75rem",
                    fontWeight: 700,
                    cursor: "pointer",
                  }}
                >
                  {label}
                </button>
              ))}
            </div>

            {activeTab === "overview" && (
              <>
                <div style={{
                  display: "flex", alignItems: "center", gap: 20,
                  padding: "1rem", borderRadius: 14,
                  background: "rgba(255,255,255,0.8)",
                  border: "1px solid #d0d8e0", marginBottom: 16,
                }}>
                  <div style={{ textAlign: "center", minWidth: 80 }}>
                    <div style={{ fontSize: "2.4rem", fontWeight: 800, color: scoreColor(result.factuality_score) }}>
                      {Math.round(result.factuality_score)}
                    </div>
                    <div style={{ fontSize: "0.75rem", color: "#3a4a58" }}>Score</div>
                  </div>
                  <div style={{ fontSize: "0.85rem", color: "#3a4a58", lineHeight: 1.6 }}>
                    <strong>{result.total_claims}</strong> claims analyzed in <strong>{result.processing_time?.toFixed(1)}s</strong><br />
                    {result.supported} supported · {result.contradicted} contradicted · {result.unverifiable + result.no_evidence} uncertain
                  </div>
                </div>
              </>
            )}

            {activeTab === "alerts" && (
              <div style={{ marginBottom: 16 }}>
                <h3 style={{ fontSize: "1rem", marginBottom: 8 }}>Alerts</h3>
                {result.alerts && result.alerts.length > 0 ? (
                  result.alerts.map((a, i) => (
                    <div key={i} style={{
                      padding: "0.6rem 0.8rem", borderRadius: 10, marginBottom: 6,
                      background: a.category === "hallucination" ? "rgba(176,48,48,0.1)" : "rgba(160,106,16,0.1)",
                      borderLeft: `3px solid ${a.category === "hallucination" ? "#b03030" : "#a06a10"}`,
                      fontSize: "0.85rem",
                    }}>
                      <strong>{a.severity.toUpperCase()}</strong>: {a.message}
                    </div>
                  ))
                ) : (
                  <div style={{ color: "#5a6a79", fontSize: "0.88rem" }}>No alerts in this response.</div>
                )}
              </div>
            )}

            {activeTab === "claims" && (
              <>
                <h3 style={{ fontSize: "1rem", marginBottom: 8 }}>Claims</h3>
                <div style={{ marginBottom: 10, display: "grid", gap: 8 }}>
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                    {Object.keys(VERDICT_STYLE).map((v) => (
                      <button
                        key={v}
                        onClick={() => toggleVerdict(v)}
                        style={{
                          border: `1px solid ${verdictFilter.includes(v) ? "#1a6e88" : "#c0cad4"}`,
                          background: verdictFilter.includes(v) ? "#e5f2f7" : "#fff",
                          color: "#0f1a24",
                          borderRadius: 999,
                          padding: "0.3rem 0.65rem",
                          fontSize: "0.76rem",
                          cursor: "pointer",
                        }}
                      >
                        {VERDICT_STYLE[v].label}
                      </button>
                    ))}
                  </div>
                  <label style={{ fontSize: "0.82rem", color: "#3a4a58" }}>
                    Minimum confidence: {minConfidence}%
                  </label>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    step={5}
                    value={minConfidence}
                    onChange={(e) => setMinConfidence(Number(e.target.value))}
                  />
                </div>

                {filteredFlags.map((c, i) => {
              const vs = VERDICT_STYLE[c.verdict] || VERDICT_STYLE.NO_EVIDENCE;
              return (
                <div key={i} style={{
                  padding: "0.7rem 0.8rem", borderRadius: 10, marginBottom: 8,
                  background: vs.bg, borderLeft: `3px solid ${vs.border}`,
                  fontSize: "0.88rem",
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <strong>{vs.label}</strong>
                    <span style={{ color: "#3a4a58" }}>{Math.round((c.confidence || 0) * 100)}%</span>
                  </div>
                  <div>{c.claim}</div>
                  {c.evidence && (
                    <div style={{ marginTop: 6, fontSize: "0.8rem", color: "#3a4a58" }}>
                      Evidence: {c.evidence.substring(0, 180)}...
                      {c.url && <> · <a href={c.url} target="_blank" rel="noreferrer" style={{ color: "#1a6e88" }}>Source</a></>}
                    </div>
                  )}
                  {c.correction && (
                    <div style={{ marginTop: 6, fontSize: "0.82rem", color: "#b03030" }}>
                      Correction: {c.correction}
                    </div>
                  )}
                </div>
              );
                })}

                {filteredFlags.length === 0 && (
                  <div style={{ color: "#5a6a79", fontSize: "0.88rem" }}>
                    No claims match the active filters.
                  </div>
                )}
              </>
            )}
          </div>
        )}

        <footer style={{ marginTop: 48, textAlign: "center", fontSize: "0.75rem", color: "#6a7a8a" }}>
          VeriFACT AI · DeBERTa NLI + FAISS + Sentence-Transformers · Local-first, zero cost
        </footer>
      </div>
    </>
  );
}
