"""
VeriFactAI — Streamlit Dashboard.

Production-grade UI for LLM hallucination detection.
Run: streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Apply conservative thread/runtime defaults before loading heavy native deps.
from utils.runtime_safety import apply_runtime_safety_defaults

apply_runtime_safety_defaults()

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config  # noqa: E402
from core.pipeline import VeriFactPipeline, VerificationResult  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Page configuration
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="VeriFactAI — Hallucination Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --fog-top: #cdd8e3;
    --fog-mid: #e9eef2;
    --fog-bot: #f5f7f9;
    --glass-bg: rgba(255, 255, 255, 0.48);
    --glass-border: rgba(255, 255, 255, 0.68);
    --text-main: #17222e;
    --text-soft: #546371;
    --accent: #2f8da9;
    --ok: #2f8f66;
    --warn: #c3872b;
    --bad: #c44545;
}

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
    color: var(--text-main);
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(900px 500px at 10% 5%, rgba(255, 255, 255, 0.78), transparent 65%),
        radial-gradient(700px 380px at 95% 10%, rgba(179, 200, 217, 0.35), transparent 70%),
        linear-gradient(150deg, var(--fog-top), var(--fog-mid) 40%, var(--fog-bot) 100%);
}

[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}

[data-testid="stSidebar"] {
    background: rgba(248, 250, 252, 0.55);
    border-right: 1px solid rgba(255, 255, 255, 0.62);
    backdrop-filter: blur(16px) saturate(130%);
    -webkit-backdrop-filter: blur(16px) saturate(130%);
}

.vf-hero {
    padding: 1.1rem 1.2rem;
    border-radius: 18px;
    border: 1px solid var(--glass-border);
    background: linear-gradient(140deg, rgba(255, 255, 255, 0.68), rgba(255, 255, 255, 0.34));
    backdrop-filter: blur(14px) saturate(130%);
    -webkit-backdrop-filter: blur(14px) saturate(130%);
    box-shadow: 0 18px 50px rgba(35, 54, 72, 0.14);
    margin-bottom: 0.75rem;
}

.vf-sub {
    color: var(--text-soft);
    margin-top: 0.25rem;
    letter-spacing: 0.01em;
}

.vf-card {
    padding: 1rem;
    border-radius: 16px;
    border: 1px solid var(--glass-border);
    background: var(--glass-bg);
    backdrop-filter: blur(12px) saturate(125%);
    -webkit-backdrop-filter: blur(12px) saturate(125%);
    box-shadow: 0 16px 42px rgba(33, 53, 74, 0.1);
}

.verified      {background:rgba(47, 143, 102, 0.18);padding:2px 8px;border-radius:8px;font-weight:600}
.contradicted  {background:rgba(196, 69, 69, 0.18);padding:2px 8px;border-radius:8px;font-weight:600}
.unverifiable  {background:rgba(195, 135, 43, 0.2);padding:2px 8px;border-radius:8px;font-weight:600}
.no-evidence   {background:rgba(90, 102, 114, 0.2);padding:2px 8px;border-radius:8px;font-weight:600}

div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.46);
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    padding: 0.6rem 0.8rem;
}

div[data-testid="stMetricValue"] {
    font-size: 2.2rem !important;
    color: var(--text-main);
}

div[data-testid="stTextArea"] textarea,
div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    border-radius: 14px !important;
    border: 1px solid rgba(255, 255, 255, 0.7) !important;
    background: rgba(255, 255, 255, 0.62) !important;
}

div.stButton > button {
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.62);
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.82), rgba(220, 232, 241, 0.72));
    color: var(--text-main);
    font-weight: 650;
    transition: transform 0.16s ease, box-shadow 0.18s ease;
}

div.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 28px rgba(40, 70, 95, 0.16);
}

div.stDownloadButton > button {
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.62);
}

[data-testid="stExpander"] {
    border: 1px solid rgba(255, 255, 255, 0.65);
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.48);
}

[data-testid="stMarkdownContainer"] code {
    font-family: 'IBM Plex Mono', monospace;
    background: rgba(255, 255, 255, 0.56);
    border: 1px solid rgba(255, 255, 255, 0.75);
    border-radius: 6px;
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# Pipeline initialisation (cached across reruns)
# ═══════════════════════════════════════════════════════════════════════════


@st.cache_resource(show_spinner="Loading VeriFactAI models …")
def _load_pipeline() -> VeriFactPipeline:
    return VeriFactPipeline(Config())


pipeline = _load_pipeline()

# ═══════════════════════════════════════════════════════════════════════════
# Example data
# ═══════════════════════════════════════════════════════════════════════════

EXAMPLES = {
    "Mixed facts (Einstein)": (
        "Albert Einstein invented the telephone in 1876. "
        "He was born in Germany and won the Nobel Prize in Physics in 1921 "
        "for his work on the photoelectric effect."
    ),
    "Medical claims": (
        "Aspirin is commonly used as a blood thinner and was first synthesized "
        "in 1897 by Felix Hoffmann at Bayer. Side effects include stomach "
        "bleeding and Reye's syndrome in children."
    ),
    "Science facts": (
        "The speed of light is approximately 300,000 km/s. Water boils at "
        "100°C at sea level. The human body has 206 bones. DNA was "
        "discovered by Watson and Crick in 1953."
    ),
    "Deliberate hallucinations": (
        "The Great Wall of China was built in 1920 by the Japanese government. "
        "It is located in South America and is approximately 50 meters long. "
        "It was designated a UNESCO World Heritage Site in 1987."
    ),
}

# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("⚙️ Settings")
    input_mode = st.radio(
        "Input mode",
        ["Ask a question", "Paste LLM text"],
        help="Choose whether to query an LLM or verify pre-existing text.",
    )
    st.markdown("---")
    st.markdown("### How VeriFactAI works")
    st.markdown(
        "1. **Decompose** — extract atomic factual claims\n"
        "2. **Retrieve** — search trusted knowledge (Wikipedia, PubMed)\n"
        "3. **Verify** — NLI model judges each claim\n"
        "4. **Score** — Bayesian confidence fusion\n"
        "5. **Annotate** — colour-coded output with citations"
    )
    st.markdown("---")

    # Startup diagnostics
    st.markdown("### System Status")

    def _check_ollama() -> bool:
        import urllib.request

        try:
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
            return True
        except Exception:
            return False

    def _check_faiss_index() -> str:
        p = Path(pipeline.config.retrieval.index_path)
        if p.exists():
            mb = p.stat().st_size / (1024 * 1024)
            return f"{mb:.0f} MB"
        return ""

    ollama_ok = _check_ollama()
    idx_size = _check_faiss_index()

    st.markdown(f"- Ollama: {'running' if ollama_ok else '**offline** — run `ollama serve`'}")
    st.markdown(f"- Model: `{pipeline.config.llm.model}`")
    st.markdown(
        f"- FAISS index: {idx_size if idx_size else '**missing** — run `python data/build_index.py`'}"
    )
    st.markdown(f"- Profile: `{pipeline.config.active_profile.value}`")

    st.markdown("---")
    st.markdown("### Verification Core")
    st.markdown("- Architecture: **HRA-v1** (Hybrid Retrieval + NLI + Confidence Fusion)")
    st.markdown("- Spec: `assets/HYBRID_RESEARCH_ARCHITECTURE.md`")
    st.markdown("---")
    st.caption("Built with DeBERTa-v3 NLI · FAISS · Sentence-Transformers")

# ═══════════════════════════════════════════════════════════════════════════
# Main content
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div class="vf-hero">
      <h1 style="margin:0; font-size:2.1rem;">VeriFactAI</h1>
      <p class="vf-sub">Minimal verification dashboard for LLM hallucination detection and evidence grounding.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Input row
col_input, col_examples = st.columns([3, 1])

with col_input:
    st.markdown('<div class="vf-card">', unsafe_allow_html=True)
    if input_mode == "Ask a question":
        user_input = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="e.g. Tell me about the discovery of penicillin …",
        )
    else:
        user_input = st.text_area(
            "Paste LLM-generated text to verify:",
            height=140,
            placeholder="Paste any AI-generated text here …",
        )
    st.markdown("</div>", unsafe_allow_html=True)

with col_examples:
    st.markdown('<div class="vf-card">', unsafe_allow_html=True)
    st.markdown("**Try an example**")
    for label, text in EXAMPLES.items():
        if st.button(label, use_container_width=True):
            st.session_state["_example_text"] = text
    st.markdown("</div>", unsafe_allow_html=True)

# Handle example button
if "_example_text" in st.session_state:
    user_input = st.session_state.pop("_example_text")

# ═══════════════════════════════════════════════════════════════════════════
# Run verification
# ═══════════════════════════════════════════════════════════════════════════

if st.button("🔍 **Verify**", type="primary", use_container_width=True) and user_input:
    try:
        with st.spinner("Analysing claims and retrieving evidence …"):
            if input_mode == "Ask a question":
                result: VerificationResult = pipeline.verify_query(user_input)
            else:
                result = pipeline.verify_text(user_input)
    except Exception as exc:
        st.error(f"Verification failed: {exc}")
        st.info(
            "**Recovery steps (local-first):**\n"
            "1. Ensure Ollama is running: `ollama serve`\n"
            "2. Ensure model is pulled: `ollama pull qwen2.5:3b-instruct`\n"
            "3. Switch to **Paste LLM text** mode (skips LLM generation)\n"
            "4. Check `verifactai.log` for full trace"
        )
        st.stop()

    st.markdown("---")

    # ── Row 1: Score · Breakdown · Meta ──────────────────────────────
    c_score, c_chart, c_meta = st.columns([1, 2, 1])

    with c_score:
        score = result.factuality_score
        colour = "#28a745" if score >= 75 else "#ffc107" if score >= 40 else "#dc3545"
        st.markdown(
            f'<div style="text-align:center">'
            f'<span style="font-size:3.2rem;font-weight:700;color:{colour}">'
            f"{score:.0f}</span>"
            f'<br/><span style="color:#666">Factuality Score</span></div>',
            unsafe_allow_html=True,
        )

    with c_chart:
        fig = go.Figure(
            go.Pie(
                labels=["Verified", "Contradicted", "Uncertain", "No Evidence"],
                values=[
                    result.supported,
                    result.contradicted,
                    result.unverifiable,
                    result.no_evidence,
                ],
                marker_colors=["#28a745", "#dc3545", "#ffc107", "#6c757d"],
                hole=0.45,
                textinfo="label+percent",
            )
        )
        fig.update_layout(
            height=280,
            margin=dict(t=30, b=0, l=0, r=0),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c_meta:
        st.metric("Total Claims", result.total_claims)
        st.metric("Processing Time", f"{result.processing_time}s")

    # ── Row 2: Annotated text ────────────────────────────────────────
    st.subheader("Annotated Output")
    if result.llm_query:
        st.caption(f"Query: *{result.llm_query}*")
    st.markdown(result.annotated_html, unsafe_allow_html=True)

    # ── Row 3: Detailed claim cards ──────────────────────────────────
    st.subheader("Claim Analysis")

    _ICONS = {
        "SUPPORTED": "✅",
        "CONTRADICTED": "❌",
        "UNVERIFIABLE": "⚠️",
        "NO_EVIDENCE": "❓",
    }

    for claim in result.claims:
        icon = _ICONS.get(claim.verdict, "❓")
        conf = f"{claim.confidence * 100:.0f}%" if claim.confidence is not None else "—"
        label = f"{icon}  **{claim.text[:90]}{'…' if len(claim.text) > 90 else ''}**  — {conf}"

        with st.expander(label):
            st.markdown(f"**Full claim:** {claim.text}")
            st.markdown(f"**Verdict:** `{claim.verdict}` &nbsp; **Confidence:** `{conf}`")

            if claim.best_evidence:
                st.markdown(f"**Best evidence:** {claim.best_evidence.text[:300]}")
                src = f"{claim.best_evidence.title} ({claim.best_evidence.source})"
                st.markdown(f"**Source:** {src}")
                if claim.best_evidence.url:
                    st.markdown(f"[Open source ↗]({claim.best_evidence.url})")

            if claim.verdict == "CONTRADICTED" and claim.correction:
                st.error(f"**Suggested correction:** {claim.correction}")

            if claim.nli_scores:
                fig_nli = go.Figure(
                    go.Bar(
                        x=[
                            claim.nli_scores["entailment"],
                            claim.nli_scores["neutral"],
                            claim.nli_scores["contradiction"],
                        ],
                        y=["Entailment", "Neutral", "Contradiction"],
                        orientation="h",
                        marker_color=["#28a745", "#ffc107", "#dc3545"],
                    )
                )
                fig_nli.update_layout(
                    height=140,
                    margin=dict(t=0, b=0, l=0, r=0),
                    xaxis=dict(range=[0, 1]),
                )
                st.plotly_chart(fig_nli, use_container_width=True)

    # ── Row 4: Confidence distribution ───────────────────────────────
    st.subheader("Confidence Distribution")
    confidences = [c.confidence * 100 for c in result.claims if c.confidence is not None]
    if confidences:
        fig_dist = px.histogram(
            x=confidences,
            nbins=10,
            labels={"x": "Confidence (%)", "y": "Count"},
            color_discrete_sequence=["steelblue"],
        )
        fig_dist.add_vline(
            x=75,
            line_dash="dash",
            line_color="green",
            annotation_text="Verified",
        )
        fig_dist.add_vline(
            x=40,
            line_dash="dash",
            line_color="red",
            annotation_text="Unreliable",
        )
        fig_dist.update_layout(height=280)
        st.plotly_chart(fig_dist, use_container_width=True)

    # ── Export ────────────────────────────────────────────────────────
    st.download_button(
        "📥 Download Report (JSON)",
        data=json.dumps(result.report_json, indent=2, ensure_ascii=False),
        file_name="verifactai_report.json",
        mime="application/json",
    )
