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
    --fog-top: #b8c8d8;
    --fog-mid: #d5dde5;
    --fog-bot: #e8ecf0;
    --glass-bg: rgba(255, 255, 255, 0.62);
    --glass-border: rgba(200, 210, 220, 0.80);
    --text-main: #0f1a24;
    --text-soft: #3a4a58;
    --accent: #1a6e88;
    --ok: #1a7a52;
    --warn: #a06a10;
    --bad: #b03030;
}

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
    color: var(--text-main);
}

[data-testid="stAppViewContainer"] *,
[data-testid="stSidebar"] *,
[data-testid="stMarkdownContainer"] *,
label,
p,
span,
h1,
h2,
h3,
h4,
h5,
h6 {
    color: var(--text-main) !important;
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

.vf-row {
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.7rem;
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
    border: 1px solid rgba(160, 175, 190, 0.6) !important;
    background: rgba(255, 255, 255, 0.85) !important;
    color: #0f1a24 !important;
}

div[data-testid="stTextArea"] textarea::placeholder {
    color: #506070 !important;
    opacity: 1 !important;
}

/* Ensure radio, selectbox, slider labels are dark */
div[data-testid="stRadio"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stMultiSelect"] label {
    color: #0f1a24 !important;
    font-weight: 500 !important;
}

/* Expander text must be dark */
[data-testid="stExpander"] summary span {
    color: #0f1a24 !important;
}

/* Tab text */
[data-baseweb="tab"] button {
    color: #0f1a24 !important;
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
    border: 1px solid rgba(180, 195, 210, 0.65);
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.72);
}

[data-testid="stMarkdownContainer"] code {
    font-family: 'IBM Plex Mono', monospace;
    background: rgba(255, 255, 255, 0.56);
    border: 1px solid rgba(255, 255, 255, 0.75);
    border-radius: 6px;
}

[data-baseweb="tab-list"] {
    background: rgba(255, 255, 255, 0.42);
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    padding: 3px;
}

[data-baseweb="tab"] {
    border-radius: 9px;
    font-weight: 650;
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

st.markdown('<div class="vf-card"><div class="vf-row">', unsafe_allow_html=True)
example_choice = st.selectbox(
    "Load example (optional)",
    ["None"] + list(EXAMPLES.keys()),
)

if input_mode == "Ask a question":
    user_input = st.text_area(
        "Enter your question:",
        height=110,
        placeholder="e.g. Tell me about the discovery of penicillin …",
    )
else:
    user_input = st.text_area(
        "Paste LLM-generated text to verify:",
        height=160,
        placeholder="Paste any AI-generated text here …",
    )

if example_choice != "None":
    user_input = EXAMPLES[example_choice]

action_col1, action_col2 = st.columns([3, 1])
with action_col1:
    verify_clicked = st.button("Verify", type="primary", use_container_width=True)
with action_col2:
    clear_clicked = st.button("Clear", use_container_width=True)

st.markdown("</div></div>", unsafe_allow_html=True)

if clear_clicked:
    st.session_state.pop("last_result", None)
    st.session_state.pop("last_input", None)

# ═══════════════════════════════════════════════════════════════════════════
# Run verification
# ═══════════════════════════════════════════════════════════════════════════

if verify_clicked and user_input:
    try:
        with st.spinner("Analysing claims and retrieving evidence …"):
            if input_mode == "Ask a question":
                result: VerificationResult = pipeline.verify_query(user_input)
            else:
                result = pipeline.verify_text(user_input)
            st.session_state["last_result"] = result
            st.session_state["last_input"] = user_input
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

if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    st.markdown("---")

    tab_overview, tab_claims, tab_export = st.tabs(["Overview", "Claims", "Export"])

    with tab_overview:
        c_score, c_chart, c_meta = st.columns([1, 2, 1])

        with c_score:
            score = result.factuality_score
            colour = "#28a745" if score >= 75 else "#ffc107" if score >= 40 else "#dc3545"
            st.markdown(
                f'<div style="text-align:center">'
                f'<span style="font-size:3.2rem;font-weight:700;color:{colour}">'
                f"{score:.0f}</span>"
                f'<br/><span style="color:#4f5f6f">Factuality Score</span></div>',
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
            fig.update_layout(height=280, margin=dict(t=30, b=0, l=0, r=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with c_meta:
            st.metric("Total Claims", result.total_claims)
            st.metric("Processing Time", f"{result.processing_time}s")

        st.subheader("Annotated Output")
        if result.llm_query:
            st.caption(f"Query: *{result.llm_query}*")
        st.markdown(result.annotated_html, unsafe_allow_html=True)

        confidences = [c.confidence * 100 for c in result.claims if c.confidence is not None]
        if confidences:
            fig_dist = px.histogram(
                x=confidences,
                nbins=10,
                labels={"x": "Confidence (%)", "y": "Count"},
                color_discrete_sequence=["steelblue"],
            )
            fig_dist.add_vline(
                x=75, line_dash="dash", line_color="green", annotation_text="Verified"
            )
            fig_dist.add_vline(
                x=40, line_dash="dash", line_color="red", annotation_text="Unreliable"
            )
            fig_dist.update_layout(height=280)
            st.plotly_chart(fig_dist, use_container_width=True)

    with tab_claims:
        _ICONS = {
            "SUPPORTED": "OK",
            "CONTRADICTED": "NO",
            "UNVERIFIABLE": "WARN",
            "NO_EVIDENCE": "NONE",
        }

        filter_col1, filter_col2 = st.columns([2, 2])
        with filter_col1:
            verdict_filter = st.multiselect(
                "Verdicts",
                ["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE", "NO_EVIDENCE"],
                default=["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE", "NO_EVIDENCE"],
            )
        with filter_col2:
            min_conf = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)

        filtered_claims = [
            c
            for c in result.claims
            if (c.verdict in verdict_filter)
            and ((c.confidence is not None and c.confidence >= min_conf) or c.confidence is None)
        ]

        st.caption(f"Showing {len(filtered_claims)} of {len(result.claims)} claims")

        for claim in filtered_claims:
            icon = _ICONS.get(claim.verdict, "?")
            conf = f"{claim.confidence * 100:.0f}%" if claim.confidence is not None else "—"
            label = f"[{icon}] {claim.text[:90]}{'…' if len(claim.text) > 90 else ''} — {conf}"

            with st.expander(label):
                st.markdown(f"**Full claim:** {claim.text}")
                st.markdown(f"**Verdict:** `{claim.verdict}` | **Confidence:** `{conf}`")

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
                                claim.nli_scores.get("entailment", 0.0),
                                claim.nli_scores.get("neutral", 0.0),
                                claim.nli_scores.get("contradiction", 0.0),
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

    with tab_export:
        st.download_button(
            "Download Report (JSON)",
            data=json.dumps(result.report_json, indent=2, ensure_ascii=False),
            file_name="verifactai_report.json",
            mime="application/json",
        )
        st.json(result.report_json)
