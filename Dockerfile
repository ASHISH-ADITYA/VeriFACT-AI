# ── Stage 1: Dependencies ─────────────────────────────────
FROM python:3.11-slim AS deps

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

COPY verifactai/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pytest && \
    python -m spacy download en_core_web_sm

# Pre-download ML models (cached in this layer)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-base'); AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-base')"

# ── Stage 2: Application ─────────────────────────────────
FROM deps AS app

WORKDIR /app

# Copy application code
COPY verifactai/ ./verifactai/
COPY verifactai/.env.example ./verifactai/.env

# Non-root user for security
RUN groupadd -r verifact && useradd -r -g verifact -d /app verifact && \
    chown -R verifact:verifact /app
USER verifact

# Environment
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434
ENV PYTHONPATH=/app/verifactai
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8501 8765

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Entrypoint: run both dashboard and overlay server
COPY <<'ENTRYPOINT' /app/entrypoint.sh
#!/bin/bash
set -e

echo "Starting VeriFACT AI..."

# Start overlay server in background
cd /app/verifactai && python overlay_server.py &
OVERLAY_PID=$!

# Start Streamlit dashboard in foreground
cd /app && streamlit run verifactai/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false &
STREAMLIT_PID=$!

# Graceful shutdown
trap "kill $OVERLAY_PID $STREAMLIT_PID 2>/dev/null; exit 0" SIGTERM SIGINT

echo "Dashboard:  http://localhost:8501"
echo "API:        http://localhost:8765"

wait
ENTRYPOINT

USER root
RUN chmod +x /app/entrypoint.sh
USER verifact

CMD ["/app/entrypoint.sh"]
