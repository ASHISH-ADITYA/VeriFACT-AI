# VeriFACT AI — Production Docker Image
# Pre-built FAISS index + ML models baked in. Zero cold start.

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY verifactai/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Pre-download ALL ML models (avoids runtime download)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-base'); AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-base')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy app
COPY verifactai/ ./verifactai/
COPY verifactai/.env.example ./verifactai/.env

# Copy pre-built FAISS index (if exists) — avoids 15-min startup build
COPY verifactai/data/prebuild_index/ ./verifactai/data/index/
COPY verifactai/data/prebuild_metadata/ ./verifactai/data/metadata/

ENV PYTHONPATH=/app/verifactai
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LLM_PROVIDER=none
ENV PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:${PORT}/health || exit 1

# Startup: skips index build (pre-built), starts API immediately
CMD ["python", "verifactai/startup.py"]
