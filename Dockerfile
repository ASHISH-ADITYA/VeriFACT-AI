# VeriFACT AI — Production Docker Image
# Models are pre-downloaded into the image so cold start is fast.

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY verifactai/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Pre-download ML models into image (avoids runtime download)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-base'); AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-base')"

# Copy app
COPY verifactai/ ./verifactai/
COPY verifactai/.env.example ./verifactai/.env

ENV PYTHONPATH=/app/verifactai
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LLM_PROVIDER=none
ENV PORT=8765

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -sf http://localhost:${PORT}/health || exit 1

# Startup: auto-builds index if missing, then starts API server
CMD ["python", "verifactai/startup.py"]
