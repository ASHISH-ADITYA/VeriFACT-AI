FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY verifactai/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pytest && \
    python -m spacy download en_core_web_sm

# Pre-download ML models (cached in image)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-base')"

# Copy application
COPY verifactai/ ./verifactai/
COPY verifactai/.env.example ./verifactai/.env

# Set Ollama to use host network (user must run ollama on host)
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434
ENV PYTHONPATH=/app/verifactai

EXPOSE 8501 8765

# Default: run Streamlit dashboard
CMD ["streamlit", "run", "verifactai/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
