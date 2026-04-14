# Deployment Guide

## Local Development (Recommended)

### Prerequisites
- Python 3.10+ (3.11 recommended)
- [Ollama](https://ollama.com) installed
- Chrome browser (for extension)
- 16 GB RAM minimum

### Quick Start
```bash
make install          # Create venv, install deps, download spaCy model
ollama pull llama3.1:8b
cp verifactai/.env.example verifactai/.env
make index-dev        # Build 5K article dev index (~5 min)
make smoke            # Verify everything works
make dashboard        # Launch at http://localhost:8501
```

### Full Index Build
```bash
make index            # Full 200K article index (~30 min, 541 MB)
```

## Docker

### Build and Run
```bash
make docker           # Build image
make docker-up        # Start with docker-compose
```

**Note**: Ollama must run on the host. The container connects via `host.docker.internal`.

### Environment Variables
```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
LLM_MODEL=llama3.1:8b
```

## Browser Extension

1. Open `chrome://extensions`
2. Enable **Developer mode** (top right)
3. Click **Load unpacked**
4. Select `verifactai/integrations/web_beacon_extension/`
5. Navigate to ChatGPT or Claude — beacon appears bottom-right
6. Ensure overlay server is running: `make server`

## Platform Notes

### macOS (Apple Silicon)
- MPS acceleration is automatic for NLI inference
- Ollama runs natively on M-series chips
- Full performance with 16 GB RAM

### Linux
- CPU inference by default (no MPS)
- Consider GPU passthrough in Docker for CUDA acceleration
- All other features work identically

### Windows
- Use WSL2 for best experience
- Install Ollama for Windows
- Chrome extension works natively
