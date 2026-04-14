# Deployment Enhancement Roadmap

## Goals
1. Free Vercel-compatible public web surface.
2. Reactive and clean dashboard UX with only essential controls.
3. Fix readability and frosted-glass visual quality.
4. Containerized deployment for RAG + API + dashboard.
5. Deployment readiness for Indus Appstore release.

## Reality Check (Vercel Free)
Vercel Free is ideal for static frontend and lightweight serverless endpoints, but not ideal for long-running Python ML inference and FAISS workloads.

Recommended architecture:
- Vercel (Free): public web shell and API gateway UI.
- Container backend: Overlay API + RAG pipeline + model services.
- Optional managed host for backend container: Render/Fly/self-host VM.

## Completed Enhancements
- Dashboard readability/frost and reactive tabbed UX updated in `verifactai/app.py`.
- Production container orchestration split into API/dashboard/ollama services in `docker-compose.prod.yml`.
- Domain benchmark and discriminator paths integrated for stronger release QA.

## Next Implementation Steps
1. Add Vercel frontend shell:
- Build a simple frontend that calls `/analyze` from the deployed API.
- Configure allowed origins for Vercel domain in overlay server.

2. Harden production backend:
- Add reverse proxy (Caddy/Nginx) + TLS.
- Add auth token enforcement (`VERIFACT_API_TOKEN`) in production.
- Add rate limiting and request logging.

3. CI/CD:
- Build and push image to GHCR.
- Deploy with one-click compose script.
- Smoke test and health checks post-deploy.

4. App store readiness (Indus):
- Wrap web app in Android shell (Capacitor preferred).
- Add privacy policy, support URL, and terms.
- Integrate production API URL and app icons.

## Production Commands
```bash
# Build and run full stack
cd /Users/adityaashish/Desktop/ENGINEERING_PROJECT_II
docker compose -f docker-compose.prod.yml up -d --build

# Verify services
curl http://localhost:8765/health
open http://localhost:8501
```
