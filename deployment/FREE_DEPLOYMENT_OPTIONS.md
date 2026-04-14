# Completely Free Deployment Options

This project has a heavy ML backend (FAISS + NLI + optional LLM), so fully free hosting requires a split architecture.

## Option 1: Vercel (Frontend) + Hugging Face Spaces (Backend)

### Why this works
- Vercel hosts only the web frontend (free tier).
- Hugging Face Space hosts backend container (free CPU tier, sleeping behavior possible).

### Steps
1. Deploy frontend folder `web/` to Vercel.
2. Deploy backend Docker to Hugging Face Space:
   - Expose `GET /health`
   - Expose `POST /analyze`
3. Set Vercel env var:
   - `NEXT_PUBLIC_API_URL=https://<your-space>.hf.space`
4. Set backend CORS env:
   - `VERIFACT_CORS_ORIGINS=https://<your-vercel-domain>`
5. Optional auth:
   - `VERIFACT_API_TOKEN=<long-random-token>`

### Pros
- 100% free stack possible.
- Fast frontend globally via Vercel.

### Cons
- Free backend can sleep and cold-start.
- Compute limits on free CPU.

## Option 2: Vercel (Frontend) + Oracle Always Free VM (Backend)

### Why this works
- Oracle VM can run Docker Compose continuously for free-tier resources.

### Steps
1. Deploy frontend `web/` on Vercel.
2. On Oracle VM:
   - clone repo
   - run `docker compose -f docker-compose.prod.yml up -d --build`
3. Put backend behind Cloudflare (free) for HTTPS and DNS.
4. Set Vercel env var `NEXT_PUBLIC_API_URL` to backend URL.
5. Set `VERIFACT_CORS_ORIGINS` to Vercel domain.

### Pros
- Stable always-on backend.
- Better for demos/production.

### Cons
- Requires VM setup and maintenance.

## Recommended for your goals
Use Option 1 first (fastest free launch), then move to Option 2 if traffic or uptime expectations increase.
