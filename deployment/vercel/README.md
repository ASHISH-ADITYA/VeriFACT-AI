# Vercel Free Deployment Plan

## What to host on Vercel
- Host a lightweight frontend (static/Next.js) that submits text and renders verification results.
- Do not run heavy FAISS/NLI/LLM inference in Vercel serverless functions.

## Required Backend
- Deploy `verifact-api` container from `docker-compose.prod.yml` on a container host.
- Expose `POST /analyze` and `GET /health` publicly over HTTPS.

## Frontend contract
Request:
```json
{ "text": "...", "top_claims": 6 }
```
Response:
- `factuality_score`
- `summary`
- `flags[]`
- `alerts[]`

## Env Vars (Frontend)
- `NEXT_PUBLIC_API_URL=https://your-api-domain`

## Completely Free Backend Options

### Option A (recommended): Hugging Face Spaces (Docker)
- Host the backend container on a free CPU Space.
- Expose `/health` and `/analyze` over HTTPS.
- Use `NEXT_PUBLIC_API_URL` in Vercel to point to Space URL.

### Option B: Oracle Cloud Always Free VM
- Deploy `docker-compose.prod.yml` on an always-free VM.
- Put Cloudflare proxy in front for HTTPS + DNS.
- Keep Vercel frontend fully free.

## CORS
Allow your Vercel domain in overlay server allowed origins.

## Backend Security (recommended)
- `VERIFACT_ENV=production`
- `VERIFACT_REQUIRE_AUTH=1`
- `VERIFACT_API_TOKEN=<long-random-token>`
- `VERIFACT_CORS_ORIGINS=https://<your-vercel-domain>`

If auth is enabled, send header `X-VeriFact-Token` from trusted clients.
For public browser clients, prefer strict CORS + rate limiting and keep token auth for private/internal deployments.

## Free-tier note
This gives you a free public frontend on Vercel while keeping model-heavy services in containers.
