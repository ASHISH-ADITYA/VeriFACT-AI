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
- `NEXT_PUBLIC_VERIFACT_API_BASE=https://your-api-domain`

## CORS
Allow your Vercel domain in overlay server allowed origins.

## Free-tier note
This gives you a free public frontend on Vercel while keeping model-heavy services in containers.
