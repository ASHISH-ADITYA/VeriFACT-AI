# Deploy VeriFACT AI — Share With Anyone

## The Goal

After these steps, you get a URL like `https://verifact-ai.vercel.app` that **anyone can open on any device** — phone, tablet, laptop — and use VeriFACT AI immediately. No install needed.

## Architecture

```
Any device (browser)
    │
    ▼
Vercel (free) — serves the web UI
    │ API calls
    ▼
Railway (free) — runs Python backend + ML models + FAISS index
```

## Step-by-Step

### 1. Deploy Backend to Railway (5 minutes)

Railway gives you 500 free hours/month. The backend runs the Python API with FAISS + DeBERTa.

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Create project and deploy from this repo
cd /path/to/VeriFACT-AI
railway init
railway up
```

Or use the dashboard:
1. Go to https://railway.app
2. Click "New Project" → "Deploy from GitHub Repo"
3. Select `ASHISH-ADITYA/VeriFACT-AI`
4. Railway will detect the `Dockerfile` and `railway.toml` automatically
5. Wait for build (~5-10 min, it downloads ML models)
6. Get your backend URL: e.g. `https://verifact-ai-production.up.railway.app`

**Set environment variables in Railway dashboard:**
```
PORT=8765
LLM_PROVIDER=ollama
VERIFACT_INDEX_SIZE=1000
VERIFACT_CORS_ORIGINS=https://verifact-ai.vercel.app
```

Note: Without Ollama on Railway, the system uses spaCy fallback for claim decomposition. The verification pipeline (FAISS + DeBERTa NLI) works fully without Ollama.

### 2. Deploy Frontend to Vercel (3 minutes)

```bash
cd web
npm install
```

1. Go to https://vercel.com
2. Click "Add New Project" → Import from GitHub → Select `VeriFACT-AI`
3. Set **Root Directory** to `web`
4. Add environment variable:
   ```
   NEXT_PUBLIC_API_URL=https://your-railway-url.up.railway.app
   ```
5. Click Deploy

That's it. You get a URL like `https://verifact-ai.vercel.app`.

### 3. Share

Send the Vercel URL to anyone. It works on:
- Any browser (Chrome, Safari, Firefox, Edge)
- Any device (phone, tablet, laptop, desktop)
- Any OS (Windows, Mac, Linux, iOS, Android)

No install. No exe. No setup. Just a link.

## Alternative: Run Locally and Share on Your Network

If you want to skip cloud deployment and share with people on the same WiFi:

```bash
# Start backend (binds to 0.0.0.0 — accessible from other devices)
cd verifactai
python overlay_server.py

# Output will show:
#   Local:   http://127.0.0.1:8765
#   Network: http://192.168.1.42:8765
#
# Share the Network URL with other devices on your WiFi
```

Then start the dashboard (also network-accessible):
```bash
streamlit run app.py --server.address 0.0.0.0
# Access from other devices at http://192.168.1.42:8501
```

## Cost

| Service | Cost | Limits |
|---|---|---|
| Vercel (frontend) | **Free** | Unlimited deploys, 100GB bandwidth |
| Railway (backend) | **Free** | 500 hours/month, 512MB RAM |

## FAQ

**Q: Will the free tier be enough?**
A: For demo and capstone presentation, yes. Railway's 500 free hours = ~20 days of continuous running. The NLI model fits in 512MB RAM. If you need more, Railway Pro is $5/month.

**Q: What about Ollama?**
A: Cloud deployment doesn't need Ollama. The claim decomposition falls back to spaCy (rule-based sentence splitting). The core verification (FAISS retrieval + DeBERTa NLI) works fully without any LLM.

**Q: How big is the FAISS index on Railway?**
A: On first deploy, the startup script auto-builds a 1000-article index (~2 min). You can set `VERIFACT_INDEX_SIZE=5000` for a larger index. The full 200K index needs more RAM than the free tier provides.

**Q: Can I use a custom domain?**
A: Yes. Both Vercel and Railway support custom domains on free tier.
