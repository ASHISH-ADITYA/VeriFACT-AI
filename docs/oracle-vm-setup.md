# Oracle Cloud Free VM — Complete Setup Guide for VeriFACT AI

## What You Get (Free Forever)

| Spec | Value |
|---|---|
| CPU | 4 ARM Ampere A1 cores (3.0 GHz) |
| RAM | **24 GB** |
| Storage | 200 GB boot volume |
| Network | 1 Gbps, 10 TB/month outbound |
| OS | Ubuntu 22.04 or 24.04 |
| Cost | **$0 forever** (Oracle Always Free tier) |
| Uptime | 24/7, no sleep, no cold start |

This is more powerful than most $25/month cloud VMs. Oracle gives it free to attract enterprise customers.

---

## Step 1: Create Oracle Cloud Account (10 minutes)

1. Go to: **https://cloud.oracle.com/sign-up**

2. Fill in:
   - **Country**: India
   - **Name**: Your full name
   - **Email**: Your email (use Gmail, works best)
   - **Home Region**: Choose **ap-mumbai-1** (Mumbai) or **ap-hyderabad-1** (Hyderabad) — closest to you, lowest latency

3. **Verification**:
   - You need a credit/debit card for identity verification
   - **You will NOT be charged** — Oracle explicitly states free tier never bills
   - A temporary hold of $1 may appear and be reversed

4. Complete email verification and set your password

5. Wait 2-5 minutes for account provisioning. You'll get an email saying "Your Oracle Cloud account is ready"

---

## Step 2: Create the Free VM Instance (10 minutes)

1. Log in to: **https://cloud.oracle.com**

2. Click **"Create a VM instance"** (on the main dashboard) or navigate to:
   - Menu (☰) → Compute → Instances → **Create Instance**

3. Configure the instance:

   **Name**: `verifact-ai`

   **Image and shape**:
   - Click **Edit** next to Image and shape
   - Image: **Canonical Ubuntu 22.04** (or 24.04)
   - Shape: Click **Change Shape**
     - **Shape series**: Ampere (ARM-based)
     - **Shape name**: VM.Standard.A1.Flex
     - **OCPUs**: 4 (slide to 4)
     - **Memory (GB)**: 24 (slide to 24)
   - This should show **Always Free-eligible** badge

   **Networking**:
   - Use the default VCN or create new
   - **Assign a public IPv4 address**: YES (important!)
   - Subnet: Use default (public subnet)

   **Add SSH keys**:
   - Select **Generate a key pair**
   - Click **Save Private Key** — download the `.key` file
   - **SAVE THIS FILE** — you need it to connect. You cannot download it again.
   - Rename it: `oracle-verifact.key`

   **Boot volume**:
   - Keep default (47 GB is fine, can use up to 200 GB free)

4. Click **Create**

5. Wait 2-5 minutes for the instance to be **RUNNING**

6. Copy the **Public IP Address** shown on the instance details page
   - Example: `129.154.xx.xx`

---

## Step 3: Open Firewall Ports (5 minutes)

Your VM needs ports 8765 (API) and optionally 8501 (dashboard) open.

1. On the instance details page, click the **Subnet** link (under Primary VNIC)

2. Click the **Security List** (default security list)

3. Click **Add Ingress Rules** and add these rules:

   **Rule 1 — API Server**:
   - Source Type: CIDR
   - Source CIDR: `0.0.0.0/0`
   - IP Protocol: TCP
   - Destination Port Range: `8765`
   - Description: VeriFACT API

   **Rule 2 — Dashboard** (optional):
   - Source Type: CIDR
   - Source CIDR: `0.0.0.0/0`
   - IP Protocol: TCP
   - Destination Port Range: `8501`
   - Description: VeriFACT Dashboard

4. Click **Add Ingress Rules**

5. Also open the ports in the VM's iptables (do this after SSH in Step 4):
   ```bash
   sudo iptables -I INPUT -p tcp --dport 8765 -j ACCEPT
   sudo iptables -I INPUT -p tcp --dport 8501 -j ACCEPT
   sudo netfilter-persistent save
   ```

---

## Step 4: Connect to Your VM (2 minutes)

On your Mac terminal:

```bash
# Set permissions on the key file
chmod 600 ~/Downloads/oracle-verifact.key

# Connect (replace IP with your instance's public IP)
ssh -i ~/Downloads/oracle-verifact.key ubuntu@YOUR_PUBLIC_IP
```

If it asks "Are you sure you want to continue connecting?" type `yes`.

You should see the Ubuntu prompt:
```
ubuntu@verifact-ai:~$
```

---

## Step 5: Deploy VeriFACT AI (15 minutes — paste this entire block)

Copy and paste this ENTIRE script into your SSH terminal:

```bash
#!/bin/bash
set -e

echo "=========================================="
echo "  VeriFACT AI — Oracle VM Deployment"
echo "=========================================="

# 1. System updates
echo "[1/8] Updating system..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3.11 python3.11-venv python3-pip git curl

# 2. Open firewall
echo "[2/8] Opening firewall ports..."
sudo iptables -I INPUT -p tcp --dport 8765 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 8501 -j ACCEPT
sudo apt-get install -y -qq iptables-persistent
sudo netfilter-persistent save

# 3. Install Ollama
echo "[3/8] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &>/dev/null &
sleep 5
ollama pull llama3.1:8b

# 4. Clone repo
echo "[4/8] Cloning VeriFACT AI..."
cd ~
git clone https://github.com/ASHISH-ADITYA/VeriFACT-AI.git
cd VeriFACT-AI

# 5. Python setup
echo "[5/8] Setting up Python environment..."
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel -q
pip install -r verifactai/requirements.txt -q
python -m spacy download en_core_web_sm -q

# 6. Configure
echo "[6/8] Configuring..."
cp verifactai/.env.example verifactai/.env
# Set model to llama3.1:8b in .env
sed -i 's/LLM_MODEL=.*/LLM_MODEL=llama3.1:8b/' verifactai/.env

# 7. Build knowledge index
echo "[7/8] Building knowledge index (5000 articles, ~5 min)..."
cd verifactai
PYTHONPATH=. ../venv/bin/python data/build_index.py --wiki-only --max-articles 5000

# 8. Start server
echo "[8/8] Starting server..."
PYTHONPATH=. nohup ../venv/bin/python overlay_server.py > /tmp/verifact.log 2>&1 &

# Wait for ready
for i in $(seq 1 60); do
    sleep 1
    if curl -sf http://localhost:8765/health > /dev/null 2>&1; then
        echo ""
        echo "=========================================="
        echo "  DEPLOYMENT COMPLETE"
        echo "=========================================="
        echo ""
        PUBLIC_IP=$(curl -sf ifconfig.me)
        echo "  API URL: http://$PUBLIC_IP:8765"
        echo "  Health:  http://$PUBLIC_IP:8765/health"
        echo ""
        echo "  Set this in Vercel:"
        echo "  NEXT_PUBLIC_API_URL=http://$PUBLIC_IP:8765"
        echo ""
        echo "  Test:"
        echo "  curl http://$PUBLIC_IP:8765/health"
        echo "=========================================="
        exit 0
    fi
done
echo "Server did not start in 60s. Check /tmp/verifact.log"
```

---

## Step 6: Make Server Survive Reboots (2 minutes)

Create a systemd service so the server starts automatically:

```bash
sudo tee /etc/systemd/system/verifact.service > /dev/null <<'EOF'
[Unit]
Description=VeriFACT AI API Server
After=network.target ollama.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/VeriFACT-AI/verifactai
Environment=PYTHONPATH=/home/ubuntu/VeriFACT-AI/verifactai
Environment=VERIFACT_HOST=0.0.0.0
Environment=PORT=8765
ExecStart=/home/ubuntu/VeriFACT-AI/venv/bin/python overlay_server.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable verifact
sudo systemctl start verifact

# Verify
sudo systemctl status verifact
```

---

## Step 7: Update Vercel Frontend (2 minutes)

1. Go to: https://vercel.com/dashboard
2. Click your project (web-five-mocha-51)
3. Go to **Settings** → **Environment Variables**
4. Update `NEXT_PUBLIC_API_URL` to: `http://YOUR_ORACLE_IP:8765`
5. Go to **Deployments** → click **Redeploy** on the latest deployment

---

## Step 8: Verify Everything Works

From your Mac:
```bash
# Health check
curl http://YOUR_ORACLE_IP:8765/health

# Test the Great Wall claim
curl -X POST http://YOUR_ORACLE_IP:8765/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"The Great Wall of China is in South America.","top_claims":3}'
```

---

## Future Deployments (30 seconds)

After initial setup, updating the code is:

```bash
ssh -i ~/Downloads/oracle-verifact.key ubuntu@YOUR_ORACLE_IP
cd ~/VeriFACT-AI
git pull
sudo systemctl restart verifact
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Can't SSH | Check security list has port 22 open (default: yes) |
| Can't reach port 8765 | Check both OCI security list AND iptables |
| Server crashes | `journalctl -u verifact -f` to see logs |
| Out of memory | Check with `free -h`. 24GB should be plenty. |
| Ollama not running | `sudo systemctl start ollama` |
| Shape not available | Try a different region. Mumbai and Hyderabad usually have ARM capacity. |
| "Out of capacity" error | This is common. Try again in a few hours or try a different AD (Availability Domain). |

## Architecture After Setup

```
User's browser
    │
    ▼
Vercel (free) — serves web UI
    │ HTTPS API calls
    ▼
Oracle VM (free, always on, 24GB RAM)
    ├── VeriFACT API (:8765)
    ├── Ollama LLM (:11434)
    ├── FAISS index (in memory)
    ├── DeBERTa NLI model
    ├── Sentence-transformers
    └── Rule-based validator
```

No cold starts. No build delays. No sleeping. Always on. Free forever.
