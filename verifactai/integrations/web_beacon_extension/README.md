# VeriFact Beacon Extension (MVP)

This extension overlays a small VeriFact beacon on supported chatbot websites and sends the latest assistant response to your local VeriFact analyzer server.

Supported websites:
- chatgpt.com
- chat.openai.com
- claude.ai
- gemini.google.com

## 1) Start the local analyzer server

From the project root:

```bash
cd verifactai
../.venv/bin/python overlay_server.py
```

Health check:

```bash
curl http://127.0.0.1:8765/health
```

## 2) Load extension in Chrome

1. Open `chrome://extensions`
2. Enable Developer mode
3. Click `Load unpacked`
4. Select this folder: `verifactai/integrations/web_beacon_extension`

## 3) Use

1. Open ChatGPT, Claude, or Gemini web app
2. Wait for assistant response
3. VeriFact beacon appears at mid-right
4. Click beacon to open analysis panel:
   - Factuality score
   - Confidence
   - Claim verdicts (supported / contradicted / unverifiable)
   - Live risk alerts (hallucination / bias / red-flag)
   - Reason, correction, and source link per flagged claim

5. During analysis, compact glass-style squircle chips appear below the VF beacon with live warnings.

## Notes

- This is an MVP parser and may require selector updates if site DOM changes.
- The extension analyzes a rolling conversation window (latest assistant messages).
- Native desktop/mobile app overlays are out-of-scope for this browser extension; this targets web UI.

## Desktop App Mode (Recommended)

For a native app-window experience (not a plain browser tab), use:

```bash
cd /Users/adityaashish/Desktop/ENGINEERING_PROJECT_II/verifactai
./desktop/launch_verifact_desktop.command
```

Optional auto-start at login (macOS):

```bash
/Users/adityaashish/Desktop/ENGINEERING_PROJECT_II/verifactai/desktop/install_autostart.command
```

### Important

- Keep VeriFact Desktop running (or auto-started), with analyzer active.
- Reload the extension after updates from `chrome://extensions`.
- The beacon appears on supported chatbot websites in Chromium-based browsers (Chrome/Edge/Brave).
- Safari requires a separate Safari Web Extension packaging flow (not enabled by default in this MVP).
