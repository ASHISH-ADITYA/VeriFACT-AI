#!/usr/bin/env python3
"""
VeriFact Overlay Server

Local HTTP endpoint used by the browser beacon extension.
Runs on 127.0.0.1:8765 and exposes:
  - GET /health
  - POST /analyze
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from config import Config
from core.pipeline import VeriFactPipeline
from utils.runtime_safety import apply_runtime_safety_defaults

apply_runtime_safety_defaults()


_PIPELINE: VeriFactPipeline | None = None
_PIPELINE_LOCK = threading.Lock()
_API_TOKEN = os.environ.get("VERIFACT_API_TOKEN", "").strip()

_ALLOWED_ORIGIN_PREFIXES = (
    "https://chatgpt.com",
    "https://chat.openai.com",
    "https://claude.ai",
    "https://gemini.google.com",
)


def get_pipeline() -> VeriFactPipeline:
    global _PIPELINE
    with _PIPELINE_LOCK:
        if _PIPELINE is None:
            _PIPELINE = VeriFactPipeline(Config())
        return _PIPELINE


def summarize(result: Any) -> str:
    if result.total_claims == 0:
        return "No factual claims detected in this message."
    return (
        f"{result.supported} supported, {result.contradicted} contradicted, "
        f"{result.unverifiable} unverifiable, {result.no_evidence} without evidence."
    )


_BIAS_CUE_WORDS = {
    "obviously",
    "clearly",
    "everyone knows",
    "always",
    "never",
    "all of them",
    "none of them",
    "superior",
    "inferior",
    "lazy",
    "stupid",
    "primitive",
    "civilized",
}


def _contains_bias_cue(text: str) -> bool:
    lower = text.lower()
    return any(cue in lower for cue in _BIAS_CUE_WORDS)


def _severity_for_claim(verdict: str, confidence: float) -> str:
    v = (verdict or "").upper()
    if v == "CONTRADICTED":
        return "high" if confidence >= 0.65 else "medium"
    if v in {"UNVERIFIABLE", "NO_EVIDENCE"}:
        return "medium" if confidence >= 0.45 else "low"
    return "low"


def _build_alerts(flags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    for flag in flags:
        verdict = str(flag.get("verdict") or "")
        conf = float(flag.get("confidence") or 0.0)
        claim = str(flag.get("claim") or "")

        category = "ok"
        message = "Supported claim detected."
        if verdict.upper() == "CONTRADICTED":
            category = "hallucination"
            message = "Potential hallucination: evidence contradicts this claim."
        elif verdict.upper() in {"UNVERIFIABLE", "NO_EVIDENCE"}:
            category = "red_flag"
            message = "Verification red flag: claim lacks reliable support."

        if _contains_bias_cue(claim):
            category = "bias"
            message = "Potential bias cue detected in phrasing."

        if category == "ok":
            continue

        alerts.append(
            {
                "category": category,
                "severity": _severity_for_claim(verdict, conf),
                "message": message,
                "claim": claim,
                "confidence": conf,
            }
        )

    return alerts[:4]


def result_payload(result: Any, top_claims: int = 6) -> dict[str, Any]:
    claim_conf = [float(c.confidence or 0.0) for c in result.claims]
    avg_conf = sum(claim_conf) / len(claim_conf) if claim_conf else 0.0

    flags = []
    for claim in result.claims[: max(1, top_claims)]:
        flags.append(
            {
                "claim": claim.text,
                "verdict": claim.verdict,
                "confidence": float(claim.confidence or 0.0),
                "reason": (
                    "Evidence contradicts this claim."
                    if claim.verdict == "CONTRADICTED"
                    else (
                        "Insufficient specific evidence to verify this claim."
                        if claim.verdict in {"UNVERIFIABLE", "NO_EVIDENCE"}
                        else "Claim is supported by retrieved evidence."
                    )
                ),
                "correction": (claim.correction or ""),
                "evidence": (claim.best_evidence.text[:220] if claim.best_evidence else ""),
                "source": (claim.best_evidence.source if claim.best_evidence else ""),
                "url": (claim.best_evidence.url if claim.best_evidence else ""),
            }
        )

    alerts = _build_alerts(flags)

    return {
        "factuality_score": float(result.factuality_score),
        "overall_confidence": float(avg_conf),
        "total_claims": int(result.total_claims),
        "supported": int(result.supported),
        "contradicted": int(result.contradicted),
        "unverifiable": int(result.unverifiable),
        "no_evidence": int(result.no_evidence),
        "processing_time": float(result.processing_time),
        "summary": summarize(result),
        "flags": flags,
        "alerts": alerts,
    }


class OverlayHandler(BaseHTTPRequestHandler):
    def _is_allowed_origin(self) -> bool:
        origin = (self.headers.get("Origin") or "").strip()
        if not origin:
            return True
        return any(origin.startswith(prefix) for prefix in _ALLOWED_ORIGIN_PREFIXES)

    def _is_authorized(self) -> bool:
        if not _API_TOKEN:
            return True
        token = (self.headers.get("X-VeriFact-Token") or "").strip()
        return bool(token) and token == _API_TOKEN

    def _json_response(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        if not self._is_allowed_origin():
            self._json_response(403, {"error": "origin_not_allowed"})
            return
        self._json_response(200, {"ok": True})

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._json_response(404, {"error": "not_found"})
            return
        self._json_response(200, {"ok": True, "service": "verifact-overlay"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/analyze":
            self._json_response(404, {"error": "not_found"})
            return

        if not self._is_allowed_origin():
            self._json_response(403, {"error": "origin_not_allowed"})
            return

        if not self._is_authorized():
            self._json_response(401, {"error": "unauthorized"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length)
            payload = json.loads(raw.decode("utf-8")) if raw else {}

            text = (payload.get("text") or "").strip()
            if not text:
                self._json_response(400, {"error": "text is required"})
                return

            top_claims = int(payload.get("top_claims", 6))
            pipeline = get_pipeline()
            result = pipeline.verify_text(text)
            self._json_response(200, result_payload(result, top_claims=top_claims))
        except Exception as exc:
            self._json_response(500, {"error": str(exc)})


def run(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), OverlayHandler)
    print(f"VeriFact overlay server listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
