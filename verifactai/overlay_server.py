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
from urllib.parse import urlsplit

from config import Config
from core.hallucination_discriminator import HallucinationDiscriminator
from core.pipeline import VeriFactPipeline
from core.risk_classifier import RiskClassifier
from utils.runtime_safety import apply_runtime_safety_defaults

apply_runtime_safety_defaults()


_PIPELINE: VeriFactPipeline | None = None
_PIPELINE_LOCK = threading.Lock()
_RISK_CLASSIFIER: RiskClassifier | None = None
_DISCRIMINATOR: HallucinationDiscriminator | None = None
_API_TOKEN = os.environ.get("VERIFACT_API_TOKEN", "").strip()
_ENV = os.environ.get("VERIFACT_ENV", "development").strip().lower()


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


_REQUIRE_AUTH = _env_bool("VERIFACT_REQUIRE_AUTH", _ENV in {"production", "prod"})
_ENABLE_RISK_CLASSIFIER = os.environ.get("VERIFACT_ENABLE_RISK_CLASSIFIER", "1") == "1"
_ENABLE_DISCRIMINATOR = os.environ.get("VERIFACT_ENABLE_DISCRIMINATOR", "0") == "1"
_DISCRIMINATOR_PATH = os.environ.get(
    "VERIFACT_DISCRIMINATOR_PATH", "assets/models/hallucination_discriminator.joblib"
)

# CORS origins — configurable via env for production (comma-separated)
_EXTRA_ORIGINS = os.environ.get("VERIFACT_CORS_ORIGINS", "").strip()
_DEFAULT_ALLOWED_ORIGINS = {
    "https://chatgpt.com",
    "https://chat.openai.com",
    "https://claude.ai",
    "https://gemini.google.com",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8501",
    "http://127.0.0.1:8501",
}
_ALLOWED_ORIGINS = _DEFAULT_ALLOWED_ORIGINS | {
    o.strip().rstrip("/") for o in _EXTRA_ORIGINS.split(",") if o.strip()
}

# Rate limiter — simple in-memory per-IP (resets on restart)
_RATE_LIMIT = int(os.environ.get("VERIFACT_RATE_LIMIT", "20"))  # requests per minute
_rate_buckets: dict[str, list[float]] = {}
_rate_lock = threading.Lock()


def _check_rate_limit(ip: str) -> bool:
    """Returns True if allowed, False if rate-limited."""
    import time

    now = time.time()
    window = 60.0
    with _rate_lock:
        bucket = _rate_buckets.setdefault(ip, [])
        # Prune old entries
        _rate_buckets[ip] = [t for t in bucket if now - t < window]
        if len(_rate_buckets[ip]) >= _RATE_LIMIT:
            return False
        _rate_buckets[ip].append(now)
        return True


def _normalized_origin(origin: str) -> str | None:
    if not origin:
        return None
    try:
        parsed = urlsplit(origin)
    except Exception:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def get_pipeline() -> VeriFactPipeline:
    global _PIPELINE
    with _PIPELINE_LOCK:
        if _PIPELINE is None:
            _PIPELINE = VeriFactPipeline(Config())
        return _PIPELINE


def get_risk_classifier() -> RiskClassifier | None:
    global _RISK_CLASSIFIER
    if not _ENABLE_RISK_CLASSIFIER:
        return None
    if _RISK_CLASSIFIER is None:
        _RISK_CLASSIFIER = RiskClassifier()
    return _RISK_CLASSIFIER


def get_discriminator() -> HallucinationDiscriminator | None:
    global _DISCRIMINATOR
    if not _ENABLE_DISCRIMINATOR:
        return None
    if _DISCRIMINATOR is None:
        model = HallucinationDiscriminator(_DISCRIMINATOR_PATH)
        if not model.load():
            return None
        _DISCRIMINATOR = model
    return _DISCRIMINATOR


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
    classifier = get_risk_classifier()
    discriminator = get_discriminator()
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

        classifier_hit = False
        classifier_score = 0.0
        if classifier is not None and classifier.available:
            pred = classifier.predict(claim)
            if pred is not None:
                classifier_score = float(pred.score)
                classifier_hit = pred.score >= 0.78 and any(
                    x in pred.label for x in ("toxic", "insult", "hate", "offens")
                )

        discriminator_score = 0.0
        if discriminator is not None:
            pred = discriminator.predict(claim)
            discriminator_score = float(pred.score)
            if pred.label == "hallucinated" and verdict.upper() in {"UNVERIFIABLE", "NO_EVIDENCE"}:
                category = "hallucination"
                message = "Discriminator flagged high hallucination likelihood for this claim."

        if _contains_bias_cue(claim) or classifier_hit:
            category = "bias"
            message = (
                "Potential bias cue detected in phrasing."
                if not classifier_hit
                else f"Classifier flagged potential biased/toxic phrasing (score={classifier_score:.2f})."
            )

        if category == "ok":
            continue

        alerts.append(
            {
                "category": category,
                "severity": _severity_for_claim(verdict, conf),
                "message": message,
                "claim": claim,
                "confidence": conf,
                "classifier_score": round(classifier_score, 4),
                "discriminator_score": round(discriminator_score, 4),
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
                "uncertainty": float(claim.uncertainty or 0.0),
                "stability": float(claim.stability or 0.0),
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
        normalized = _normalized_origin(origin)
        return bool(normalized and normalized in _ALLOWED_ORIGINS)

    def _is_authorized(self) -> bool:
        if not _REQUIRE_AUTH:
            return True

        if not _API_TOKEN:
            return False

        token = (self.headers.get("X-VeriFact-Token") or "").strip()
        return bool(token) and token == _API_TOKEN

    def _json_response(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        request_origin = (self.headers.get("Origin") or "").strip()
        response_origin = "*"
        normalized = _normalized_origin(request_origin)
        if normalized and normalized in _ALLOWED_ORIGINS:
            response_origin = normalized

        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # CORS
        self.send_header("Access-Control-Allow-Origin", response_origin)
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-VeriFact-Token")
        # Security headers
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
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
            if _REQUIRE_AUTH and not _API_TOKEN:
                self._json_response(
                    503,
                    {
                        "error": "server_misconfigured",
                        "message": "VERIFACT_REQUIRE_AUTH is enabled but VERIFACT_API_TOKEN is empty",
                    },
                )
                return
            self._json_response(401, {"error": "unauthorized"})
            return

        client_ip = self.client_address[0]
        if not _check_rate_limit(client_ip):
            self._json_response(429, {"error": "rate_limited", "retry_after_seconds": 60})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))

            # Request size limit: 100KB max
            max_body = 100_000
            if content_length > max_body:
                self._json_response(413, {"error": f"Request too large (max {max_body} bytes)"})
                return

            raw = self.rfile.read(content_length)
            payload = json.loads(raw.decode("utf-8")) if raw else {}

            text = (payload.get("text") or "").strip()
            if not text:
                self._json_response(400, {"error": "text is required"})
                return

            # Text length limit: 50,000 chars
            max_text = 50_000
            if len(text) > max_text:
                self._json_response(400, {"error": f"Text too long (max {max_text} chars)"})
                return

            top_claims = min(int(payload.get("top_claims", 6)), 20)  # Cap at 20
            pipeline = get_pipeline()
            result = pipeline.verify_text(text)
            self._json_response(200, result_payload(result, top_claims=top_claims))
        except json.JSONDecodeError:
            self._json_response(400, {"error": "Invalid JSON body"})
        except Exception as exc:
            self._json_response(500, {"error": str(exc)})


def run() -> None:
    import signal

    host = os.environ.get("VERIFACT_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", os.environ.get("VERIFACT_PORT", "8765")))

    server = ThreadingHTTPServer((host, port), OverlayHandler)

    print("=" * 50)
    print("  VeriFACT AI — API Server")
    print("=" * 50)
    print(f"  Listening on http://{host}:{port}")
    print(f"  Rate limit: {_RATE_LIMIT} req/min per IP")
    print(f"  Environment: {_ENV}")
    print(f"  Require auth: {'yes' if _REQUIRE_AUTH else 'no'}")
    if _REQUIRE_AUTH and not _API_TOKEN:
        print("  WARNING: auth required but VERIFACT_API_TOKEN is not set")
    print("  Press Ctrl+C to stop")
    print("=" * 50)

    def _shutdown(signum, frame):
        print("\nShutting down...")
        server.shutdown()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
