"""
Unified LLM client with ordered fallback chain.

Query mode (strict=False):  Ollama → Anthropic → OpenAI → returns None
Eval mode  (strict=True):   Primary provider only; raises on failure.

Supports: Ollama (free local), Anthropic Claude, OpenAI GPT.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Optional

from config import LLMConfig
from utils.helpers import logger, retry_with_backoff


class LLMClient:
    """Provider-agnostic LLM wrapper with fallback chain."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._primary_provider = config.provider
        self._clients: dict[str, object] = {}
        self._init_providers()

    # ------------------------------------------------------------------
    # Provider initialisation
    # ------------------------------------------------------------------
    def _init_providers(self) -> None:
        """Initialise all available providers (primary first)."""
        # Always try primary
        self._try_init(self._primary_provider)

        # Prepare optional fallbacks (only if keys are configured)
        if self.config.anthropic_api_key and "anthropic" not in self._clients:
            self._try_init("anthropic")
        if self.config.openai_api_key and "openai" not in self._clients:
            self._try_init("openai")

        if not self._clients:
            logger.warning("No LLM providers initialised. Decomposition will use spaCy fallback only.")

    def _try_init(self, provider: str) -> None:
        try:
            if provider == "ollama":
                self._clients["ollama"] = {"base_url": self.config.ollama_base_url.rstrip("/")}
                logger.info(f"Ollama configured (model={self.config.model}, url={self.config.ollama_base_url})")
            elif provider == "anthropic":
                import anthropic
                self._clients["anthropic"] = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
                logger.info(f"Anthropic initialised (model={self.config.model})")
            elif provider == "openai":
                import openai
                self._clients["openai"] = openai.OpenAI(api_key=self.config.openai_api_key)
                logger.info(f"OpenAI initialised (model={self.config.model})")
            else:
                logger.warning(f"Unknown provider: {provider}")
        except Exception as exc:
            logger.warning(f"Failed to initialise {provider}: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        user: str,
        system: str = "You are a helpful, precise assistant.",
        temperature: float | None = None,
        max_tokens: int | None = None,
        strict: bool = False,
    ) -> Optional[str]:
        """
        Generate a completion.

        Args:
            strict: If True (eval mode), use primary provider only.
                    If False (query mode), try fallback chain.

        Returns:
            Response text, or None if all providers failed and strict=False.
        """
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        if strict:
            return self._call_strict(user=user, system=system, temperature=temp, max_tokens=tokens)
        return self._call_with_fallback(user=user, system=system, temperature=temp, max_tokens=tokens)

    # ------------------------------------------------------------------
    # Strict mode (eval): primary only, raise on failure
    # ------------------------------------------------------------------
    def _call_strict(self, *, user: str, system: str, temperature: float, max_tokens: int) -> str:
        provider = self._primary_provider
        if provider not in self._clients:
            raise RuntimeError(f"Primary provider '{provider}' not available (strict mode)")
        return retry_with_backoff(
            self._dispatch,
            provider=provider,
            user=user,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            attempts=self.config.retry_attempts,
            base_delay=self.config.retry_base_delay,
        )

    # ------------------------------------------------------------------
    # Fallback mode (query): try each provider in order
    # ------------------------------------------------------------------
    def _call_with_fallback(
        self, *, user: str, system: str, temperature: float, max_tokens: int
    ) -> Optional[str]:
        # Build ordered fallback chain: primary first, then others
        chain = [self._primary_provider]
        for p in ["ollama", "anthropic", "openai"]:
            if p != self._primary_provider and p in self._clients:
                chain.append(p)

        last_exc: Exception | None = None
        for provider in chain:
            if provider not in self._clients:
                continue
            try:
                result = retry_with_backoff(
                    self._dispatch,
                    provider=provider,
                    user=user,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    attempts=2,  # fewer retries per provider in fallback mode
                    base_delay=0.5,
                )
                if provider != self._primary_provider:
                    logger.warning(f"Fallback: using {provider} (primary {self._primary_provider} failed)")
                return result
            except Exception as exc:
                last_exc = exc
                logger.warning(f"Provider {provider} failed: {exc}")
                continue

        logger.error(f"All LLM providers failed. Last error: {last_exc}")
        return None  # triggers spaCy-only decomposition in ClaimDecomposer

    # ------------------------------------------------------------------
    # Provider dispatch
    # ------------------------------------------------------------------

    # Fallback model names when primary model is Ollama-specific.
    # Only used when fallback chain reaches a paid provider.
    _FALLBACK_MODELS = {
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "gpt-4o-mini",
    }

    def _resolve_model(self, provider: str) -> str:
        """Return a valid model name for the given provider.

        If the configured model is Ollama-specific (e.g. 'llama3.1:8b')
        and we're falling back to Anthropic/OpenAI, use a safe default.
        """
        model = self.config.model
        if provider == "ollama":
            return model
        # Ollama model names contain ':' — not valid for cloud providers
        if ":" in model or model.startswith("llama") or model.startswith("qwen"):
            fallback = self._FALLBACK_MODELS.get(provider, model)
            logger.info(f"Model '{model}' is Ollama-specific; using '{fallback}' for {provider}")
            return fallback
        return model

    def _dispatch(
        self, *, provider: str, user: str, system: str, temperature: float, max_tokens: int
    ) -> str:
        model = self._resolve_model(provider)
        try:
            if provider == "ollama":
                return self._call_ollama(user=user, system=system, temperature=temperature, max_tokens=max_tokens)
            elif provider == "anthropic":
                client = self._clients["anthropic"]
                resp = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return resp.content[0].text
            elif provider == "openai":
                client = self._clients["openai"]
                resp = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                return resp.choices[0].message.content
            else:
                raise ValueError(f"Unknown provider: {provider}")
        except Exception as exc:
            request_id = self._extract_request_id(exc)
            details = f"provider={provider}, model={self.config.model}"
            if request_id:
                details += f", request_id={request_id}"
            logger.error(f"LLM call failed ({details}): {exc}")
            raise

    # ------------------------------------------------------------------
    # Ollama HTTP call
    # ------------------------------------------------------------------
    def _call_ollama(
        self, *, user: str, system: str, temperature: float, max_tokens: int
    ) -> str:
        base_url = self._clients["ollama"]["base_url"]
        payload = {
            "model": self.config.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        req = urllib.request.Request(
            f"{base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {base_url}. "
                f"Ensure `ollama serve` is running and model `{self.config.model}` is pulled."
            ) from exc

        parsed = json.loads(body)
        content = parsed.get("message", {}).get("content", "")
        if not content:
            raise RuntimeError("Ollama returned an empty response")
        return content

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_request_id(exc: Exception) -> str | None:
        text = str(exc)
        match = re.search(r"req_[A-Za-z0-9]+", text)
        return match.group(0) if match else None
