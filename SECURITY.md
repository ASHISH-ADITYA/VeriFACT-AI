# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 1.0.x | Yes |

## Local-First Architecture

VeriFACT AI is designed to run entirely on your local machine. In default configuration:

- No data is transmitted to external servers
- All LLM inference runs via Ollama on localhost
- The overlay server binds to 127.0.0.1 only
- The browser extension communicates only with localhost

## Reporting a Vulnerability

If you discover a security vulnerability, please report it privately:

1. Do NOT open a public GitHub Issue
2. Email the maintainer directly
3. Include a clear description and steps to reproduce

We will acknowledge receipt within 48 hours and provide a fix timeline within 7 days.

## Known Security Considerations

- The `.env` file may contain API keys if cloud providers are configured. It is excluded from git via `.gitignore`.
- The Chrome extension has minimal permissions (`storage` only). It reads page content only from matched domains (chatgpt.com, claude.ai).
- The overlay server has no authentication (acceptable for localhost-only binding). Do not expose it to a network.
