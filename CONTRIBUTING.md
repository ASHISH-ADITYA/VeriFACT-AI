# Contributing to VeriFACT AI

## Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Follow the setup instructions in README.md
4. Run tests: `cd verifactai && python -m pytest tests/ -v`
5. Run smoke test: `python smoke_test.py`

## Code Style

- Python: follow existing patterns in the codebase
- Use type hints for function signatures
- Use `loguru` for logging (via `utils/helpers.py`)

## Pull Requests

- Keep PRs focused on a single change
- Include test coverage for new functionality
- Ensure smoke test passes before submitting
- Describe what changed and why in the PR description

## Reporting Issues

- Use GitHub Issues
- Include: Python version, OS, Ollama version, error traceback
- Include output of `python smoke_test.py` if relevant
