---
name: safe-write
description: Safety protocol for writing or modifying pipeline code. Must be followed before any edit to hallucination detection, RAG, chunking, or NLI modules. Auto-load when editing pipeline files.
disable-model-invocation: false
allowed-tools: Read, Bash, Write, Edit
---

Before any code change, mandatory sequence:
1. Run existing tests: `python3 -m pytest tests/ -q` — record baseline pass/fail.
2. Read the target file fully. Identify all callers.
3. State explicitly: "The following currently works and will NOT be broken: [list]"
4. Make the minimal, surgical change. Never refactor working code opportunistically.
5. Run tests again after change. If regression: revert immediately and report.
6. Only mark complete when test delta is: new_tests_added >= 1, regressions = 0.
