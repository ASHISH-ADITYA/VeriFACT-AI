"""
Runtime safety guards for native-threaded ML libraries.

These defaults reduce OpenMP/libomp instability on macOS ARM when FAISS,
tokenizers, and other native libs are loaded in the same process.
"""

from __future__ import annotations

import os


def apply_runtime_safety_defaults() -> None:
    """Set conservative thread defaults unless user already configured them."""
    defaults = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_WAIT_POLICY": "PASSIVE",
        "KMP_BLOCKTIME": "0",
        "KMP_INIT_AT_FORK": "FALSE",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_ENABLE_HF_XET": "0",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def configure_faiss_threads() -> None:
    """Best-effort FAISS OpenMP thread cap."""
    try:
        import faiss  # type: ignore

        threads = int(os.environ.get("VERIFACTAI_FAISS_THREADS", "1"))
        threads = max(1, threads)
        faiss.omp_set_num_threads(threads)
    except Exception:
        # Keep startup resilient even if FAISS is unavailable during some flows.
        pass
