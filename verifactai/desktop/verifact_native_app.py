#!/usr/bin/env python3
"""
VeriFact native desktop shell.

Starts local services when needed and opens a native webview window instead of a browser tab.
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OVERLAY_URL = "http://127.0.0.1:8765/health"
DASH_URL = "http://127.0.0.1:8501"

OVERLAY_LOG_OUT = Path("/tmp/verifact_overlay.out.log")
OVERLAY_LOG_ERR = Path("/tmp/verifact_overlay.err.log")
DASH_LOG_OUT = Path("/tmp/verifact_dashboard.out.log")
DASH_LOG_ERR = Path("/tmp/verifact_dashboard.err.log")


def endpoint_up(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


def start_if_needed(command: list[str], health_url: str, out_log: Path, err_log: Path) -> bool:
    if endpoint_up(health_url):
        return True

    with out_log.open("ab") as out_fh, err_log.open("ab") as err_fh:
        subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=out_fh,
            stderr=err_fh,
            start_new_session=True,
        )

    for _ in range(20):
        if endpoint_up(health_url):
            return True
        time.sleep(0.5)
    return False


def notify(msg: str) -> None:
    script = f'display notification "{msg}" with title "VeriFact Desktop"'
    subprocess.run(["osascript", "-e", script], check=False)


def open_native_window() -> None:
    try:
        import webview
    except Exception:
        subprocess.run(["open", DASH_URL], check=False)
        notify("Dashboard opened in browser (native webview unavailable).")
        return

    window = webview.create_window(
        "VeriFact AI",
        DASH_URL,
        width=1360,
        height=880,
        min_size=(980, 680),
        text_select=True,
    )
    webview.start(gui="cocoa", debug=False)


def main() -> int:
    overlay_ok = start_if_needed(
        [PY, "overlay_server.py"],
        OVERLAY_URL,
        OVERLAY_LOG_OUT,
        OVERLAY_LOG_ERR,
    )

    dash_ok = start_if_needed(
        [
            PY,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.port",
            "8501",
            "--server.headless",
            "true",
        ],
        DASH_URL,
        DASH_LOG_OUT,
        DASH_LOG_ERR,
    )

    if not overlay_ok or not dash_ok:
        notify("Startup issue. Check /tmp/verifact_overlay.err.log and /tmp/verifact_dashboard.err.log")
        return 1

    open_native_window()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
