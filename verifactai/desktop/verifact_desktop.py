#!/usr/bin/env python3
"""VeriFact Desktop Control App.

A local app-like control panel to start/stop core VeriFact services:
- Overlay API server (for browser extension)
- Streamlit dashboard

This provides a persistent laptop app workflow similar to Docker Desktop style control.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
VENV_PY = ROOT.parent / "venv" / "bin" / "python"


class ServiceManager:
    def __init__(self) -> None:
        self.overlay_proc: subprocess.Popen[str] | None = None
        self.streamlit_proc: subprocess.Popen[str] | None = None

    def start_overlay(self) -> None:
        if self.overlay_proc and self.overlay_proc.poll() is None:
            return
        cmd = [str(VENV_PY), "overlay_server.py"]
        self.overlay_proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def stop_overlay(self) -> None:
        if self.overlay_proc and self.overlay_proc.poll() is None:
            self.overlay_proc.terminate()
            self.overlay_proc.wait(timeout=5)

    def start_streamlit(self) -> None:
        if self.streamlit_proc and self.streamlit_proc.poll() is None:
            return
        cmd = [str(VENV_PY), "-m", "streamlit", "run", "app.py", "--server.port", "8501"]
        self.streamlit_proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def stop_streamlit(self) -> None:
        if self.streamlit_proc and self.streamlit_proc.poll() is None:
            self.streamlit_proc.terminate()
            self.streamlit_proc.wait(timeout=5)

    def overlay_running(self) -> bool:
        return bool(self.overlay_proc and self.overlay_proc.poll() is None)

    def streamlit_running(self) -> bool:
        return bool(self.streamlit_proc and self.streamlit_proc.poll() is None)

    def shutdown(self) -> None:
        self.stop_overlay()
        self.stop_streamlit()


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("VeriFact Desktop")
        self.geometry("500x320")
        self.resizable(False, False)
        self.mgr = ServiceManager()

        self.configure(bg="#0f172a")
        tk.Label(self, text="VeriFact Desktop", fg="#f8fafc", bg="#0f172a", font=("Helvetica", 18, "bold")).pack(pady=14)

        self.overlay_status = tk.StringVar(value="Background Analyzer: stopped")
        self.streamlit_status = tk.StringVar(value="Dashboard (optional): stopped")

        tk.Label(self, textvariable=self.overlay_status, fg="#e2e8f0", bg="#0f172a").pack()
        tk.Label(self, textvariable=self.streamlit_status, fg="#e2e8f0", bg="#0f172a").pack(pady=(0, 14))

        row1 = tk.Frame(self, bg="#0f172a")
        row1.pack(pady=6)
        tk.Button(row1, text="Start Analyzer", width=18, command=self.start_overlay).pack(side=tk.LEFT, padx=6)
        tk.Button(row1, text="Stop Analyzer", width=18, command=self.stop_overlay).pack(side=tk.LEFT, padx=6)

        row2 = tk.Frame(self, bg="#0f172a")
        row2.pack(pady=6)
        tk.Button(row2, text="Start Dashboard (Optional)", width=24, command=self.start_streamlit).pack(side=tk.LEFT, padx=6)
        tk.Button(row2, text="Stop Dashboard", width=12, command=self.stop_streamlit).pack(side=tk.LEFT, padx=6)

        row3 = tk.Frame(self, bg="#0f172a")
        row3.pack(pady=10)
        tk.Button(row3, text="Open Dashboard", width=18, command=lambda: webbrowser.open("http://127.0.0.1:8501")).pack(side=tk.LEFT, padx=6)
        tk.Button(row3, text="Browser Setup", width=18, command=self.open_extension_guide).pack(side=tk.LEFT, padx=6)

        tk.Label(
            self,
            text="Start this app, keep Analyzer running, then open ChatGPT/Claude/Gemini in browser.",
            fg="#94a3b8",
            bg="#0f172a",
        ).pack(pady=8)

        # App-first behavior: automatically start background analyzer.
        threading.Timer(0.8, self.start_overlay).start()

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(1200, self.refresh_status)

    def open_extension_guide(self) -> None:
        guide = ROOT / "integrations" / "web_beacon_extension" / "README.md"
        messagebox.showinfo(
            "Browser Setup",
            "1) Load unpacked extension from:\n"
            f"{guide.parent}\n\n"
            "2) Keep 'Background Analyzer' status as running\n"
            "3) Open ChatGPT / Claude / Gemini in browser\n"
            "4) VeriFact beacon appears at mid-right",
        )

    def start_overlay(self) -> None:
        try:
            self.mgr.start_overlay()
            self.refresh_status()
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to start overlay server:\n{exc}")

    def stop_overlay(self) -> None:
        try:
            self.mgr.stop_overlay()
            self.refresh_status()
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to stop overlay server:\n{exc}")

    def start_streamlit(self) -> None:
        try:
            self.mgr.start_streamlit()
            self.refresh_status()
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to start dashboard:\n{exc}")

    def stop_streamlit(self) -> None:
        try:
            self.mgr.stop_streamlit()
            self.refresh_status()
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to stop dashboard:\n{exc}")

    def refresh_status(self) -> None:
        self.overlay_status.set("Background Analyzer: running" if self.mgr.overlay_running() else "Background Analyzer: stopped")
        self.streamlit_status.set("Dashboard (optional): running" if self.mgr.streamlit_running() else "Dashboard (optional): stopped")
        self.after(1500, self.refresh_status)

    def on_close(self) -> None:
        if messagebox.askyesno("Quit VeriFact", "Stop services and quit VeriFact Desktop?"):
            self.mgr.shutdown()
            self.destroy()


def main() -> None:
    if not VENV_PY.exists():
        print(f"Python interpreter not found: {VENV_PY}", file=sys.stderr)
        sys.exit(1)
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
