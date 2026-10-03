"""
CareerPulse Dashboard Launcher
==============================
Launches the interactive Streamlit Career & Salary Intelligence dashboard.
"""

import sys
import os
import subprocess

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, "dashboard", "app.py")
    port = 8503

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--theme.base",
        "dark",
        "--theme.primaryColor",
        "#38bdf8",
        "--theme.backgroundColor",
        "#091224",
        "--theme.secondaryBackgroundColor",
        "#0f1f38",
        "--theme.textColor",
        "#f8fafc"
    ]

    print(f"[CareerPulse] Launching Tech Career & Salary Dashboard on port {port}...")
    print(f"[CareerPulse] URL: http://localhost:{port}")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[CareerPulse] Dashboard stopped.")
