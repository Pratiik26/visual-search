"""
AURA Diamonds Visual Search Server Launcher
Production Uvicorn server launcher with auto-port freeing on Windows.
"""

import os
import sys
import subprocess
import uvicorn
from backend.config import DEFAULT_HOST, DEFAULT_PORT


def free_port(port: int):
    """Free port if occupied on Windows"""
    if sys.platform == "win32":
        try:
            cmd = f"Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }}"
            subprocess.run(["powershell", "-Command", cmd], capture_output=True, timeout=5)
        except Exception:
            pass


def main():
    port = int(os.environ.get("PORT", DEFAULT_PORT))
    host = os.environ.get("HOST", DEFAULT_HOST)

    free_port(port)

    print("=" * 65)
    print(f"[*] AURA DIAMONDS Visual Search Engine")
    print(f"[*] Web Interface: http://{host}:{port}/")
    print(f"[*] Swagger Docs:  http://{host}:{port}/docs")
    print(f"[*] ReDoc Docs:    http://{host}:{port}/redoc")
    print("=" * 65)

    uvicorn.run("backend.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
