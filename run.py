"""
Precious Edu LLM — Root Entry Point

Delegates execution directly to backend/run.py so commands can be run from the root directory.

Usage:
    python run.py [server|train-all|train-tokenizer|train-base|train-domain|finetune|evaluate|import-qa|crawl-website]
"""

import os
from pathlib import Path
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
BACKEND_RUN = BACKEND_DIR / "run.py"

if __name__ == "__main__":
    if not BACKEND_RUN.exists():
        print(f"Error: {BACKEND_RUN} not found.", file=sys.stderr)
        sys.exit(1)

    cmd = [sys.executable, str(BACKEND_RUN)] + sys.argv[1:]
    try:
        proc = subprocess.run(cmd, cwd=str(BACKEND_DIR))
        sys.exit(proc.returncode)
    except KeyboardInterrupt:
        sys.exit(130)
