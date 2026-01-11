#!/usr/bin/env python3
"""
Wrapper to run the shared teacher SFT generator for SurgVidLM jobs.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def run() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "generate_teacher_sft.py"
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    run()
