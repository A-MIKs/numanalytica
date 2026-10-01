#!/usr/bin/env python3
"""Run the appendix example scripts to verify that the thesis figures reproduce cleanly."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXAMPLES_DIR = ROOT / "examples"


def main() -> int:
    scripts = sorted(EXAMPLES_DIR.glob("*.py"))
    if not scripts:
        print("No example scripts found in examples/.")
        return 1

    failed = []
    for script in scripts:
        print(f"Running {script.name}...")
        try:
            subprocess.run([sys.executable, str(script)], cwd=str(ROOT), check=True)
        except subprocess.CalledProcessError as exc:
            print(f"FAILED: {script.name} (exit code {exc.returncode})")
            failed.append(script.name)
        print()

    if failed:
        print("Reproducibility check failed for:")
        for item in failed:
            print(f"  - {item}")
        return 1

    print(f"All {len(scripts)} example scripts completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
