#!/usr/bin/env python3
"""Record installed non-editable dependencies after a successful t3 setup."""
import argparse
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
text = subprocess.check_output(["uv", "pip", "freeze", "--python", sys.executable, "--exclude-editable"], text=True)
lines = [line for line in text.splitlines() if line and not line.startswith("#")]
if any("==" not in line or " @ " in line for line in lines):
    raise SystemExit("dependency snapshot contains an unportable direct reference")
a.out.write_text("# Linux x86_64, Python 3.12; generated from the verified CPU-builder installation.\n" + "\n".join(sorted(lines, key=str.lower)) + "\n")
print(f"Locked {len(lines)} non-editable packages; vLLM itself is pinned by the flake.")
