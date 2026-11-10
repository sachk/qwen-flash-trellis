#!/usr/bin/env python3
"""Upload a staged full model using the authenticated hf CLI; never touch serving."""
import argparse
import json
import subprocess
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("repo")
p.add_argument("folder", type=Path)
a = p.parse_args()
inventory = json.loads((a.folder / "inventory.json").read_text())
print(f"UPLOAD_STARTED repo={a.repo} weights={inventory['weight_files']} bytes={inventory['checkpoint_bytes']}", flush=True)
subprocess.run(["hf", "upload-large-folder", a.repo, str(a.folder), "--repo-type", "model", "--num-workers", "2", "--no-bars"], check=True)
print("UPLOAD_FINISHED", flush=True)
