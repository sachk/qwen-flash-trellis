#!/usr/bin/env python3
"""Verify every expected checkpoint file and byte size through the public Hub API."""
import argparse
import json
from pathlib import Path
import urllib.parse
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("repo")
p.add_argument("inventory", type=Path)
a = p.parse_args()
expected = {r["file"]: r["bytes"] for r in json.loads(a.inventory.read_text())["files"]}
url = f"https://huggingface.co/api/models/{a.repo}/tree/main?recursive=true&limit=1000"
actual = {}
while url:
    with urllib.request.urlopen(url, timeout=60) as response:
        entries = json.load(response)
        links = response.headers.get("Link", "")
    actual.update({r["path"]: r["size"] for r in entries if r["type"] == "file"})
    url = None
    for link in links.split(","):
        if 'rel="next"' in link:
            url = link.split("<", 1)[1].split(">", 1)[0]
missing = sorted(set(expected) - set(actual))
wrong = {name: {"expected": size, "actual": actual[name]} for name, size in expected.items() if name in actual and actual[name] != size}
print(json.dumps({"repo": a.repo, "expected_weight_files": sum(n.endswith(".safetensors") for n in expected),
                  "checked_files": len(expected), "checkpoint_bytes": sum(expected.values()), "missing": missing, "size_mismatches": wrong}, indent=2))
if missing or wrong:
    raise SystemExit(1)
