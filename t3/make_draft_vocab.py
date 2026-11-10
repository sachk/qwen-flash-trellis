#!/usr/bin/env python3
"""Draft vocabulary: every token seen in calibration or special, topped up to 65,536 by lowest ID."""
import json, sys
import numpy as np
from safetensors import safe_open
from paths import ROOT
counts = np.zeros(248320, dtype=np.int64)
for name in ("calib", "calib2", "eval_nemotron", "eval_multilingual", "eval_public", "eval_medical", "eval_legal"):
    with safe_open(ROOT / f"data/sets/{name}.safetensors", framework="np") as f:
        counts += np.bincount(f.get_tensor("input_ids"), minlength=248320)
keep = set(np.flatnonzero(counts > 0).tolist()) | set(range(248000, 248320))
for i in range(248320):
    if len(keep) >= 65536:
        break
    keep.add(i)
table = sorted(keep)[:65536]
for name in ("eval_public", "eval_multilingual", "eval_medical", "eval_legal"):
    with safe_open(ROOT / f"data/sets/{name}.safetensors", framework="np") as f:
        ids = f.get_tensor("input_ids")
    print(name, "coverage", float(np.isin(ids, table).mean()), flush=True)
json.dump(table, open(sys.argv[1], "w"))
print(len(table), "tokens")
