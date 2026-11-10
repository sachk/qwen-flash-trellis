#!/usr/bin/env python3
"""Per-layer KL weight for the allocator, from block runs: measured KL change over the error the allocator predicted for that change."""
import json, sys
import numpy as np
from paths import ROOT
base_name, low_name, lib, out = sys.argv[1:5]
SIM = str(ROOT / "sim") + "/"
def kl(n):
    d = json.load(open(f"{SIM}{n}/result.json"))
    return np.array([d[k]["direct_kl"] for k in ("eval_public", "eval_multilingual", "eval_medical")])
base = kl("p-44")
alloc_low = json.load(open(f"{low_name}/alloc.json"))["layers"]
alloc_base = json.load(open(f"{base_name}/alloc.json"))["layers"]
weights = {}
blocks = list(range(0, 48, 4))
raw = []
for s in blocks:
    measured = float(((kl(f"p-b{s}") - base) * [0.5, 0.25, 0.25]).sum())
    saved = sum(alloc_base[str(l)]["bytes"] - alloc_low[str(l)]["bytes"] for l in range(s, s + 4)) / 2**30
    raw.append(measured / max(saved, 1e-3))
raw = np.array(raw)
# Runs of 4 layers carry noise of about 0.001 KL, so smooth along depth and keep a floor: nothing is free outside the tested domains.
smooth = np.convolve(np.pad(raw, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
floor = 0.15 * smooth.max()
smooth = np.maximum(smooth, floor)
smooth /= smooth.mean()
for i, s in enumerate(blocks):
    for l in range(s, s + 4):
        weights[str(l)] = float(smooth[i])
json.dump(weights, open(out, "w"))
print({s: round(float(v), 3) for s, v in zip(blocks, smooth)})
