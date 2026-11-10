#!/usr/bin/env python3
"""decode(pack(q, s)) must reproduce the rounding's hat exactly, for each format, on real weights."""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import s2_pass as s
from requant_common import FP8_SRC, Checkpoint
import t3lib as t, t3pack as p
torch.set_grad_enabled(False)
D = "cuda"
src = Checkpoint(FP8_SRC)
base = f"{s.PREFIX}layers.3.mlp.experts."
exp = [0, 100, 300, 500]
gu = torch.stack([torch.cat([src.weight(f"{base}{e}.gate_proj.weight", D), src.weight(f"{base}{e}.up_proj.weight", D)]) for e in exp]).float()
dn = torch.stack([src.weight(f"{base}{e}.down_proj.weight", D) for e in exp]).float()
CUBIC = "c2"
for bits, alpha in ((3, .135 / 1000), (2, .22 / 250), (4, .07 / 4000)):
    for w, m in ((gu, "w13"), (dn, "w2")):
        K = w.shape[2]
        H = torch.eye(K, device=D).expand(len(exp), K, K).clone() + 0.1
        hat, sc, q = t.babai_round(w, H, f"T{bits}", alpha, CUBIC, beta=0.8)
        words = p.pack(q, sc, m, bits)
        back = p.decode(words, m, bits, CUBIC)
        bad = int((back != hat).sum())
        print(f"T{bits} {m}: mismatched {bad} of {hat.numel()}; rel diff {float((back - hat).norm() / hat.norm()):.2e}", flush=True)
        assert bad == 0
print("PASS")
