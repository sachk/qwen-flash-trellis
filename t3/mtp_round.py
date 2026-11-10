#!/usr/bin/env python3
"""MTP experts in the trellis format, shaped levels. No Hessian: draft precision only changes acceptance, never outputs."""
import json, struct, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from requant_common import FP8_SRC, Checkpoint
import t3lib as t, t3pack as p
torch.set_grad_enabled(False)
D = "cuda"
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
bits13, bits2 = int(sys.argv[2]), int(sys.argv[3])
SCALE = {2: (0.22 / 250, 0.55), 3: (0.135 / 1000, 0.8), 4: (0.07 / 4000, 0.9)}
src = Checkpoint(FP8_SRC)
base = "mtp.layers.0.mlp.experts."
CHUNK = 16
s13 = 80 * 20 * p.PKT[bits13]; s2 = 160 * 5 * p.PKT[bits2]
files = {"t3_w13": out / "w13.bin", "t3_w2": out / "w2.bin"}
for name, K, bits, parts, path in (("gate_up", 2560, bits13, ("gate_proj", "up_proj"), files["t3_w13"]), ("down", 640, bits2, ("down_proj",), files["t3_w2"])):
    H = torch.eye(K, device=D).expand(CHUNK, K, K).clone()
    with open(path, "wb") as f:
        for b in range(0, 512, CHUNK):
            w = torch.stack([torch.cat([src.weight(f"{base}{e}.{pn}.weight", D) for pn in parts]) for e in range(b, b + CHUNK)]).float()
            alpha, beta = SCALE[bits]
            hat, sc, q = t.babai_round(w, H, f"T{bits}", alpha, "c2", beta=beta)
            f.write(p.pack(q, sc, "w13" if K == 2560 else "w2", bits).cpu().numpy().astype(np.uint32).tobytes())
            print(name, b, float(((hat - w) ** 2).sum() / (w ** 2).sum()), flush=True)
prefix = base
head, offset, blobs = {}, 0, [("t3_off13", "I64", (np.arange(512, dtype=np.int64) * s13).tobytes()), ("t3_off2", "I64", (np.arange(512, dtype=np.int64) * s2).tobytes()), ("t3_fmt13", "U8", bytes([bits13]) * 512), ("t3_fmt2", "U8", bytes([bits2]) * 512)]
for name, size in (("t3_w13", s13 * 512), ("t3_w2", s2 * 512)):
    head[prefix + name] = {"dtype": "U8", "shape": [size], "data_offsets": [offset, offset + size]}; offset += size
for name, dtype, data in blobs:
    head[prefix + name] = {"dtype": dtype, "shape": [512], "data_offsets": [offset, offset + len(data)]}; offset += len(data)
blob = json.dumps(head).encode(); blob = blob.ljust(len(blob) + (8 - (8 + len(blob)) % 8) % 8)
with open(out / "mtp-t3.safetensors", "wb") as g:
    g.write(struct.pack("<Q", len(blob))); g.write(blob)
    for key in ("t3_w13", "t3_w2"):
        with open(files[key], "rb") as r:
            while True:
                c = r.read(1 << 24)
                if not c:
                    break
                g.write(c)
    for _, _, data in blobs:
        g.write(data)
for path in files.values():
    path.unlink()
(out / "sizes.json").write_text(json.dumps({"w13": s13 * 512, "w2": s2 * 512}))
print("done", out)
