#!/usr/bin/env python3
"""Chooses a format per expert from the pass's library, under a byte budget, and writes one file per layer.

Each expert starts at T3/T3. Moving gate_up to T2 saves bytes; moving down to T4 costs bytes. Moves are taken in
order of least added error per byte (or most removed), until total bytes meet the budget. Error for a move is
  layer weight * importance * (relative objective after - before)
importance = (1 - floor) * energy / layer energy + floor / 512. The floor keeps rarely routed experts from being
treated as free, since calibration undercounts topics it didn't cover.
"""
import argparse, json, os, struct
from pathlib import Path
import numpy as np

EXPERT13 = {3: 80 * 20 * 800, 2: 80 * 20 * 544}
EXPERT2 = {3: 160 * 5 * 800, 4: 160 * 5 * 1056, 2: 160 * 5 * 544}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--lib", required=True, help="pass output directory")
    a.add_argument("--out", required=True)
    a.add_argument("--gib", type=float, required=True, help="total expert bytes, GiB")
    a.add_argument("--floor", type=float, default=0.25)
    a.add_argument("--down-weight", type=float, default=1.0)
    a.add_argument("--layers", type=int, default=48)
    a.add_argument("--layer-weights", default=None, help="JSON file {layer: weight}")
    a.add_argument("--plan", action="store_true", help="write only plan.json; t3pack.load_layer reads the library through it")
    args = a.parse_args()
    lib, out = Path(args.lib) / "lib", Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    weights = json.loads(Path(args.layer_weights).read_text()) if args.layer_weights else {}
    metas, moves, unusable = {}, [], {}
    used = 0
    for n in range(args.layers):
        m = json.loads((lib / f"L{n}/meta.json").read_text())
        metas[n] = m
        energy = np.array(m["energy"], dtype=np.float64)
        importance = (1 - args.floor) * energy / max(energy.sum(), 1e-12) + args.floor / 512
        lw = float(weights.get(str(n), 1.0))
        # A failed rounding has a NaN objective: price it as unusable, so the first saves replace it.
        t3_13, t3_2 = (np.nan_to_num(np.array(m[k], dtype=np.float64), nan=1e9) for k in ("t3_13", "t3_2"))
        unusable[n] = (np.flatnonzero(t3_13 >= 1e9).tolist(), np.flatnonzero(t3_2 >= 1e9).tolist())
        used += 512 * (EXPERT13[3] + EXPERT2[3])
        t2_set = m.get("t2_13_set", m.get("cold", []))
        for pos, e in enumerate(t2_set):
            added = lw * importance[e] * (m["t2_13"][pos] - t3_13[e])
            moves.append((max(added, 0) / (EXPERT13[3] - EXPERT13[2]), n, e, "t2_13", pos, -(EXPERT13[3] - EXPERT13[2])))
        for pos, e in enumerate(m.get("t2_2_set", [])):
            added = lw * importance[e] * args.down_weight * (m["t2_2"][pos] - t3_2[e])
            moves.append((max(added, 0) / (EXPERT2[3] - EXPERT2[2]), n, e, "t2_2", pos, -(EXPERT2[3] - EXPERT2[2])))
        for pos, e in enumerate(m.get("t4_2_set", m.get("hot", []))):
            removed = lw * importance[e] * args.down_weight * (t3_2[e] - m["t4_2"][pos])
            moves.append((-max(removed, 0) / (EXPERT2[4] - EXPERT2[3]), n, e, "t4_2", pos, EXPERT2[4] - EXPERT2[3]))
    target = int(args.gib * 2**30)
    saves = sorted((x for x in moves if x[5] < 0), key=lambda x: x[0])
    spends = sorted((x for x in moves if x[5] > 0), key=lambda x: x[0])
    chosen, si, pi = [], 0, 0
    # Merge: take every profitable spend funded by the cheapest saves, while the marginal trade improves error.
    while used > target and si < len(saves):
        chosen.append(saves[si]); used += saves[si][5]; si += 1
    while pi < len(spends) and si < len(saves):
        spend, save = spends[pi], saves[si]
        if -spend[0] <= save[0]:
            break
        chosen.append(spend); used += spend[5]; pi += 1
        while used > target and si < len(saves):
            chosen.append(saves[si]); used += saves[si][5]; si += 1
    while pi < len(spends) and used + spends[pi][5] <= target:
        chosen.append(spends[pi]); used += spends[pi][5]; pi += 1
    t2 = {n: {} for n in metas}
    t2d = {n: {} for n in metas}
    t4 = {n: {} for n in metas}
    for _, n, e, kind, pos, delta in chosen:
        if kind == "t2_13":
            t2[n][e] = pos
        elif kind == "t2_2":
            t2d[n][e] = pos
        elif e not in t2d[n]:
            t4[n][e] = pos
        else:
            used -= delta                 # An expert can't take both; drop the spend.
    if args.plan:
        for n in metas:
            if any(e not in t2[n] for e in unusable[n][0]) or any(e not in t2d[n] and e not in t4[n] for e in unusable[n][1]):
                raise SystemExit(f"L{n}: a failed T3 rounding has no replacement: {unusable[n]}")
        (out / "plan.json").write_text(json.dumps({"lib": str(lib.resolve()), "t2_13": t2, "t2_2": t2d, "t4_2": t4, "total_gib": used / 2**30}))
        print(f"plan {used / 2**30:.2f} GiB; bpw {used * 8 / (48 * 512 * 4915200):.3f}; T2 gate/up {sum(len(v) for v in t2.values())}; T2 down {sum(len(v) for v in t2d.values())}; T4 down {sum(len(v) for v in t4.values())}")
        return
    sizes = {}
    report = {}
    for n in metas:
        fmt13 = np.full(512, 3, dtype=np.uint8)
        fmt2 = np.full(512, 3, dtype=np.uint8)
        for e in t2[n]:
            fmt13[e] = 2
        for e in t4[n]:
            fmt2[e] = 4
        for e in t2d[n]:
            fmt2[e] = 2
        if any(fmt13[e] == 3 for e in unusable[n][0]) or any(fmt2[e] == 3 for e in unusable[n][1]):
            raise SystemExit(f"L{n}: a failed T3 rounding has no replacement: {unusable[n]}")
        off13, off2 = np.zeros(512, dtype=np.int64), np.zeros(512, dtype=np.int64)
        s13 = s2 = 0
        for e in range(512):
            off13[e], off2[e] = s13, s2
            s13 += EXPERT13[int(fmt13[e])]
            s2 += EXPERT2[int(fmt2[e])]
        d = lib / f"L{n}"
        prefix = f"model.language_model.layers.{n}.mlp.experts."
        head, offset = {}, 0
        for name, dtype, shape, size in (("t3_w13", "U8", [s13], s13), ("t3_w2", "U8", [s2], s2), ("t3_off13", "I64", [512], 4096), ("t3_off2", "I64", [512], 4096), ("t3_fmt13", "U8", [512], 512), ("t3_fmt2", "U8", [512], 512)):
            head[prefix + name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
            offset += size
        blob = json.dumps(head).encode()
        blob = blob.ljust(len(blob) + (8 - (8 + len(blob)) % 8) % 8)
        files = {k: open(d / f"{k}.bin", "rb") for k in ("t3_13", "t3_2", "t2_13", "t2_2", "t4_2") if (d / f"{k}.bin").exists()}
        with open(out / f"L{n}.tmp", "wb", buffering=1 << 20) as f:
            f.write(struct.pack("<Q", len(blob)))
            f.write(blob)
            for e in range(512):
                if e in t2[n]:
                    src, size, pos = files["t2_13"], EXPERT13[2], t2[n][e]
                else:
                    src, size, pos = files["t3_13"], EXPERT13[3], e
                src.seek(pos * size)
                f.write(src.read(size))
            for e in range(512):
                if e in t4[n]:
                    src, size, pos = files["t4_2"], EXPERT2[4], t4[n][e]
                elif e in t2d[n]:
                    src, size, pos = files["t2_2"], EXPERT2[2], t2d[n][e]
                else:
                    src, size, pos = files["t3_2"], EXPERT2[3], e
                src.seek(pos * size)
                f.write(src.read(size))
            f.write(off13.tobytes()); f.write(off2.tobytes()); f.write(fmt13.tobytes()); f.write(fmt2.tobytes())
        for handle in files.values():
            handle.close()
        os.replace(out / f"L{n}.tmp", out / f"L{n}.safetensors")
        sizes[str(n)] = {"w13": s13, "w2": s2}
        report[n] = {"t2_13": len(t2[n]), "t2_2": len(t2d[n]), "t4_2": len(t4[n]), "bytes": s13 + s2}
    (out / "sizes.json").write_text(json.dumps(sizes))
    (out / "alloc.json").write_text(json.dumps({"args": vars(args), "layers": report, "total_gib": used / 2**30}))
    print(f"total {used / 2**30:.2f} GiB; bpw {used * 8 / (48 * 512 * 4915200):.3f}; T2 gate/up {sum(len(v) for v in t2.values())}; T2 down {sum(len(v) for v in t2d.values())}; T4 down {sum(len(v) for v in t4.values())}")


if __name__ == "__main__":
    main()
