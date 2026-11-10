#!/usr/bin/env python3
"""Candidate directory: the base checkpoint without routed expert tensors, plus one T3 shard per layer.

Unchanged files are hard-linked. A shard holding expert tensors is rewritten without them, and each new shard's
non-expert tensors stay byte-identical. Output is ready for vLLM with KUDA_T3MOE set to the directory.
"""
import argparse, json, os, re, struct, subprocess
from pathlib import Path

EXPERT = re.compile(r"\.mlp\.experts\.")


def rewrite(src, dst, drop):
    with open(src, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
        base = 8 + size
        keep = {k: v for k, v in header.items() if k != "__metadata__" and not drop(k)}
        new, offset = {}, 0
        order = sorted(keep, key=lambda k: keep[k]["data_offsets"][0])
        for k in order:
            a, b = keep[k]["data_offsets"]
            new[k] = {**keep[k], "data_offsets": [offset, offset + b - a]}
            offset += b - a
        if "__metadata__" in header:
            new["__metadata__"] = header["__metadata__"]
        blob = json.dumps(new, separators=(",", ":")).encode()
        pad = (8 - (8 + len(blob)) % 8) % 8
        blob += b" " * pad
        with open(dst, "wb") as g:
            g.write(struct.pack("<Q", len(blob)))
            g.write(blob)
            for k in order:
                a, b = keep[k]["data_offsets"]
                f.seek(base + a)
                left = b - a
                while left:
                    chunk = f.read(min(left, 1 << 26))
                    g.write(chunk)
                    left -= len(chunk)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--base", required=True)
    a.add_argument("--lib", required=True, help="directory of L{n}.safetensors from t3_alloc")
    a.add_argument("--out", required=True)
    a.add_argument("--mtp", default=None, help="directory from mtp_round.py")
    args = a.parse_args()
    base, lib, out = Path(args.base), Path(args.lib), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads((base / "model.safetensors.index.json").read_text())
    weight_map = {}
    has_experts = {}
    for k, shard in index["weight_map"].items():
        is_routed = EXPERT.search(k) and "mtp." not in k and "shared_expert" not in k
        if is_routed:
            has_experts[shard] = True
        else:
            weight_map[k] = shard
    for item in base.iterdir():
        if item.name in ("model.safetensors.index.json", "t3layers.json"):
            continue
        target = out / item.name
        if target.exists():
            continue
        if item.name in has_experts:
            drop = lambda k: bool(EXPERT.search(k)) and "mtp." not in k and "shared_expert" not in k
            rewrite(item, target, drop)
            print("rewrote", item.name, flush=True)
        elif item.is_file():
            os.link(item, target)
    for n in range(48):
        shard = f"t3-L{n}.safetensors"
        target = out / shard
        if target.exists():
            target.unlink()
        subprocess.run(["cp", "--reflink=auto", str(lib / f"L{n}.safetensors"), str(target)], check=True)
    json_sizes = json.loads((lib / "sizes.json").read_text())
    if args.mtp:
        target = out / "mtp-t3.safetensors"
        if target.exists():
            target.unlink()
        os.link(Path(args.mtp) / "mtp-t3.safetensors", target)
        json_sizes["mtp"] = json.loads((Path(args.mtp) / "sizes.json").read_text())
        for t in ("t3_w13", "t3_w2", "t3_off13", "t3_off2", "t3_fmt13", "t3_fmt2"):
            weight_map[f"mtp.layers.0.mlp.experts.{t}"] = "mtp-t3.safetensors"
    (out / "t3layers.json").write_text(json.dumps(json_sizes))
    for n in range(48):
        for t in ("t3_w13", "t3_w2", "t3_off13", "t3_off2", "t3_fmt13", "t3_fmt2"):
            weight_map[f"model.language_model.layers.{n}.mlp.experts.{t}"] = f"t3-L{n}.safetensors"
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": index.get("metadata", {}), "weight_map": weight_map}))
    print("assembled", out, flush=True)


if __name__ == "__main__":
    main()
