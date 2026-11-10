#!/usr/bin/env python3
"""S3.1: assemble a pruned, all-W3 candidate. CPU only; streams, so RAM stays small.

Keeps the live checkpoint's shard layout: the PLE-only shard and MTP are hard links, and
every other shard is rewritten with library W3 experts, renumbered and with pruned ones gone.
"""
import argparse
import json
import os
import re
import shutil
import struct
from pathlib import Path

import numpy as np
from safetensors import safe_open

from requant_common import LIVE, ROOT, Checkpoint, Job

EXPERT = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(qweight|qzeros|scales)$")
ROUTER = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.gate\.weight$")
EDGE = {*range(12), *range(36, 48)}
MIN_ROUTED = 32


class TensorFile:
    def __init__(self, path):
        with open(path, "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(size))
        self.base = 8 + size
        self.path = path

    def location(self, name):
        entry = self.header[name]
        return self.path, self.base + entry["data_offsets"][0], entry["data_offsets"][1] - entry["data_offsets"][0], entry


def choose(stats_dir, experts, job):
    kept, disagreements = {}, []
    for layer in range(48):
        with safe_open(stats_dir / f"L{layer}.safetensors", framework="np") as f:
            routed, reap, man = f.get_tensor("routed"), f.get_tensor("reap"), f.get_tensor("man")
        reliable = routed >= MIN_ROUTED
        score = np.where(reliable, reap / np.maximum(routed, 1), 0)
        other = np.where(reliable, man / np.maximum(routed, 1), 0)
        top = np.argsort(-score)[:experts]
        kept[layer] = sorted(int(e) for e in top)
        top_man = set(int(e) for e in np.argsort(-other)[:experts])
        jaccard = len(set(kept[layer]) & top_man) / len(set(kept[layer]) | top_man)
        if jaccard < 0.8:
            disagreements.append((layer, round(jaccard, 3)))
    job.log(f"REAP/MAN disagreement (Jaccard<0.8): {disagreements}")
    return kept


def choose_by_domain(stats_dir, experts, job, general_scale=1.0):
    """Rank experts by contribution share within each domain; keep those ranked well in any."""
    kept, coverage = {}, []
    for layer in range(48):
        with safe_open(stats_dir / f"L{layer}.safetensors", framework="np") as f:
            reap, count = f.get_tensor("reap"), f.get_tensor("count")
        share = reap / np.maximum(reap.sum(axis=0, keepdims=True), 1e-12)
        ranks = np.argsort(np.argsort(-share, axis=0), axis=0).astype(np.float64)
        ranks[:, 3] *= general_scale  # The language-balanced domain competes at lower priority above 1.
        best = ranks.min(axis=1)
        order = np.lexsort((-share.max(axis=1), best))
        kept[layer] = sorted(int(e) for e in order[:experts])
        coverage.append(share[kept[layer]].sum(axis=0))
    job.log(f"mean share of each domain's contribution kept: {np.mean(coverage, axis=0).round(4).tolist()}; worst layer: {np.min(coverage, axis=0).round(4).tolist()}")
    return kept


def write_shard(destination, entries):
    """entries: (name, dtype, shape, path, offset, length) or (name, dtype, shape, bytes)."""
    header, cursor = {}, 0
    for entry in entries:
        length = entry[5] if len(entry) == 6 else len(entry[3])
        header[entry[0]] = {"dtype": entry[1], "shape": entry[2], "data_offsets": [cursor, cursor + length]}
        cursor += length
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-(8 + len(blob)) % 8)
    temporary = destination.with_suffix(".tmp")
    with open(temporary, "wb") as out:
        out.write(struct.pack("<Q", len(blob)) + blob)
        for entry in entries:
            if len(entry) != 6:
                out.write(entry[3])
                continue
            with open(entry[3], "rb") as source:
                source.seek(entry[4])
                remaining = entry[5]
                while remaining:
                    chunk = source.read(min(remaining, 1 << 24))
                    out.write(chunk)
                    remaining -= len(chunk)
    temporary.replace(destination)
    return cursor


def rewrite_quant(config, kept, renumber):
    """Every kept expert gets explicit W3G128: W2 layers relied on the top-level default."""
    quant = config["quantization_config"]
    extra = {}
    for key, value in quant["extra_config"].items():
        if re.match(r"^model\.language_model\.layers\.\d+\.mlp\.experts\.\d+\.", key):
            continue
        extra[key] = value
    for layer, experts in kept.items():
        for new in range(len(experts)):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                extra[f"model.language_model.layers.{layer}.mlp.experts.{new}.{proj}"] = {"bits": 3, "group_size": 128}
    quant["extra_config"] = extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--name", required=True)
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--edge", choices=["klee", "gptq"], default="klee")
    parser.add_argument("--s2", default=str(ROOT / "s2"))
    parser.add_argument("--head", default=str(ROOT / "s3-head/lm_head.safetensors"))
    parser.add_argument("--domain-stats", default=None, help="s2b stats directory: select per domain")
    parser.add_argument("--general-rank-scale", type=float, default=1.0)
    args = parser.parse_args()
    job = Job(f"s3-{args.name}")
    out = ROOT / "cand" / args.name
    out.mkdir(parents=True, exist_ok=True)
    s2 = Path(args.s2)
    kept = choose_by_domain(Path(args.domain_stats), args.experts, job, args.general_rank_scale) if args.domain_stats else choose(s2 / "stats", args.experts, job)
    renumber = {layer: {old: new for new, old in enumerate(experts)} for layer, experts in kept.items()}
    live = Checkpoint(LIVE)
    libraries = {}
    head = TensorFile(args.head) if Path(args.head).exists() else None
    job.log(f"head: {'GPTQ INT8' if head else 'live'}")

    index = json.loads((LIVE / "model.safetensors.index.json").read_text())
    shards = {}
    for name, shard in index["weight_map"].items():
        shards.setdefault(shard, []).append(name)
    new_map, total, expert_bytes = {}, 0, 0
    for shard, names in sorted(shards.items()):
        destination = out / shard
        hard_link = shard.startswith("mtp-") or all(".ple." in n for n in names)
        if hard_link:
            if not destination.exists():
                os.link(LIVE / shard, destination)
            for name in names:
                new_map[name] = shard
            continue
        entries = []
        for name in names:
            match = EXPERT.match(name)
            if match:
                layer, expert, proj, part = int(match[1]), int(match[2]), match[3], match[4]
                if expert not in renumber[layer]:
                    continue
                new_name = f"model.language_model.layers.{layer}.mlp.experts.{renumber[layer][expert]}.{proj}.{part}"
                if layer in EDGE and args.edge == "klee":
                    path, offset, length, entry = live.location(name)
                else:
                    if layer not in libraries:
                        libraries[layer] = TensorFile(s2 / f"lib/L{layer}/w3.safetensors")
                    path, offset, length, entry = libraries[layer].location(name)
                    if part == "qzeros" and entry["shape"][0] > entry["shape"][1]:
                        # S2 libraries built before the fix store qzeros as [words, groups].
                        raw = np.fromfile(path, dtype=np.int32, count=length // 4, offset=offset).reshape(entry["shape"]).T
                        entries.append((new_name, entry["dtype"], list(raw.shape), raw.tobytes()))
                        new_map[new_name] = shard
                        expert_bytes += length
                        continue
                entries.append((new_name, entry["dtype"], entry["shape"], path, offset, length))
                new_map[new_name] = shard
                expert_bytes += length
                continue
            match = ROUTER.match(name)
            if match:
                layer = int(match[1])
                path, offset, _, entry = live.location(name)
                rows = np.memmap(path, dtype=np.uint16, mode="r", offset=offset, shape=tuple(entry["shape"]))
                data = rows[kept[layer]].tobytes()
                entries.append((name, entry["dtype"], [len(kept[layer]), entry["shape"][1]], data))
                new_map[name] = shard
                continue
            if name == "lm_head.weight" and head:
                path, offset, length, entry = head.location(name)
            else:
                path, offset, length, entry = live.location(name)
            entries.append((name, entry["dtype"], entry["shape"], path, offset, length))
            new_map[name] = shard
        total += write_shard(destination, entries)
        job.log(f"{shard}: {len(entries)} tensors")
        if job.stop:
            job.log("stopped; completed shards are rewritten on resume")
            raise SystemExit(0)

    for path in LIVE.iterdir():
        if path.is_file() and path.name not in shards and path.suffix != ".safetensors" and path.name != "model.safetensors.index.json":
            shutil.copy2(path, out / path.name)
    index["weight_map"] = new_map
    index["metadata"]["total_size"] = total
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=1))
    for filename in ("config.json", "quantization_config.json"):
        path = out / filename
        document = json.loads(path.read_text())
        if filename == "config.json":
            document["text_config"]["num_experts"] = args.experts
            rewrite_quant(document, kept, renumber)
        else:
            rewrite_quant({"quantization_config": document}, kept, renumber)
        path.write_text(json.dumps(document, indent=1))
    (out / "kept_experts.json").write_text(json.dumps(kept))
    job.done({"experts": args.experts, "edge": args.edge, "expert_gib": expert_bytes / 2**30, "rewritten_gib": total / 2**30, "tensors": len(new_map)})


if __name__ == "__main__":
    main()
