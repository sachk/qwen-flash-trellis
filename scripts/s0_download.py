#!/usr/bin/env python3
"""S0.2/S0.3: resumably fetch the FP8 source (minus PLE and vision) and datasets.

Run with the vLLM venv's Python. Safe to rerun: completed files are skipped.
"""
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from paths import ROOT, FP8_SRC

SRC_REPO = "Qwen/Qwen3.8-Flash-Next-FP8"
SRC_REV = "236dfdf285828023ca3bcd3f37366c58a3469b13"
SRC_DIR = FP8_SRC
DATA_DIR = ROOT / "data"
DATASETS = {
    "Vemloir/Qwen3.8-Flash-Next-calib-agents": ("e6df0813badc40448e82de3d844d52a48a4d41ad", None),
    "Vemloir/Qwen3.8-Flash-Next-calib-chat": ("7fa0edebb8b1dc97fe2d220fc4da43a00b78d36c", None),
    "Vemloir/Qwen3.8-Flash-Next-evaluation": (
        "15b2f5e61a200cb8a79f8ebb4a47c52e5933652f",
        [
            "README.md",
            "eval_traces/nemotron_trace.json",
            "eval_traces/multilingual_trace.json",
            "results/split_grid.json",
            "results/layer_sens.json",
        ],
    ),
}
SKIP_MARKERS = ("ple", "ngram", "n_gram", "visual", "vision")


def source_patterns():
    index = hf_hub_download(
        SRC_REPO, "model.safetensors.index.json", revision=SRC_REV, local_dir=SRC_DIR
    )
    weight_map = json.load(open(index))["weight_map"]
    shards = {}
    for name, shard in weight_map.items():
        shards.setdefault(shard, []).append(any(m in name.lower() for m in SKIP_MARKERS))
    keep = sorted(s for s, skips in shards.items() if not all(skips))
    print(f"{len(keep)}/{len(shards)} shards kept", flush=True)
    return keep + ["*.json", "*.jinja", "*.txt", "README.md", "LICENSE"]


def main():
    snapshot_download(
        SRC_REPO, revision=SRC_REV, local_dir=SRC_DIR,
        allow_patterns=source_patterns(), max_workers=8,
    )
    print("source complete", flush=True)
    for repo, (rev, patterns) in DATASETS.items():
        snapshot_download(
            repo, repo_type="dataset", revision=rev,
            local_dir=DATA_DIR / repo.split("/")[1], allow_patterns=patterns,
        )
        print(f"{repo} complete", flush=True)


if __name__ == "__main__":
    main()
