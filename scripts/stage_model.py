#!/usr/bin/env python3
"""Stage every weight plus required metadata without copying checkpoint bytes."""
import argparse
import json
import shutil
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--source", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
source = a.source.resolve()
a.out.mkdir(parents=True, exist_ok=True)
if any(a.out.iterdir()):
    p.error("output must be empty; do not overwrite an existing upload")
metadata = {"config.json", "generation_config.json", "model.safetensors.index.json", "t3layers.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja", "quantization_config.json", "mtp_pack_plan.json", "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json", "LICENSE"}
files = sorted(f for f in source.iterdir() if f.is_file() and (f.suffix == ".safetensors" or f.name in metadata))
if not (source / "t3layers.json").is_file():
    p.error("source is not a T3 checkpoint")
index = json.loads((source / "model.safetensors.index.json").read_text())
selected = {f.name for f in files}
missing = set(index["weight_map"].values()) - selected
if missing:
    p.error(f"checkpoint references missing shards: {sorted(missing)}")
records = []
for f in files:
    (a.out / f.name).symlink_to(f)
    records.append({"file": f.name, "bytes": f.stat().st_size})
repo = Path(__file__).resolve().parents[1]
shutil.copyfile(repo / "MODEL_CARD.md", a.out / "README.md")
shutil.copyfile(repo / "benchmarks/results.json", a.out / "benchmarks.json")
report = {"files": records, "checkpoint_bytes": sum(r["bytes"] for r in records), "weight_files": sum(r["file"].endswith(".safetensors") for r in records), "note": "Every source safetensors weight file is included. Stale upstream quality/build reports and caches are deliberately excluded; see benchmarks.json for this model's results."}
(a.out / "inventory.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "files"}), flush=True)
