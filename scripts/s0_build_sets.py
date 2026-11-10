#!/usr/bin/env python3
"""Prepare public-only calibration/evaluation; no private traffic is required."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file
from paths import ROOT


def partition(source, session):
    digest = int(hashlib.sha256(f"{source}:{session}".encode()).hexdigest(), 16) % 100
    return "eval_public" if digest < 20 else "calib2" if digest < 40 else "calib"


def write(out, name, rows, spans=None):
    if not rows:
        raise ValueError(f"no sequences collected for {name}")
    ids = np.concatenate([r[2] for r in rows]).astype(np.int32)
    offsets = np.cumsum([0] + [len(r[2]) for r in rows], dtype=np.int64)
    metadata = {"sources": json.dumps([r[0] for r in rows]), "sessions": json.dumps([r[1] for r in rows])}
    if spans is not None:
        metadata["score_spans"] = json.dumps(spans)
    save_file({"input_ids": ids, "offsets": offsets}, out / f"{name}.safetensors", metadata=metadata)
    print(json.dumps({"set": name, "sequences": len(rows), "tokens": len(ids)}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data")
    p.add_argument("--window", type=int, default=6000)
    p.add_argument("--calib-tokens", type=int, default=300000)
    p.add_argument("--calib2-tokens", type=int, default=150000)
    p.add_argument("--eval-tokens", type=int, default=100000)
    a = p.parse_args()
    out = a.data / "sets"
    out.mkdir(parents=True, exist_ok=True)
    rows = {name: [] for name in ("calib", "calib2", "eval_public")}
    used = dict.fromkeys(rows, 0)
    budgets = {"calib": a.calib_tokens, "calib2": a.calib2_tokens, "eval_public": a.eval_tokens}
    for source in ("agents", "chat"):
        shares = {"calib": 15, "calib2": 50, "eval_public": 50}
        limits = {name: budget * shares[name] // 100 if source == "agents"
                  else budget - budget * shares[name] // 100 for name, budget in budgets.items()}
        source_used = dict.fromkeys(rows, 0)
        path = a.data / f"Qwen3.8-Flash-Next-calib-{source}" / "data.jsonl"
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                session = f"{source}:{row['id']}"
                name = partition(source, row["id"])
                ids = np.asarray(row["input_ids"], dtype=np.int32)
                for begin in range(0, min(len(ids), 12000), a.window):
                    remaining = min(budgets[name] - used[name], limits[name] - source_used[name])
                    piece = ids[begin:begin + min(a.window, remaining)]
                    if len(piece) < 64:
                        break
                    rows[name].append((source, session, piece))
                    used[name] += len(piece)
                    source_used[name] += len(piece)
    sessions = {name: {r[1] for r in values} for name, values in rows.items()}
    if any(sessions[x] & sessions[y] for x in sessions for y in sessions if x != y):
        raise ValueError("session leakage across calibration/evaluation")
    for name, values in rows.items():
        write(out, name, values)
    for name in ("nemotron", "multilingual"):
        path = a.data / "Qwen3.8-Flash-Next-evaluation/eval_traces" / f"{name}_trace.json"
        trace = json.loads(path.read_text())["rows"]
        values = [(name, f"{name}:{i}", np.asarray(r["input_ids"] + r["response_ids"], dtype=np.int32)) for i, r in enumerate(trace)]
        write(out, f"eval_{name}", values, [r["score_spans"] for r in trace])


if __name__ == "__main__":
    main()
