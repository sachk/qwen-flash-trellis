#!/usr/bin/env python3
"""Replay a running API without restarting it. Use an otherwise idle API for timings."""
import argparse
import concurrent.futures
import json
import os
import re
import statistics
import time
import urllib.request
from pathlib import Path


def headers():
    key = os.environ.get("OPENAI_API_KEY", "")
    if os.environ.get("OPENAI_API_KEY_FILE"):
        key = Path(os.environ["OPENAI_API_KEY_FILE"]).read_text().strip()
    return {"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})}


def metrics(base):
    req = urllib.request.Request(base + "/metrics", headers=headers())
    with urllib.request.urlopen(req, timeout=10) as r:
        text = r.read().decode()
    result = {}
    for line in text.splitlines():
        match = re.match(r"(vllm:spec_decode[^ {]*)(?:\{[^}]*\})?\s+([\d.eE+\-]+)$", line)
        if match:
            result[match[1]] = result.get(match[1], 0) + float(match[2])
    return result


def request_one(base, model, prompt, tokens, seed):
    body = {"model": model, "prompt": prompt, "max_tokens": tokens, "temperature": 1.0, "top_k": 20,
            "top_p": 0.95, "seed": seed, "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(), headers=headers())
    start = time.monotonic()
    first = last = None
    usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            value = json.loads(line[6:])
            if value.get("usage"):
                usage = value["usage"]
            if any(c.get("text") for c in value.get("choices", [])):
                now = time.monotonic()
                first = first or now
                last = now
    if usage is None or first is None or last is None or usage["completion_tokens"] != tokens:
        raise RuntimeError("missing token usage, empty stream, or completion length mismatch")
    elapsed = last - first
    if elapsed <= 0:
        raise RuntimeError("completion fits one SSE event; increase --tokens for meaningful decode timing")
    return {"prompt_tokens": usage["prompt_tokens"], "completion_tokens": tokens, "ttft_s": first - start,
            "decode_s": elapsed, "decode_tps": (tokens - 1) / elapsed,
            "prefill_tps": usage["prompt_tokens"] / (first - start)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8099")
    p.add_argument("--model", default="qwen3.8-flash-next")
    p.add_argument("--prompts", type=Path, required=True, help="JSONL rows with a text or token-ID prompt field")
    p.add_argument("--streams", type=int, nargs="+", default=[1, 4])
    p.add_argument("--rounds", type=int, default=72, help="4-stream phase uses half this many rounds")
    p.add_argument("--tokens", type=int, default=400)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prompts = [json.loads(line)["prompt"] for line in a.prompts.read_text().splitlines() if line.strip()]
    if not prompts or a.rounds < 1 or a.tokens < 2 or any(s < 1 for s in a.streams):
        p.error("need prompts, positive rounds/streams and at least two output tokens")
    result = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, "phases": {}}
    for streams in a.streams:
        before = metrics(a.base_url)
        records = []
        started = time.monotonic()
        rounds = a.rounds if streams == 1 else max(1, a.rounds // 2)
        with concurrent.futures.ThreadPoolExecutor(max_workers=streams) as pool:
            for round_index in range(rounds):
                futures = []
                for i in range(streams):
                    prompt = prompts[(round_index + i) % len(prompts)]
                    salt = (round_index * 7919 + i + 1) % 200000
                    prompt = [salt] + prompt if isinstance(prompt, list) else f"Replay nonce {salt}.\n{prompt}"
                    futures.append(pool.submit(request_one, a.base_url, a.model, prompt, a.tokens, 3821 + i))
                records.extend(f.result() for f in futures)
        after = metrics(a.base_url)
        delta = {k: after.get(k, 0) - before.get(k, 0) for k in after}
        accepted = delta.get("vllm:spec_decode_num_accepted_tokens_total", 0)
        drafts = delta.get("vllm:spec_decode_num_drafts_total", 0)
        each = statistics.mean(r["decode_tps"] for r in records)
        acceptance = accepted / drafts if drafts else None
        phase = {"streams": streams, "rounds": rounds, "decode_tps_each": each, "decode_tps_total": each * streams,
                 "prefill_tps": statistics.mean(r["prefill_tps"] for r in records),
                 "accepted_per_step": acceptance, "estimated_ms_step": 1000 * (1 + acceptance) / each if acceptance is not None else None,
                 "wall_s": time.monotonic() - started, "requests": records,
                 "note": "Acceptance counters are server-global; concurrent unrelated traffic invalidates this estimate."}
        result["phases"][str(streams)] = phase
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in phase.items() if k != "requests"}), flush=True)


if __name__ == "__main__":
    main()
