#!/usr/bin/env python3
"""GPQA Diamond pass@1 against a live OpenAI-compatible API, with full provenance.

One sample per question, normal EOS (no forced length), native chat template on the
server side, streaming so time-to-first-token and inter-chunk gaps are measurable.
Every request is recorded as it finishes; transport failures are recorded, never
scored as a model answer.

Prompt template, choice-shuffle semantics and answer parsing mirror Inspect:
  inspect_evals gpqa (commit a68dade7efb803ef50c8442d3f16264b293d1414) and
  inspect_ai 0.3.289 (SINGLE_ANSWER_TEMPLATE_COT, Dataset.shuffle_choices,
  solver._multiple_choice.parse_answers). See local-docs/gpqa/PROTOCOL.md.
Stdlib only, so it runs inside the serving venv without extra dependencies.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# Inspect SINGLE_ANSWER_TEMPLATE_COT, verbatim (chain-of-thought, single answer).
TEMPLATE = r"""Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{question}

{choices}"""

# gpqa_diamond.csv as pinned by inspect_evals (SHA256 documented there and re-verified here).
DATASET_SHA256 = "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305"
ANSWER_FIELDS = ("Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")

write_lock = threading.Lock()


def letter(index):
    return chr(ord("A") + index)


def build_prompt(question, choices):
    letters = ",".join(letter(i) for i in range(len(choices)))
    body = "\n".join(f"{letter(i)}) {value}" for i, value in enumerate(choices))
    return TEMPLATE.format(letters=letters, question=question, choices=body)


def load_items(csv_path, shuffle_seed):
    """Read the pinned CSV and permute choices exactly like inspect_ai Dataset.shuffle_choices."""
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    rand = random.Random(shuffle_seed)
    items = []
    for index, row in enumerate(rows):
        for field in ("Question",) + ANSWER_FIELDS:
            if row.get(field) is None:
                raise SystemExit(f"row {index} is missing field {field!r}")
        choices = [str(row[field]) for field in ANSWER_FIELDS]  # raw CSV lists the correct answer first
        positions = list(range(len(choices)))
        rand.shuffle(positions)
        # position_map[original_position] = presented letter, as in inspect_ai._dataset
        position_map = {i: letter(new_i) for new_i, i in enumerate(positions)}
        items.append({
            "index": index,
            "item_id": row.get("Record ID") or f"row{index}",
            "question": str(row["Question"]),
            "choices": [choices[i] for i in positions],
            "correct_letter": position_map[0],
            "position_map": position_map,
            "domain": row.get("High-level domain", ""),
            "subdomain": row.get("Subdomain", ""),
        })
    return items


def parse_answer(completion, allowed):
    """Mirror of inspect_ai parse_answers for single-answer MCQ.

    Returns (letter, status). status is "ok", "unparsed" (no ANSWER line) or
    "no_valid_answer" (an ANSWER line that is not exactly one allowed letter).
    """
    text = re.sub(r"\x24\s*([A-Za-z\d][A-Za-z\d ,]*?)\s*\x24", r"\1", completion)
    text = re.sub(r"\*\*\s*([A-Za-z\d][A-Za-z\d ,]*?)\s*\*\*", r"\1", text)
    text = re.sub(r"(?i)(^|[:,])(\s*)\(([A-Za-z\d])\)(?=\s*(?:,|\.|\n|\Z))", r"\1\2\3", text)
    matches = re.findall(r"(?i)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)", text, flags=re.MULTILINE)
    if not matches:
        matches = re.findall(r"(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)", text)
    if not matches:
        return None, "unparsed"
    matched = matches[-1].strip().rstrip(".").upper()
    tokens = [token for token in matched.split(",") if token]
    if len(tokens) == 1 and tokens[0] in allowed:
        return tokens[0], "ok"
    return None, "no_valid_answer"


def api_headers():
    key = os.environ.get("OPENAI_API_KEY", "")
    key_file = os.environ.get("OPENAI_API_KEY_FILE")
    if key_file:
        key = Path(key_file).read_text().strip()
    return {"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})}


def stream_chat(base_url, body, timeout):
    """POST /v1/chat/completions with SSE streaming; return the raw observation.

    Chunk counts are never treated as token counts: speculative decoding emits several
    tokens per chunk, so token totals come only from the usage object the API reports.
    """
    request = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers=api_headers())
    started = time.monotonic()
    first_token = first_content = last_chunk = None
    gaps = []
    content_parts, reasoning_parts = [], []
    delta_keys, finish_reason, usage, model, completion_id = set(), None, None, None, None
    sse_events, saw_done = 0, False
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for line in response:
                if line.strip() == b"data: [DONE]":
                    saw_done = True
                    break
                if not line.startswith(b"data: "):
                    continue
                value = json.loads(line[6:])
                model = value.get("model") or model
                completion_id = value.get("id") or completion_id
                if value.get("usage"):
                    usage = value["usage"]
                for choice in value.get("choices", []):
                    delta = choice.get("delta") or {}
                    delta_keys.update(delta)
                    piece = delta.get("content") or ""
                    thinking = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if piece or thinking:
                        now = time.monotonic()
                        if first_token is None:
                            first_token = now
                        elif last_chunk is not None:
                            gaps.append(now - last_chunk)
                        if piece and first_content is None:
                            first_content = now
                        last_chunk = now
                        sse_events += 1
                        content_parts.append(piece)
                        reasoning_parts.append(thinking)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
        reason = getattr(error, "reason", error)
        detail = ""
        if isinstance(error, urllib.error.HTTPError):
            detail = error.read().decode("utf-8", "replace")[:2000]
        return {"ok": False, "error_type": type(error).__name__,
                "error": str(reason)[:2000], "error_body": detail,
                "wall_s": time.monotonic() - started}
    elapsed = time.monotonic() - started
    details = (usage or {}).get("prompt_tokens_details") or {}
    out_details = (usage or {}).get("completion_tokens_details") or {}
    tokens_out = (usage or {}).get("completion_tokens")
    decode_window = (last_chunk - first_token) if (first_token and last_chunk and last_chunk > first_token) else None
    return {
        "ok": saw_done and usage is not None and finish_reason is not None,
        "error_type": None if saw_done and usage is not None and finish_reason is not None else "IncompleteStream",
        "stream_done": saw_done,
        "request_wall_s": elapsed,
        "ttft_s": (first_token - started) if first_token else None,
        "first_content_s": (first_content - started) if first_content else None,
        "decode_window_s": decode_window,
        "decode_tps": (tokens_out - 1) / decode_window if (tokens_out and decode_window) else None,
        "chunk_gap_ms_mean": statistics.fmean(gaps) * 1000 if gaps else None,
        "chunk_gap_ms_p95": sorted(gaps)[max(0, int(0.95 * len(gaps)) - 1)] * 1000 if gaps else None,
        "sse_text_events": sse_events,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": tokens_out,
        "total_tokens": (usage or {}).get("total_tokens"),
        "cached_tokens": details.get("cached_tokens"),
        "reasoning_tokens": out_details.get("reasoning_tokens"),
        "usage_present": usage is not None,
        "usage_raw": usage,
        "delta_keys": sorted(delta_keys),
        "finish_reason": finish_reason,
        "model": model,
        "completion_id": completion_id,
        "answer_text": "".join(content_parts),
        "reasoning_text": "".join(reasoning_parts),
        "response_body": None,
    }


def request_item(base_url, model, item, args, seed):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": build_prompt(item["question"], item["choices"])}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k, "min_p": args.min_p,
        "seed": seed, "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"reasoning_effort": args.reasoning_effort},
    }
    result = stream_chat(base_url, body, args.timeout)
    record = {
        "index": item["index"], "item_id": item["item_id"], "seed": seed,
        "correct_letter": item["correct_letter"], "position_map": item["position_map"],
        "domain": item["domain"], "subdomain": item["subdomain"],
        "model": model, "sampling": {k: body[k] for k in
                                     ("max_tokens", "temperature", "top_p", "top_k", "min_p")},
        "reasoning_effort": args.reasoning_effort,
        "prompt_sha256": hashlib.sha256(
            build_prompt(item["question"], item["choices"]).encode()).hexdigest(),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    record.update(result)
    if not result["ok"]:
        record.update({"status": "transport_error", "correct": False, "parsed_letter": None})
        return record, False
    if result["finish_reason"] == "length":
        record["truncated"] = True
    if not result["answer_text"].strip():
        record.update({"status": "empty", "correct": False, "parsed_letter": None})
        return record, True
    allowed = {letter(i) for i in range(len(item["choices"]))}
    parsed, status = parse_answer(result["answer_text"], allowed)
    record.update({"status": status, "parsed_letter": parsed,
                   "correct": parsed is not None and parsed == item["correct_letter"]})
    return record, True


def load_done(path):
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[record["item_id"]] = record
    return done


def metric_counters(base_url):
    """Aggregate vllm:* counters by metric name (labels dropped). Server-global: other
    clients' traffic is included, so deltas are only indicative."""
    request = urllib.request.Request(base_url.rstrip("/") + "/metrics", headers=api_headers())
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            text = response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError):
        return None
    counters = {}
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        head, _, value = line.partition(" ")
        name = head.split("{", 1)[0]
        try:
            counters[name] = counters.get(name, 0.0) + float(value)
        except ValueError:
            continue
    return counters


def gpu_facts():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.current.graphics,clocks.max.graphics,"
                             "power.draw,pstate,utilization.gpu,memory.used", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        return {"available": out.returncode == 0, "lines": out.stdout.strip().splitlines(),
                "error": out.stderr.strip()[:400] or None}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "lines": [], "error": str(error)[:400]}


def service_facts(base_url, unit):
    """Sanitized identity/config snapshot of the live serving process. Read-only."""
    facts = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "base_url": base_url}
    try:
        request = urllib.request.Request(base_url.rstrip("/") + "/v1/models", headers=api_headers())
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.load(response)
        facts["served"] = [{"id": m.get("id"), "root": m.get("root"),
                            "max_model_len": m.get("max_model_len")} for m in payload.get("data", [])]
    except (urllib.error.URLError, OSError) as error:
        facts["served_error"] = str(error)[:400]
    if unit:
        try:
            out = subprocess.run(["systemctl", "show", unit, "-p", "Id", "-p", "ActiveState", "-p",
                                  "SubState", "-p", "MainPID", "-p", "ExecMainStartTimestamp",
                                  "-p", "FragmentPath"], capture_output=True, text=True, timeout=20)
            facts["unit"] = {k: v for k, v in
                             (line.split("=", 1) for line in out.stdout.strip().splitlines() if "=" in line)}
            env = subprocess.run(["systemctl", "show", unit, "-p", "Environment"],
                                 capture_output=True, text=True, timeout=20).stdout
            kept, secret = [], re.compile(r"(?i)(key|token|secret|password)")
            for entry in env.replace("\\n", " ").split():
                name, _, _ = entry.partition("=")
                kept.append(f"{name}=<redacted>" if secret.search(name) else entry)
            facts["unit_env"] = [entry for entry in kept
                                 if entry.split("=", 1)[0].split()[-1].startswith(("KUDA", "QSA", "CUDA_VISIBLE"))]
        except (OSError, subprocess.TimeoutExpired) as error:
            facts["unit_error"] = str(error)[:400]
    facts["gpu"] = gpu_facts()
    return facts


def aggregate(records, failures, denominator, wall_s, counters_delta, isolated=False):
    attempted = records + failures
    scored = [r for r in attempted if r.get("ok")]
    known = lambda key: [r[key] for r in scored if isinstance(r.get(key), (int, float))]

    def stat(values):
        if not values:
            return None
        ordered = sorted(values)
        return {"n": len(values), "mean": statistics.fmean(values), "median": ordered[len(ordered) // 2],
                "min": ordered[0], "max": ordered[-1], "p95": ordered[max(0, int(0.95 * len(ordered)) - 1)]}

    def token_sum(key):
        values = known(key)
        return {"total": sum(values) if len(values) == len(scored) else None,
                "reported_items": len(values), "unknown_items": len(scored) - len(values),
                "note": "null total means at least one item had no API-reported value; never inferred"}

    statuses = {}
    for record in attempted:
        statuses[record.get("status", "?")] = statuses.get(record.get("status", "?"), 0) + 1
    correct = sum(1 for r in attempted if r.get("correct"))
    complete = len(attempted) == denominator
    valid = complete and len(scored) == denominator
    timed = [r for r in scored if not r.get("resumed")]
    decode_windows = [r["decode_window_s"] for r in timed if r.get("decode_window_s")]
    completion = [r["completion_tokens"] for r in timed if r.get("completion_tokens") is not None]
    return {
        "denominator": denominator,
        "attempted_records": len(attempted), "failed_records": len(failures), "complete": complete,
        "valid_quality_result": valid,
        "correct": correct, "accuracy_strict_denominator": correct / denominator if valid and denominator else None,
        "accuracy_scored_only": correct / len(scored) if scored else None,
        "statuses": statuses,
        "transport_errors": sum(1 for r in attempted if r.get("status") == "transport_error"),
        "truncated_items": sum(1 for r in attempted if r.get("truncated")),
        "usage_missing_items": sum(1 for r in scored if not r.get("usage_present")),
        "tokens": {key: token_sum(key) for key in
                   ("prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_tokens", "total_tokens")},
        "timing_s": {key: stat([r[key] for r in timed if isinstance(r.get(key), (int, float))])
                     for key in ("ttft_s", "first_content_s", "request_wall_s", "decode_window_s")},
        "chunk_gap_ms": {"mean_of_means": (statistics.fmean(
            [r["chunk_gap_ms_mean"] for r in timed if r.get("chunk_gap_ms_mean")])
            if any(r.get("chunk_gap_ms_mean") for r in timed) else None),
            "note": "mean interval between SSE text chunks, not per-token intervals; speculative decoding "
                    "emits several tokens per chunk"},
        "decode_tps_per_request": stat([r["decode_tps"] for r in timed if r.get("decode_tps")]),
        "throughput": {
            "run_wall_s": wall_s,
            "completion_tokens_per_run_s": (sum(completion) / wall_s) if wall_s and completion else None,
            "completion_tokens_per_stream_s": (sum(completion) / sum(decode_windows)) if decode_windows else None,
            "note": "Current invocation only, excluding resumed records; aggregate across concurrent streams. "
                    "Overall token totals include all completed items.",
        },
        "server_counters_delta": counters_delta,
        "external_successful_requests": (
            max(0, counters_delta["vllm:request_success_total"] - len(timed))
            if counters_delta and "vllm:request_success_total" in counters_delta else None),
        "speed_caveat": (
            "Dedicated benchmark port; compare server request counts with benchmark completions to detect "
            "unexpected traffic. Throughput includes real prefill and changing request lengths."
            if isolated else
            "Server-global counters and timings include unrelated client traffic; not isolated."),
    }


def summarize(out, records, failures, denominator, wall_s, counters_before, counters_after, isolated=False):
    delta = None
    if counters_before and counters_after:
        delta = {name: counters_after[name] - counters_before.get(name, 0.0) for name in counters_after}
    summary = aggregate(records, failures, denominator, wall_s, delta, isolated)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8099")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--streams", type=int, default=2, help="in-flight requests, exactly")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--seed-base", type=int, default=20261114)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--unit", default="vllm.service", help="serving unit for the identity snapshot")
    parser.add_argument("--isolated", action="store_true", help="record that the caller isolated the backend port")
    parser.add_argument("--skip-existing", action="store_true",
                        help="resume: never re-request an item already recorded (scored or failed)")
    parser.add_argument("--dataset-sha256", default=DATASET_SHA256)
    parser.add_argument("--summary-interval", type=int, default=10)
    args = parser.parse_args()

    digest = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    if args.dataset_sha256 and digest != args.dataset_sha256:
        raise SystemExit(f"dataset sha256 {digest} != pinned {args.dataset_sha256}")
    items = load_items(args.csv, args.shuffle_seed)
    items = items[args.offset:][:(args.limit if args.limit is not None else None)]
    if args.streams < 1 or args.max_tokens < 1:
        raise SystemExit("--streams and --max-tokens must be positive")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    answers_path, failures_path = out / "answers.jsonl", out / "failures.jsonl"
    if not args.skip_existing and any(p.exists() and p.stat().st_size for p in (answers_path, failures_path)):
        raise SystemExit("output already contains attempts; choose a fresh --out or explicitly --skip-existing")
    done = load_done(answers_path) if args.skip_existing else {}
    done.update(load_done(failures_path) if args.skip_existing else {})
    done = {key: value for key, value in done.items()
            if value.get("ok") and value.get("usage_present") and value.get("finish_reason")}
    for value in done.values():
        value["resumed"] = True

    counters_before = metric_counters(args.base_url)
    facts = service_facts(args.base_url, args.unit)
    facts["gpu_during"] = []
    run_config = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "dataset": {"path": str(args.csv), "sha256": digest, "items_total": len(load_items(args.csv, args.shuffle_seed))},
                  "template_sha256": hashlib.sha256(TEMPLATE.encode()).hexdigest(),
                  "server_facts": facts, "counters_before": counters_before,
                  "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "pid": os.getpid(), "denominator": len(items), "resumed_items": len(done)}
    (out / "run.json").write_text(json.dumps(run_config, indent=2) + "\n")

    pending = [item for item in items if item["item_id"] not in done]
    print(json.dumps({"event": "start", "items": len(items), "pending": len(pending),
                      "pid": os.getpid(), "streams": args.streams}), flush=True)

    records, failures = list(done.values()), []
    started = time.monotonic()
    completed = 0
    aborted = threading.Event()

    def handle(item):
        nonlocal completed
        if aborted.is_set():
            return None
        seed = args.seed_base + item["index"]
        record, scored = request_item(args.base_url, args.model, item, args, seed)
        path = answers_path if scored else failures_path
        with write_lock:
            with open(path, "a") as handle_file:
                handle_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle_file.flush()
                os.fsync(handle_file.fileno())
            (failures if not scored else records).append(record)
            if not record.get("ok"):
                aborted.set()
            completed += 1
            if completed % args.summary_interval == 0 or not record.get("ok"):
                facts["gpu_during"].append({"utc": time.strftime("%H:%M:%S", time.gmtime()), **gpu_facts()})
                summarize(out, records, failures, len(items), time.monotonic() - started,
                          counters_before, metric_counters(args.base_url), args.isolated)
        print(json.dumps({"event": "item", "index": record["index"], "item_id": record["item_id"],
                          "status": record["status"], "correct": record["correct"],
                          "finish": record.get("finish_reason"),
                          "completion_tokens": record.get("completion_tokens"),
                          "ttft_s": record.get("ttft_s"), "decode_tps": record.get("decode_tps")}), flush=True)
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.streams) as pool:
        futures = [pool.submit(handle, item) for item in pending]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    wall = time.monotonic() - started
    summary = summarize(out, records, failures, len(items), wall, counters_before,
                        metric_counters(args.base_url), args.isolated)
    facts["gpu_final"] = gpu_facts()
    run_config["server_facts"] = facts
    run_config["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run_config["counters_after"] = metric_counters(args.base_url)
    (out / "run.json").write_text(json.dumps(run_config, indent=2) + "\n")
    print(json.dumps({"event": "done", "denominator": summary["denominator"], "correct": summary["correct"],
                      "accuracy_strict_denominator": summary["accuracy_strict_denominator"],
                      "statuses": summary["statuses"], "wall_s": round(wall, 1)}), flush=True)
    if not summary["valid_quality_result"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
