#!/usr/bin/env python3
"""S4/S5: serve a candidate on a private port and score it: KL against the teacher, then speed.

KL estimator (§8), identical for every model: over the union of teacher and student top-64.
A token missing on one side was below that side's 64th token, so it is imputed at half that
probability, capped by the tail mass. One binary term covers the mass outside the union.
Own traffic scores assistant turns only; the public traces use their own scored spans.
"""
import argparse
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file
from transformers import AutoTokenizer

from requant_common import LIVE, ROOT, Job
import sys

PORT = 8199
API = f"http://127.0.0.1:{PORT}"
SETS = ROOT / "data/sets"
TEACHER = ROOT / "s2/teacher"
KEY = os.environ.get("OPENAI_API_KEY", "")
if os.environ.get("OPENAI_API_KEY_FILE"):
    KEY = Path(os.environ["OPENAI_API_KEY_FILE"]).read_text().strip()
SERVED = "qwen3.8-flash-next"
PYTHONPATH = None




def start_server(job, model, speculative, extra, log_name):
    argv = [sys.executable, str(Path(__file__).with_name("serve.py")),
            "--model", str(model), "--host", "127.0.0.1", "--port", str(PORT)]
    if not speculative:
        argv.append("--no-mtp")
    if extra:
        argv += ["--", *extra]
    environment = {**os.environ, "VLLM_API_KEY": KEY}
    if PYTHONPATH:
        environment["PYTHONPATH"] = PYTHONPATH
    log_file = open(job.dir / log_name, "a")
    process = subprocess.Popen(argv, env=environment, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
    started = time.time()
    while time.time() - started < 900:
        if process.poll() is not None:
            raise SystemExit(f"candidate server exited {process.returncode}; see {log_name}")
        if job.stop:
            stop_server(process)
            raise SystemExit(0)
        try:
            with urllib.request.urlopen(API + "/health", timeout=5) as response:
                if response.status == 200:
                    job.log(f"server up in {time.time() - started:.0f}s ({log_name})")
                    return process
        except OSError:
            pass
        time.sleep(3)
    stop_server(process)
    raise SystemExit("candidate server never became healthy")


def stop_server(process):
    with __import__("contextlib").suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def post(path, body, timeout=3600):
    request = urllib.request.Request(API + path, data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)




def kl_position(t_ids, t_logp, t_tail, student):
    """student: {token: logprob}. Returns (KL, whether the top-1 tokens agree)."""
    s_tail = max(1e-9, 1.0 - sum(math.exp(v) for v in student.values()))
    teacher = dict(zip(t_ids.tolist(), t_logp.tolist()))
    p_tail = 0.5 * min(max(float(t_tail), 1e-9), math.exp(min(teacher.values())))
    q_tail = 0.5 * min(s_tail, math.exp(min(student.values())))
    kl, p_sum, q_sum = 0.0, 0.0, 0.0
    for token in teacher.keys() | student.keys():
        p = math.exp(teacher[token]) if token in teacher else p_tail
        q = math.exp(student[token]) if token in student else q_tail
        kl += p * (math.log(p) - math.log(max(q, 1e-12)))
        p_sum += p
        q_sum += q
    p_rest, q_rest = max(1.0 - p_sum, 1e-9), max(1.0 - q_sum, 1e-9)
    kl += p_rest * math.log(p_rest / q_rest)
    return max(kl, 0.0), int(t_ids[0]) == max(student, key=student.get)


def score_domain(job, domain, out):
    tokenizer = AutoTokenizer.from_pretrained(LIVE)
    excluded = set(tokenizer.all_special_ids) | set(tokenizer.convert_tokens_to_ids(["<think>", "</think>"]))
    im_start, im_end = tokenizer.convert_tokens_to_ids("<|im_start|>"), tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant = tokenizer.encode("assistant", add_special_tokens=False)[0]
    call_open, call_close = tokenizer.convert_tokens_to_ids("<tool_call>"), tokenizer.convert_tokens_to_ids("</tool_call>")
    with safe_open(SETS / f"{domain}.safetensors", framework="np") as f:
        ids, offsets = f.get_tensor("input_ids"), f.get_tensor("offsets")
    # The S2 job numbers sequences across all sets; the teacher file holds this set's own.
    teacher = load_file(TEACHER / f"{domain}.safetensors")
    first_global = int(teacher["sequence"].min())
    results = json.loads(out.read_text()) if out.exists() else {"sequences": {}}
    for global_index in np.unique(teacher["sequence"]).tolist():
        key = str(global_index)
        if key in results["sequences"]:
            continue
        if job.stop:
            return None
        local = global_index - first_global
        sequence = ids[offsets[local]:offsets[local + 1]].tolist()
        rows = np.flatnonzero(teacher["sequence"] == global_index)
        response = post("/v1/completions", {"model": SERVED, "prompt": sequence, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 64})
        student = response["choices"][0]["prompt_logprobs"]
        kls, agreements, tool_kls = [], [], []
        inside_call = np.cumsum((np.array(sequence) == call_open).astype(int) - (np.array(sequence) == call_close).astype(int)) > 0
        in_assistant = np.zeros(len(sequence), dtype=bool)
        speaking = False
        for position, token in enumerate(sequence):
            if token == im_end:
                speaking = False
            elif token == assistant and position > 0 and sequence[position - 1] == im_start:
                speaking = True
            in_assistant[position] = speaking
        for row in rows.tolist():
            target = int(teacher["position"][row]) + 1
            if sequence[target] in excluded or student[target] is None:
                continue
            if domain == "eval_private" and not in_assistant[target]:
                continue
            entry = {int(k): float(v["logprob"]) for k, v in student[target].items()}
            kl, agree = kl_position(teacher["ids"][row], teacher["logprobs"][row].astype(np.float64), teacher["tail"][row], entry)
            kls.append(kl)
            agreements.append(agree)
            if inside_call[target]:
                tool_kls.append(kl)
        results["sequences"][key] = {"tokens": len(kls), "kl": float(np.sum(kls)), "top1": int(np.sum(agreements)), "tool_tokens": len(tool_kls), "tool_kl": float(np.sum(tool_kls))}
        out.write_text(json.dumps(results))
        job.log(f"{domain} seq {key}: {len(kls)} scored, mean KL {np.mean(kls) if kls else 0:.4f}")
    values = results["sequences"].values()
    tokens = sum(v["tokens"] for v in values)
    results["mean_kl"] = sum(v["kl"] for v in values) / max(tokens, 1)
    results["top1"] = sum(v["top1"] for v in values) / max(tokens, 1)
    tool_tokens = sum(v["tool_tokens"] for v in values)
    results["tool_kl"] = sum(v["tool_kl"] for v in values) / max(tool_tokens, 1)
    results["tokens"] = tokens
    rng = np.random.default_rng(3821)
    rows = list(values)
    samples = []
    for _ in range(2000):
        picked = rng.integers(0, len(rows), len(rows))
        samples.append(sum(rows[i]["kl"] for i in picked) / max(sum(rows[i]["tokens"] for i in picked), 1))
    results["kl_ci95"] = [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]
    out.write_text(json.dumps(results))
    job.log(f"{domain}: KL {results['mean_kl']:.4f} {results['kl_ci95']} top1 {results['top1']:.4f} tool {results['tool_kl']:.4f}")
    return results




def speed(job, out, rounds_1=4, rounds_4=2):
    with safe_open(SETS / "eval_public.safetensors", framework="np") as f:
        ids, offsets = f.get_tensor("input_ids"), f.get_tensor("offsets")
    prompts = job.dir / "speed-prompts.jsonl"
    prompts.write_text("".join(json.dumps({"prompt": ids[offsets[i]:offsets[i + 1]][:4000].tolist()}) + "\n"
                               for i in range(min(8, len(offsets) - 1))))
    subprocess.run([sys.executable, str(Path(__file__).with_name("benchmark.py")),
                    "--base-url", API, "--model", SERVED, "--prompts", str(prompts),
                    "--rounds", str(rounds_1), "--output", str(out)], check=True)
    return json.loads(out.read_text())["phases"]


def startup_facts(job, log_name):
    facts = {}
    for line in (job.dir / log_name).read_text(errors="ignore").splitlines():
        for marker in ("GPU KV cache size", "Model loading took", "Available KV cache memory", "scaling factor"):
            if marker in line:
                facts[marker] = line.split("]", 1)[-1].strip()
    return facts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--domains", default="eval_public,eval_nemotron,eval_multilingual")
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--speed-rounds", type=int, default=4)
    parser.add_argument("--extra", default="", help="extra vLLM arguments, shell-quoted")
    parser.add_argument("--pythonpath", default=None)
    args = parser.parse_args()
    global PYTHONPATH
    PYTHONPATH = args.pythonpath
    job = Job(f"s4-{args.candidate}")
    extra = shlex.split(args.extra)
    summary = {"candidate": args.candidate}
    domains = [d for d in args.domains.split(",") if d]
    pending = [d for d in domains if not (job.dir / f"done-{d}").exists()]
    if pending:
        server = start_server(job, args.model, False, extra, "server-kl.log")
        try:
            for domain in pending:
                result = score_domain(job, domain, job.dir / f"{domain}.json")
                if result is None:
                    job.log("stopped cleanly")
                    return
                (job.dir / f"done-{domain}").touch()
        finally:
            stop_server(server)
    if domains:
        summary["kl"] = {}
    for domain in domains:
        data = json.loads((job.dir / f"{domain}.json").read_text())
        summary["kl"][domain] = {k: data[k] for k in ("mean_kl", "kl_ci95", "top1", "tool_kl", "tokens")}
    if not args.skip_speed:
        if not (job.dir / "speed.json").exists():
            server = start_server(job, args.model, True, extra, "server-speed.log")
            try:
                if speed(job, job.dir / "speed.json", args.speed_rounds, args.speed_rounds // 2) is None:
                    job.log("stopped cleanly")
                    return
            finally:
                stop_server(server)
        summary["speed"] = json.loads((job.dir / "speed.json").read_text())
        summary["startup"] = startup_facts(job, "server-speed.log")
    (ROOT / "eval").mkdir(exist_ok=True)
    (ROOT / "eval" / f"{args.candidate}.json").write_text(json.dumps(summary, indent=1))
    job.done(summary)


if __name__ == "__main__":
    main()
