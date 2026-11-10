#!/usr/bin/env python3
"""Launch pinned vLLM without hardware controls or host-specific services."""
import argparse
import json
import os
import sys
from pathlib import Path


def command(args):
    model = args.model.expanduser().resolve()
    runtime = Path(os.environ["T3_RUNTIME"])
    if not (model / "t3layers.json").is_file():
        raise ValueError(f"missing T3 metadata in {model}")
    index = json.loads((model / "model.safetensors.index.json").read_text())
    missing = sorted({s for s in index["weight_map"].values() if not (model / s).is_file()})
    if missing:
        raise ValueError(f"incomplete checkpoint; missing shards: {missing}")
    os.environ["KUDA_T3MOE"] = str(model)
    os.environ["HF_HUB_OFFLINE"] = "1"
    if os.environ.get("CREDENTIALS_DIRECTORY"):
        os.environ["VLLM_API_KEY"] = (Path(os.environ["CREDENTIALS_DIRECTORY"]) / "api-key").read_text().strip()
    compilation = {"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [*range(1, 17), 24, 32]}
    additional = {"ple_ssd_offload": True, "ple_ssd_workers": args.ssd_workers, "ple_ssd_cache_mb": 512,
                  "ple_ssd_native_library": str(runtime / "optimization/ple_ssd_io.so"), "ple_ssd_io_depth": 256, "ple_ssd_prefetch_tokens": 16384}
    speculation = {"method": "mtp", "num_speculative_tokens": 3, "draft_sample_method": "probabilistic", "disable_eagle_block_drop": True}
    argv = [str(runtime / ".venv/bin/vllm"), "serve", str(model), "--served-model-name", "qwen3.8-flash-next",
            "--host", args.host, "--port", str(args.port), "--dtype", "bfloat16", "--language-model-only",
            "--max-model-len", str(args.max_model_len), "--max-num-seqs", str(args.max_num_seqs),
            "--kv-cache-memory-bytes", str(int(args.kv_gib * 2**30)), "--attention-config.indexer_kv_dtype", "fp8",
            "--max-num-batched-tokens", "2048", "--gpu-memory-utilization", "0.96", "--kv-cache-dtype", "fp8",
            "--safetensors-load-strategy", "lazy", "--mamba-cache-mode", "align", "--enable-prefix-caching",
            "--enable-prompt-tokens-details", "--prefix-match-unit", "16", "--enable-chunked-prefill",
            "--reasoning-parser", "qwen3", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml",
            "--chat-template", os.environ["T3_CHAT_TEMPLATE"],
            "--compilation-config", json.dumps(compilation), "--additional-config", json.dumps(additional)]
    if args.no_mtp:
        os.environ["KUDA_T3MTP"] = "0"
        argv += ["--max-logprobs", "64"]
    else:
        argv += ["--speculative-config", json.dumps(speculation)]
    return argv + args.extra


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8099)
    p.add_argument("--kv-gib", type=float, default=14.6)
    p.add_argument("--max-model-len", type=int, default=262144)
    p.add_argument("--max-num-seqs", type=int, default=8)
    p.add_argument("--ssd-workers", type=int, default=16)
    p.add_argument("--no-mtp", action="store_true")
    p.add_argument("--print-command", action="store_true")
    p.add_argument("extra", nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.extra[:1] == ["--"]:
        a.extra = a.extra[1:]
    if a.kv_gib <= 0 or not 1 <= a.max_num_seqs <= 8:
        p.error("positive KV budget and 1..8 sequences required by the captured profile")
    try:
        argv = command(a)
    except ValueError as e:
        p.error(str(e))
    if a.print_command:
        print(json.dumps(argv))
    else:
        os.execv(argv[0], argv)


if __name__ == "__main__":
    main()
