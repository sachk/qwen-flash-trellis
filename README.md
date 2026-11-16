# Qwen Flash trellis

> **Quality update:** Further testing has been disappointing: this release scored **75.76% on GPQA Diamond**, with substantial token-limit failures and repetitive output. Earlier KL checks overstated our confidence in reasoning quality. A replacement is planned soon; do not treat this release as near-original quality. Weight-compression and runtime/sampling causes are still being separated.

Replacement research for independent review: [method comparison](QUANT-METHOD-REVIEW.md) and [fixed-budget SM80 testbed plan](SM80-REQUANT-PLAN.md). These are proposals, not a validated replacement.

Standalone quantisation tools and pinned Nix/vLLM runtime for Qwen3.8-Flash-Next T3a392. The production model retains all 512 routed experts per layer using mixed 2/3/4-bit ring-coded trellis packets. Public weights: [sachk0/Qwen3.8-Flash-Next-T3a392-Ampere-SM80-46GiB-GPU-Weights](https://huggingface.co/sachk0/Qwen3.8-Flash-Next-T3a392-Ampere-SM80-46GiB-GPU-Weights).

This public repository contains no GPU unlock/overclock tools, firmware, clock/power controls, home-network services, credentials, private conversations or calibration token dumps. SM80 is an **architecture requirement**, not a particular card: the CUDA kernels target Ampere GA100. Other architectures are not validated. Launch occupancy uses device properties, not a fixed SM count. Legacy `KUDA_*` environment names are retained as loader API names only; they do not control hardware.

## Requirements

- Linux x86_64 with Nix flakes and an existing NVIDIA driver supporting CUDA 13; the module does not install or configure a driver.
- A 64 GiB-class SM80 GPU for the measured configuration; ~46.16 GiB loaded model weights **plus** 14.6 GiB KV, graphs and runtime overhead.
- Fast local SSD for the ~95.37 GiB BF16 PLE table. The complete checkpoint is **155,110,650,462 bytes (~144.46 GiB)** including metadata, with 62 safetensors files. Allow additional disk space for the runtime/download caches.
- Quantising again needs substantially more temporary disk and GPU memory than serving. The original calibration path peaked around 56 GiB GPU memory and writes large layer libraries. Never run calibration alongside a GPU-resident server.

## Run with Nix

```sh
export T3_WORK_DIR=/path/on/ssd/qwen-trellis
export T3_RUNTIME="$T3_WORK_DIR/runtime"
nix develop
# First installation downloads pinned Python/CUDA dependencies; no weights are in Git/Nix.
t3 setup
t3 download sachk0/Qwen3.8-Flash-Next-T3a392-Ampere-SM80-46GiB-GPU-Weights
t3 serve --model "$T3_WORK_DIR/model" --kv-gib 14.6
```

`nix run . -- setup`, `nix run . -- download ...` and `nix run . -- serve ...` are equivalent. A change to pinned runtime sources requires a fresh `T3_RUNTIME` directory; setup deliberately refuses to delete an existing runtime. Keep `T3_WORK_DIR`/`T3_RUNTIME` consistent between commands. On non-NixOS Linux, provide a working NVIDIA driver library path through `LD_LIBRARY_PATH` and `TRITON_LIBCUDA_PATH`; only the NixOS driver layout is used by default.

The API defaults to **127.0.0.1:8099**, served model name `qwen3.8-flash-next`. Set `VLLM_API_KEY` for authenticated access. Do not expose an unauthenticated listener to an untrusted network. No daemon, power-controller socket, gateway, scheduler or production service is touched by CLI commands. `t3 serve --help` lists the memory/context/sequence options. The native per-request limit is 262,144 tokens; the measured 906,051-token figure is **aggregate KV capacity**.

The pinned runtime is vLLM `a5a30471ff2bb7f0824f2da10e358af98d304472`, with generic PLE-SSD support, software FP8 for Ampere, INT8 head/shared/embedding paths, fused HC and exact top-k/top-p verify mask. Trellis MTP depth 3 uses the included 65,536-token draft vocabulary. Expert scratch is sized to the scheduler's actual 2,048-token maximum and graph capture stops at 32 rows. The bounded prefix-cache copy fix avoids the original whole-block gather temporary while preserving overlap semantics.

`patches/runtime-lock.txt` pins all 205 non-editable Python dependencies from a verified installation; the wheel/source revision pins vLLM itself. `patches/tested-constraints.txt` preserves the upstream core-version record. Dependency-lock changes are part of the runtime stamp. Maintainers can regenerate a snapshot with `t3 python scripts/lock_runtime.py --out patches/runtime-lock.txt` after verifying an updated environment.

Only generic architecture/model support is installed: no fixed 70/74-SM tuning profile. The optional `KUDA_QSA_TABLE` launch-table hook is present but unset by default. Historical performance depends on hardware and launch configuration; the portable untuned profile has not been measured on every card.

### NixOS module

Add this flake as an input, then:

```nix
{
  imports = [ inputs.qwen-flash-trellis.nixosModules.default ];
  services.qwen-flash-trellis = {
    enable = true;
    modelPath = "/srv/models/qwen-trellis";
    stateDirectory = "/srv/qwen-runtime";
    kvGiB = 14.6;
    apiKeyFile = "/run/secrets/qwen-api-key";
  };
}
```

The service user defaults to `qwen-trellis`. Make the model directory readable/traversable by that user; keep model files read-only. The state directory must have room for the runtime and caches. API credentials are loaded at runtime with systemd credentials, not embedded in the Nix store. Changing the module can restart its **own** service; do not enable it beside an already running server on the same GPU. No card-specific services or hardware settings are imported.

Build on a suitable remote builder rather than a busy desktop, for example evaluate the `.drv` with `nix eval --raw .#packages.x86_64-linux.default.drvPath`, then pass that derivation's `^out` to `nix build --store ssh-ng://YOUR-BUILDER --eval-store auto --no-link`.

## Benchmarks and quality

Full records, both paired baselines and both deployed runs: [`benchmarks/results.json`](benchmarks/results.json). The [model card](MODEL_CARD.md) contains the performance and quality tables. Historical production measurements on a 74-SM/64-GiB GA100 card used non-stock hardware configuration, deliberately not reproduced by this repo:

| Metric | Measured result |
|---|---:|
| Single-stream decode | 185.27–189.25 tok/s |
| Four-stream total decode | 375.48–383.04 tok/s |
| Single-stream prefill | 3,238–3,265 tok/s |
| One-stream step time | 14.413–14.429 ms |
| Four-stream step time | 28.139–28.280 ms |
| Pooled KV capacity at 14.6 GiB | 906,051 tokens |

These are real replay measurements with prefill, 72/36 rounds at 1/4 streams, 400 generated tokens, temperature 1/top-k 20/top-p .95. They are not directly comparable to endless synthetic completion benchmarks. Tokens/s moves with draft acceptance; compare step time first. The final update was a **memory/reliability win, not a demonstrated decode speedup**. The 250/500 tok/s goals were not reached.

Teacher-forced server KL against the FP32 source teacher was 0.0924 own/private replay, 0.0391 Nemotron, 0.0365 multilingual, 0.0312 medical, 0.0271 legal. This is distribution divergence, **not a task-accuracy or preference benchmark**. The memory update moved each domain by at most 0.0007. Private replay data are not distributed, so outsiders cannot reproduce those exact private-domain numbers. Stress qualification passed two simultaneous 250k-token contexts followed by eight concurrent short streams; peak GPU memory was 64,710 MiB. This does not measure long-context reasoning quality.

### Benchmark your already-running server

```sh
export OPENAI_API_KEY_FILE=/path/to/api-key  # omit for a trusted unauthenticated loopback server
t3 benchmark --prompts benchmarks/prompts.jsonl --rounds 72 --output replay.json
```

For comparable results provide representative long prompts as JSONL rows with `prompt` strings or token-ID arrays. The included short public prompts are only a smoke/example input, not the original benchmark corpus. Fresh salt prevents identical-prefix cache hits. The tool uses actual completion-token usage rather than counting SSE chunks, and propagates request failures. Run against an otherwise idle server: draft counters are server-global, so unrelated traffic invalidates acceptance/step-time estimates. The tool never restarts serving.

## Requantise from source

The public-only pipeline reproduces the **procedure**, not the original private calibration statistics or exact released bytes. Keep all work on a large SSD. First `t3 setup`, then:

```sh
export T3_WORK_DIR=/path/on/ssd/requant
export T3_BASE_MODEL="$T3_WORK_DIR/base"
export T3_FP8_SOURCE="$T3_WORK_DIR/source"
# With the same T3_RUNTIME used for setup:
t3 python scripts/s0_download.py
# Fetch the pinned dense/PLE base using the runtime's hf executable:
t3 python -m huggingface_hub.cli.hf download klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP --revision ce0e0b94083895bd836b916f29bf105c40a8162a --local-dir "$T3_BASE_MODEL"
t3 python scripts/s0_build_sets.py
t3 python t3/s0_domains.py
t3 python scripts/s2_pass.py --edge-bits 3 --middle-bits 3
t3 python scripts/s3_head.py
t3 python scripts/s3_assemble.py --name R --experts 512 --edge gptq
t3 python t3/t3_pass.py --name t3pass2
t3 python t3/t3_alloc.py --lib "$T3_WORK_DIR/t3pass2" --out "$T3_WORK_DIR/t4a-392" --gib 39.2 --layer-weights t3/layer_weights.json
t3 python t3/mtp_round.py "$T3_WORK_DIR/mtp-t3" 3 3
t3 python t3/t3_assemble.py --base "$T3_WORK_DIR/cand/R" --lib "$T3_WORK_DIR/t4a-392" --out "$T3_WORK_DIR/model" --mtp "$T3_WORK_DIR/mtp-t3"
```

Run from the checkout when passing relative script paths. `s0_download.py` fetches the pinned FP8 source excluding PLE/vision-only shards and public Vemloir agent/chat/evaluation datasets. `s0_build_sets.py` creates source/session-disjoint `calib`, `calib2`, `eval_public` plus the public evaluation traces. Supplemental medical/legal documents are fetched by `s0_domains.py` with document-disjoint calibration/evaluation and source licensing filters. Network-sourced document collections can change over time; retain your own manifests for reproducibility. Dataset licenses remain their owners' terms.

The public-only primary calibration budget reserves 15% for agent traces and 85% for chat traces; secondary/public evaluation budgets are split equally between those sources. Per-source caps prevent the first dataset from consuming the entire budget. Actual collected counts are printed; private replay is not silently substituted into a public evaluation.

`--resume` is supported by the long layer passes. The supplied layer-allocation weights and draft vocabulary are the released production artifacts; recomputing them on different calibration/evaluation changes the result. No experimental gain-fitting, lattice or alternate decoder candidates are used by the serving runtime.

### Verification and quality tools

Only run GPU checks with the server stopped and GPU resources available:

```sh
t3 python t3/t3_check.py
t3 python t3/topk_check.py --no-bench
t3 python t3/hc_check.py --checkpoint "$T3_WORK_DIR/model"
t3 python t3/t3_roundtrip.py
```

`sim_eval.py` provides offline teacher-forced KL, including precise teacher generation (`--precise --save-teacher`). `scripts/s4_eval.py` provides real-server teacher-forced KL against `s2/teacher`, launching and stopping **only its own candidate subprocess** on port 8199; use `--skip-speed` and the standalone live benchmark for timings. It never reads host service definitions or root credential paths. Public re-evaluation defaults to `eval_public`, not the original private replay domain. Do not conflate simulator and server KL figures.

## Release verification

The portable Nix runtime, all three CUDA libraries and generated vLLM overlay built on a CPU-only build server. Fresh setup also succeeded with the complete dependency lock; imports reported vLLM `0.29.1rc1.dev402+ga5a30471f.ple1`, Torch `2.13.0+cu130`, Transformers `5.17.0`. The NixOS module evaluated with the intended 14.6 GiB KV argument.

The public-data CLI regression first failed on source starvation, then passed after per-source budget accounting was fixed. It checks saved source/session separation, token budgets, original token identity and held-out score spans. Run it with `t3 python -m unittest discover -s tests -v`.

A real 16-token streamed API request exercised the benchmark client against the already-running production server without restarting it. This is a client smoke, not a performance measurement. Launch-command generation checked the actual complete checkpoint index and shard presence. Python compilation and shell syntax checks passed. No independent reviewer was used.

No new GPU server or full replay benchmark was launched during packaging, so production could remain available. Original GPU qualification is documented above; fresh GPU qualification of this portable untuned runtime and other cards remains unperformed.

## Files and provenance

- `nix/`: pinned runtime, CUDA toolkit layout, kernel builds, overlay and opt-in NixOS module.
- `t3/`: quantiser, packet format, allocator, assembler, MTP and arithmetic checkers.
- `scripts/`: checkpoint streaming, source preparation, base/head construction, evaluation, serving and uploads.
- `patches/`: only the generic PLE-SSD patch, tested dependency constraints and architecture support patches; upstream repository hardware utilities are neither vendored nor fetched.
- `benchmarks/`: measured data and public example prompts, never private replay text.

See [NOTICE.md](NOTICE.md) for upstream provenance and licensing. Model weights retain Qwen Community License 1.0, not an Apache/MIT license. No credentials or weights belong in this Git repo.
