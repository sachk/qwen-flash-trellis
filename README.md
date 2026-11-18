# Qwen Flash trellis

Requantisation research, now an efficiency/headroom project rather than a quality fix: [method comparison](QUANT-METHOD-REVIEW.md) and [fixed-budget SM80 testbed plan](SM80-REQUANT-PLAN.md). These are proposals, not a validated replacement.

Standalone quantisation tools and pinned Nix/vLLM runtime for Qwen3.8-Flash-Next T3a392. The production model retains all 512 routed experts per layer using mixed 2/3/4-bit ring-coded trellis packets. Public weights: [sachk0/Qwen3.8-Flash-Next-T3a392-Ampere-SM80-46GiB-GPU-Weights](https://huggingface.co/sachk0/Qwen3.8-Flash-Next-T3a392-Ampere-SM80-46GiB-GPU-Weights).

This public repository contains no GPU unlock/overclock tools, firmware, clock/power controls, home-network services, credentials, private conversations or calibration token dumps. SM80 is an **architecture requirement**, not a particular card: the CUDA kernels target Ampere GA100. Other architectures are not validated. Launch occupancy uses device properties, not a fixed SM count. Legacy `KUDA_*` environment names are retained as loader API names only; they do not control hardware.

## Benchmarks and quality

Measured 2026-11-18 on a 64 GiB CMP 170HX (GA100) in a stable configuration: SM clock capped at 1400 MHz, stock memory clock. **This card is not yet fully SM- or memory-bus-unlocked** (74 SMs enabled; the memory bus is not fully unlocked), so these are not full-GA100 figures. Hardware settings are deliberately not reproduced by this repo. Record: [`benchmarks/speed-20261118.json`](benchmarks/speed-20261118.json) (withdrawn 2026-11-09 records: [`benchmarks/results.json`](benchmarks/results.json)); the [model card](MODEL_CARD.md) carries the same tables.

| Metric | Measured result |
|---|---:|
| GPU-resident model weights | 46.16 GiB |
| GPU KV pool | 14.6 GiB |
| Single-stream decode | 166.5 tok/s |
| Four-stream total decode | 352.6 tok/s |
| Single-stream prefill | 3,073 tok/s |
| One-stream step time | 16.29 ms |
| Four-stream step time | 30.64 ms |
| Accepted draft tokens per step (1 / 4 streams) | 1.71 / 1.70 |
| Aggregate KV capacity on a 64 GiB GPU | 906,051 tokens |

Real replay with prefill: eight 4,000-token private-traffic prompts, 72/36 rounds at 1/4 streams, 400 generated tokens, temperature 1/top-k 20/top-p .95, production serving configuration. Not directly comparable to endless synthetic completion benchmarks. Tokens/s moves with draft acceptance, so compare step time first. Figures published earlier (2026-11-09) were measured on a GPU configuration later found unstable and are withdrawn.

KV capacity is pooled across requests; the native single-request limit is **262,144 tokens**.

### GPQA Diamond: task score and token use

Measured 2026-11-18 on stock GPU settings: one sample per question, xhigh reasoning, temperature 1, top-k 20, top-p 0.95, seed 20261114 + question index, eight concurrent streams, production serving configuration. [Measurement record with per-question outcomes](benchmarks/gpqa-diamond-20261118.json).

| Metric | 32,768-token cap | 98,304-token cap | Original-model published reference |
|---|---:|---:|---:|
| GPQA Diamond accuracy | **159/198 (80.30%)** | **178/198 (89.90%)** | **91.7%** |
| Requests reaching the output cap | 32/198 (16.2%) | 1/198 (0.5%) | Not published |
| Generated tokens, including reasoning | 2,474,928 | 2,988,418 | Not published |
| Mean / median generated tokens per question | 12,500 / 8,673 | 15,093 / 8,673 | Not published |
| NaN/collapsed responses | 0 | 0 | — |

The 98,304-token column reruns only the 32 capped questions with a larger cap and identical seeds. Every other question finished below 32,768 tokens, so a larger cap would not change it; the column therefore equals a 98k-cap run up to batch-level nondeterminism. Capped answers count as incorrect. The [official 91.7% reference](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/812d96e10a0dd82b69d749e68693698cf6c36daa/README.md) does not publish its prompt, sampling, aggregation or token budget, so the remaining gap is **not a measured quantisation penalty**, and one sample per question carries roughly ±2–3 points of sampling noise.

**Compared with the AutoRound base.** T3a392 reuses the dense layers of [`klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP`](https://huggingface.co/klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP) and replaces its 3-bpw routed experts. On the same own-traffic teacher-forced evaluation its KL is **0.090 versus 0.168 for AutoRound**, a little over half the divergence. Its last production speed figures on this card were also lower (2026-10-30, same vLLM stack, same stable 1400 MHz clock): 127.35 tok/s single-request and 316.13 tok/s four-request aggregate, on ~8K-token code prompts with 1,024 generated tokens and time to first token included. That is a different workload from the replay table above, so it is not a matched speed ratio. On the matched replay benchmark at the same clock, the first T3 build equalled AutoRound's speed (152.7 tok/s single-stream) with the lower KL. Later runtime work (64K draft vocabulary, expert-kernel tuning, verify-mask kernel) raised T3a392 to the figures above.

**Superseded 2026-11-14 run** ([record](benchmarks/gpqa-diamond-20261114.json)): 150/198 (75.76%) at 32,768 tokens with 42 capped, two streams. That machine's non-stock GPU configuration produced rare all-NaN logit steps (about 24 per hour of long-reasoning decode). Each affected request then emitted token ID 0 until the cap, which appeared as "repetitive output" and token-limit failures. On stock settings the identical software produced no NaN steps in about 3.5 hours of generation, and a model-free bit-exact GPU stress replay passed. Of the 16 questions that collapsed on 11-14, 10 are correct at 32k and 13 at 98k.

### Teacher-forced fidelity

Teacher-forced server KL against the FP32 source teacher was 0.0924 own/private replay, 0.0391 Nemotron, 0.0365 multilingual, 0.0312 medical, 0.0271 legal. This is distribution divergence, **not a task-accuracy or preference benchmark**. The memory update moved each domain by at most 0.0007. Private replay data are not distributed, so outsiders cannot reproduce those exact private-domain numbers. Stress qualification passed two simultaneous 250k-token contexts followed by eight concurrent short streams; peak GPU memory was 64,710 MiB. This does not measure long-context reasoning quality.

## Method and references

The released checkpoint stores routed-expert weights as **ring-coded trellis packets**: mixed 2/3/4-bit payloads driven through a custom 4-state trellis code. Each ring is a tail-biting loop through a unit's packet words, so the path closes on itself and no start/end boundary states cost anything per block. Compared with scalar quantisation, the trellis constrains successive reconstruction levels, giving richer reconstruction per transmitted bit. Packing solves each ring with **Hessian-aware Viterbi rounding**: per-expert second-order costs feed an exact tail-biting min-sum path search, colour-swept with sequential error feedback, in the spirit of GPTQ's second-order objective and error compensation. Serving decode is bit-exact and uses tensor-core-friendly packet layouts; that is a design property, not a speed guarantee on any GPU.

The format and solver are custom to this repository, not a stock QTIP deployment: they use neither QTIP's randomized-Hadamard incoherence processing (RHT) nor its 1MAD/3INST decode settings, and no QuIP#/GLQ lattice codebooks are involved.

References:

- A. Tseng, Q. Sun, D. Hou, C. De Sa. *QTIP: Quantization with Trellises and Incoherence Processing.* NeurIPS 2024. [arXiv:2406.11235](https://arxiv.org/abs/2406.11235) — hardware-efficient parallel trellis structures, lookup/computed codes, and tail-biting (§3.2).
- E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) — second-order rounding objective and error compensation.
- M. W. Marcellin, T. R. Fischer. *Trellis Coded Quantization of Memoryless and Gauss-Markov Sources.* IEEE Transactions on Communications 38(1):82–93, 1990. [doi:10.1109/26.46532](https://doi.org/10.1109/26.46532) — original TCQ.

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

### Benchmark your already-running server

```sh
export OPENAI_API_KEY_FILE=/path/to/api-key  # omit for a trusted unauthenticated loopback server
t3 benchmark --prompts benchmarks/prompts.jsonl --rounds 72 --output replay.json
```

For comparable results provide representative long prompts as JSONL rows with `prompt` strings or token-ID arrays. The included short public prompts are only a smoke/example input, not the original benchmark corpus. Fresh salt prevents identical-prefix cache hits. The tool uses actual completion-token usage rather than counting SSE chunks, and propagates request failures. Run against an otherwise idle server: draft counters are server-global, so unrelated traffic invalidates acceptance/step-time estimates. The tool never restarts serving.

### GPQA Diamond quality and token accounting

[`scripts/gpqa_eval.py`](scripts/gpqa_eval.py) evaluates the pinned 198-question Diamond CSV with normal EOS, deterministic choice shuffling, one sample per question and final-answer-only scoring. It records raw answers/reasoning, API-reported token usage, truncations, failures and per-request/aggregate timing. Download the CSV from [OpenAI's GPQA Diamond distribution](https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv); the runner checks its SHA-256.

```sh
t3 python scripts/gpqa_eval.py --csv /path/to/gpqa_diamond.csv \
  --out local-docs/gpqa/run --base-url http://127.0.0.1:8099 \
  --streams 2 --max-tokens 32768 --reasoning-effort xhigh
```

These are local evaluation settings, not a reproduction of Qwen's unpublished protocol. A cited original-model score does not provide matched reasoning-token usage. `local-docs/` is gitignored for local source snapshots and evaluation artifacts. For clean speed measurements, isolate the backend from other clients; two-stream timing is not the historical single-stream benchmark. Scoring contract checks: `python3 -m unittest discover -s tests -p 'test_gpqa_eval.py' -v`.

Incomplete streams or transport failures invalidate the quality result and stop new dispatch; the client exits nonzero rather than presenting an infrastructure outage as model accuracy. Completed token-limit responses without an answer remain incorrect. `--skip-existing` preserves completed attempts when resuming; recovery-phase speed excludes previously generated tokens.

### Small BF16-versus-FP8 source pilot

[`scripts/source_precision_probe.py`](scripts/source_precision_probe.py) downloads only safetensors headers and selected tensor byte ranges, then compares source weights and re-rounds one T4 expert with the real C2/Babai solver. It runs with ordinary CUDA Torch, NumPy and HTTPX on an RTX 5070; no SM80 serving kernels or full model are required.

```sh
python scripts/source_precision_probe.py --out local-docs/source-precision --device cuda
```

On NixOS, expose the driver with `LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH`. This is a weight-space pilot with an identity Hessian, not task accuracy or a calibrated full-model quality result; its W8 diagnostic is symmetric RTN rather than AutoRound.


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

On 2026-11-14, the standalone Nix package, all three CUDA libraries and generated vLLM overlay built on the CMP 170HX host while production remained available. `t3 setup` installed a fresh, isolated runtime with the complete dependency lock: vLLM `0.29.1rc1.dev402+ga5a30471f.ple1`, Torch `2.13.0+cu130`, Transformers `5.17.0`. Production's runtime and overlays were not reused. The NixOS module was previously evaluated with the intended 14.6 GiB KV argument.

The public-data CLI regression first failed on source starvation, then passed after per-source budget accounting was fixed. It checks saved source/session separation, token budgets, original token identity and held-out score spans. Run it with `t3 python -m unittest discover -s tests -v`.

With production temporarily stopped, the repo-built `t3 serve` launched the existing checkpoint using the default untuned profile: 14.6 GiB KV, MTP depth 3, eight sequences and 262,144-token maximum context. The server reported **906,051 pooled KV tokens**. Streamed chat answered the arithmetic probe correctly, streamed completions and all eight concurrent requests succeeded, usage accounting matched, and the packaged benchmark client passed a short smoke. Production was then restored; its health endpoint and authenticated chat both returned HTTP 200.

All **77 checkpoint inventory files**, including **62 safetensors shards**, matched the public Hub by size and content digest (LFS SHA-256 or Git-blob SHA-1). The documented `t3 download` command resolved the public repository and began transferring files, then was deliberately cancelled and its partial download removed. A complete download was not attempted: available disk space was below the 144.46 GiB checkpoint size. See [`benchmarks/publication.json`](benchmarks/publication.json) for the qualification record.

This was a packaging/API smoke, not a fresh throughput, KL or long-context qualification. Historical performance and quality results above remain unchanged; other GPUs are still unverified.

## Files and provenance

- `nix/`: pinned runtime, CUDA toolkit layout, kernel builds, overlay and opt-in NixOS module.
- `t3/`: quantiser, packet format, allocator, assembler, MTP and arithmetic checkers.
- `scripts/`: checkpoint streaming, source preparation, base/head construction, evaluation, serving and uploads.
- `patches/`: only the generic PLE-SSD patch, tested dependency constraints and architecture support patches; upstream repository hardware utilities are neither vendored nor fetched.
- `benchmarks/`: measured data and public example prompts, never private replay text.

See [NOTICE.md](NOTICE.md) for upstream provenance and licensing. Model weights retain Qwen Community License 1.0, not an Apache/MIT license. No credentials or weights belong in this Git repo.
