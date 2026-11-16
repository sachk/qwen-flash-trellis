---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model:
- Qwen/Qwen3.8-Flash-Next-FP8
- klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP
tags:
- quantized
- trellis
- ampere
- sm80
- vllm
- text-generation
pipeline_tag: text-generation
---
# Qwen3.8-Flash-Next T3a392 — Ampere SM80, ~46 GiB GPU-resident weights

> **Quality update:** Further testing has been disappointing: this release scored **75.76% on GPQA Diamond**, with substantial token-limit failures and repetitive output. Earlier KL checks overstated our confidence in reasoning quality. A replacement is planned soon; do not treat this release as near-original quality. Weight-compression and runtime/sampling causes are still being separated.

Mixed 2/3/4-bit ring-coded trellis quantisation of the routed experts, retaining all 512 experts per layer. Expert packets occupy ~39.2 GiB; loaded GPU model weights were measured at ~46.16 GiB. **46 GiB is not the total VRAM requirement:** the measured serving setup adds a 14.6 GiB KV pool, graphs, workspace and runtime overhead, and requires a 64 GiB-class GPU. The original BF16 PLE stays on SSD rather than GPU memory.

**This is a custom format, not GPTQ/GGUF/AWQ and not a drop-in stock-vLLM model.** It needs the pinned vLLM PLE-SSD patch, trellis loader and CUDA kernels. Companion Nix/code repository: https://github.com/sachk/qwen-flash-trellis (currently private while packaging is validated). Standard `transformers.from_pretrained` and hosted HF inference are not supported by this artifact. Validated kernel architecture is Ampere SM80 (GA100), **not Ada**; other GPUs are unverified.

All serving checkpoint weights are included, including the ~102.4 GB decimal PLE shard. Total download is approximately **155 GB decimal (~144 GiB)**, not 39 or 46 GB. See `inventory.json` for exact file sizes. Use a fast local SSD, sufficient free storage, a suitable NVIDIA driver for CUDA 13, and the companion runtime. Native maximum single-request context is **262,144 tokens**; **906,051 tokens is pooled KV capacity across requests**, not a single-request context claim.

## Measured performance

Historical production replay, 2026-11-09, NVIDIA CMP 170HX (74-SM GA100, 64 GiB HBM). Non-stock hardware configuration; no overclocking/unlock code or settings are distributed. Numbers are not promises for other GPUs or the portable runtime's untuned architecture defaults.

| Deployed run | One stream tok/s | One stream ms/step | Four streams total tok/s | Four-stream ms/step |
|---|---:|---:|---:|---:|
| 1 | 189.25 | 14.413 | 383.04 | 28.280 |
| 2 | 185.27 | 14.429 | 375.48 | 28.139 |

Single-stream prefill: **3,238–3,265 tok/s**. Replay uses 72 one-stream rounds and 36 four-stream rounds, real text/prefill, 400 generated tokens, temperature 1, top-k 20, top-p 0.95. MTP depth 3 and a 65,536-token draft vocabulary; acceptance varies, so compare ms/step before tokens/s. Do not compare directly to synthetic endless-completion benchmarks. Original private replay conversations are not published.

The last deployed update increased pooled KV capacity from 893,325 to **906,051 tokens (+1.42%)**, reduced expert scratch to 55.33 MiB and removed a 1.15 GiB prefix-copy temporary. It was a **memory/reliability improvement, not a demonstrated decode-speed gain**. The 250 one-stream / 500 four-stream-total tok/s goals were not met.

## Quality

Real-server teacher-forced KL against an FP32 source teacher, estimated over the union of teacher/student top-64 plus a tail term. Lower is better. These are distribution-divergence measurements, **not** task-accuracy, blinded preference or long-context reasoning benchmarks.

| Domain | Reference before memory update | Released serving configuration |
|---|---:|---:|
| Own/private replay | 0.0919 | 0.0924 |
| Nemotron | 0.0391 | 0.0391 |
| Multilingual | 0.0366 | 0.0365 |
| Medical | 0.0305 | 0.0312 |
| Legal | 0.0273 | 0.0271 |

All absolute changes were at most 0.0007, within recorded run noise. This does not establish losslessness against the source model. No failed overnight gain-fitting, native-allocation, scalar-codebook or lattice experiments replace the released production checkpoint. `benchmarks.json` includes both baseline/candidate replay pairs and measurement caveats.

Memory stress passed two simultaneous 250k-token contexts followed by eight short concurrent streams, peak 64,710 MiB GPU use. This is a memory/lifecycle test, not a long-context quality benchmark. Production pack/decode, graph replay, cache-copy regression and applicable Compute Sanitizer checks passed during original qualification.

## Provenance and licensing

FP8 source: `Qwen/Qwen3.8-Flash-Next-FP8` at `236dfdf285828023ca3bcd3f37366c58a3469b13`. Dense/PLE base: `klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP` at `ce0e0b94083895bd836b916f29bf105c40a8162a`. vLLM revision: `a5a30471ff2bb7f0824f2da10e358af98d304472`.

Weights retain the **Qwen Community License 1.0**, reproduced in `LICENSE`. It is **not Apache-2.0/MIT**: commercial Model-as-a-Service / AI Work Assistant businesses have additional licensing requirements; read the full license. Preserve source notices. Calibration included private traffic; raw conversations, tokens, traces, credentials and private session identifiers are not distributed. Public-only recalibration can reproduce the procedure, not the exact calibration statistics or checkpoint bytes.
