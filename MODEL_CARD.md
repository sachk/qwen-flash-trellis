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

Mixed 2/3/4-bit ring-coded trellis quantisation of the routed experts, retaining all 512 experts per layer. Expert packets occupy ~39.2 GiB; loaded GPU model weights were measured at ~46.16 GiB. **46 GiB is not the total VRAM requirement:** the measured serving setup adds a 14.6 GiB KV pool, graphs, workspace and runtime overhead, and requires a 64 GiB-class GPU. The original BF16 PLE stays on SSD rather than GPU memory.

## Measured quality and performance

| GPQA Diamond | T3a392 | Original model (published) |
|---|---:|---:|
| Accuracy | **178/198 (89.90%)** | **91.7%** |
| Output cap, including reasoning | 98,304 tokens | Not published |
| Requests reaching the output cap | 1/198 (0.5%) | Not published |
| Mean / median generated tokens per question | 15,093 / 8,673 | Not published |
| Generated tokens, including reasoning | 2,988,418 | Not published |
| NaN/collapsed responses | 0 | — |

| Speed and memory (64 GiB CMP 170HX) | T3a392 |
|---|---:|
| GPU-resident model weights | 46.16 GiB |
| GPU KV pool | 14.6 GiB |
| Aggregate KV capacity on a 64 GiB GPU | 906,051 tokens |
| One-stream decode | 166.5 tok/s |
| One-stream step time | 16.29 ms |
| Four-stream total decode | 352.6 tok/s |
| Four-stream step time | 30.64 ms |
| One-stream prefill | 3,073 tok/s |
| Accepted draft tokens per step (1 / 4 streams) | 1.71 / 1.70 |

Measured 2026-11-18 on stock GPU settings: one sample per question, xhigh reasoning, temperature 1, top-k 20, top-p 0.95, seed 20261114 + question index, eight concurrent streams, production serving configuration. [Measurement record with per-question outcomes](https://github.com/sachk/qwen-flash-trellis/blob/master/benchmarks/gpqa-diamond-20261118.json).

Questions that hit a 32,768-token cap (32/198) were rerun with a 98,304-token cap and identical seeds; every other question finished below 32,768 tokens, so the table equals a 98k-cap run up to batch-level nondeterminism. At a 32,768-token cap the same run scores 159/198 (80.30%). Capped answers count as incorrect. The [official 91.7% reference](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/812d96e10a0dd82b69d749e68693698cf6c36daa/README.md) does not publish its prompt, sampling, aggregation or token budget, so the gap is **not a measured quantisation penalty**, and one sample per question carries roughly ±2–3 points of sampling noise.

Measured 2026-11-18, NVIDIA CMP 170HX (74-SM GA100, 64 GiB HBM) in a stable configuration: SM clock capped at 1400 MHz, stock memory clock. **This card is not yet fully SM- or memory-bus-unlocked**, so these are not full-GA100 figures. No overclocking/unlock code or settings are distributed. Numbers are not promises for other GPUs.

Replay uses eight 4,000-token private-traffic prompts, 72 one-stream rounds and 36 four-stream rounds, real text/prefill, 400 generated tokens, temperature 1, top-k 20, top-p 0.95, MTP depth 3 and a 65,536-token draft vocabulary. Tokens/s moves with draft acceptance, so compare step time first. Figures published earlier (2026-11-09) were measured on a GPU configuration later found unstable and are withdrawn.

KV capacity is pooled across requests; the native single-request limit is **262,144 tokens**.

The last deployed update increased pooled KV capacity from 893,325 to **906,051 tokens (+1.42%)**, reduced expert scratch to 55.33 MiB and removed a 1.15 GiB prefix-copy temporary. It was a **memory/reliability improvement, not a demonstrated decode-speed gain**. The 250 one-stream / 500 four-stream-total tok/s goals were not met.

**Compared with the AutoRound base.** T3a392 reuses the dense layers of [`klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP`](https://huggingface.co/klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP) and replaces its 3-bpw routed experts. On the same own-traffic teacher-forced evaluation its KL is **0.090 versus 0.168 for AutoRound**, a little over half the divergence. Its last production speed figures on this card were also lower (2026-10-30, same vLLM stack, same stable 1400 MHz clock): 127.35 tok/s single-request and 316.13 tok/s four-request aggregate, on ~8K-token code prompts with 1,024 generated tokens and time to first token included. That is a different workload from the performance table above, so it is not a matched speed ratio. On the matched replay benchmark at the same clock, the first T3 build equalled AutoRound's speed (152.7 tok/s single-stream) with the lower KL. Later runtime work (64K draft vocabulary, expert-kernel tuning, verify-mask kernel) raised T3a392 to the performance figures above.

**Superseded 2026-11-14 run** ([record](gpqa-diamond.json)): 150/198 (75.76%) at 32,768 tokens with 42 capped, two streams. That machine's non-stock GPU configuration produced rare all-NaN logit steps (about 24 per hour of long-reasoning decode). Each affected request then emitted token ID 0 until the cap, which appeared as "repetitive output" and token-limit failures. On stock settings the identical software produced no NaN steps in about 3.5 hours of generation, and a model-free bit-exact GPU stress replay passed. Of the 16 questions that collapsed on 11-14, 10 are correct at 32k and 13 at 98k.

## Quality details

### Teacher-forced fidelity

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

**This is a custom format, not GPTQ/GGUF/AWQ and not a drop-in stock-vLLM model.** It needs the pinned vLLM PLE-SSD patch, trellis loader and CUDA kernels. Companion Nix/code repository: https://github.com/sachk/qwen-flash-trellis. Standard `transformers.from_pretrained` and hosted HF inference are not supported by this artifact. Validated kernel architecture is Ampere SM80 (GA100), **not Ada**; other GPUs are unverified.

All serving checkpoint weights are included, including the ~102.4 GB decimal PLE shard. Total download is approximately **155 GB decimal (~144 GiB)**, not 39 or 46 GB. See `inventory.json` for exact file sizes. Use a fast local SSD, sufficient free storage, a suitable NVIDIA driver for CUDA 13, and the companion runtime. Native maximum single-request context is **262,144 tokens**; **906,051 tokens is pooled KV capacity across requests**, not a single-request context claim.

Requantisation research, now an efficiency/headroom project rather than a quality fix: [method comparison](https://github.com/sachk/qwen-flash-trellis/blob/master/QUANT-METHOD-REVIEW.md) and [fixed-budget SM80 testbed plan](https://github.com/sachk/qwen-flash-trellis/blob/master/SM80-REQUANT-PLAN.md). These are proposals, not a validated replacement.

## Format and references

Expert weights are ring-coded trellis packets on a custom 4-state trellis code: each ring tail-bits through a unit's packet words, so the closed path pays no start/end boundary-state overhead, and the trellis's constrained reconstruction levels give richer quality per transmitted bit than scalar quantisation at the same rate. Rings are packed by Hessian-aware Viterbi rounding with per-expert second-order costs and sequential error feedback, in the spirit of GPTQ's objective. This pipeline is custom, not stock QTIP: no QTIP randomized-Hadamard incoherence processing (RHT), 1MAD/3INST decode settings, or QuIP#/GLQ lattice codebooks are used. Kernel decode is bit-exact with tensor-core-friendly packet layouts; that is a design property, not a speed guarantee on other GPUs.

References: A. Tseng, Q. Sun, D. Hou, C. De Sa, "QTIP: Quantization with Trellises and Incoherence Processing," NeurIPS 2024, [arXiv:2406.11235](https://arxiv.org/abs/2406.11235) (hardware-efficient parallel trellis/computed codes; tail-biting, §3.2). E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh, "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers," ICLR 2023, [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) (second-order objective and error compensation). M. W. Marcellin, T. R. Fischer, "Trellis Coded Quantization of Memoryless and Gauss-Markov Sources," IEEE Transactions on Communications 38(1):82–93, 1990, [doi:10.1109/26.46532](https://doi.org/10.1109/26.46532) (original TCQ).

## Provenance and licensing

FP8 source: `Qwen/Qwen3.8-Flash-Next-FP8` at `236dfdf285828023ca3bcd3f37366c58a3469b13`. Dense/PLE base: `klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP` at `ce0e0b94083895bd836b916f29bf105c40a8162a`. vLLM revision: `a5a30471ff2bb7f0824f2da10e358af98d304472`.

Weights retain the **Qwen Community License 1.0**, reproduced in `LICENSE`. It is **not Apache-2.0/MIT**: commercial Model-as-a-Service / AI Work Assistant businesses have additional licensing requirements; read the full license. Preserve source notices. Calibration included private traffic; raw conversations, tokens, traces, credentials and private session identifiers are not distributed. Public-only recalibration can reproduce the procedure, not the exact calibration statistics or checkpoint bytes.
