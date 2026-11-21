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

**Code, pinned runtime & Nix flake:** [github.com/sachk/qwen-flash-trellis](https://github.com/sachk/qwen-flash-trellis)

Mixed 2/3/4-bit ring-coded trellis quantisation of all 512 routed experts per layer (~39.2 GiB expert packets; **46.16 GiB** loaded GPU weights). Needs a **64 GiB-class Ampere SM80 (GA100) GPU** (46.16 GiB weights + 14.6 GiB FP8 KV pool + runtime overhead) and fast local SSD for the ~95.37 GiB BF16 PLE table (**~144.46 GiB / 155 GB decimal** total download).

> **Custom format, not stock vLLM / GGUF / AWQ.** Requires the companion [vLLM PLE-SSD patch, trellis loader, and SM80 CUDA kernels](https://github.com/sachk/qwen-flash-trellis). Standard `transformers`, hosted HF inference, and non-SM80 GPUs are unsupported/unverified.

## Benchmarks

### GPQA Diamond

| Metric | T3a392 (ours) | [Qwen (published)](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/812d96e10a0dd82b69d749e68693698cf6c36daa/README.md) | [Artificial Analysis](https://artificialanalysis.ai/models/qwen3-8-flash-next) | [ISTA-DASLab (BF16)](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | [OpenRouter (Alibaba API)](https://openrouter.ai/qwen/qwen3.8-flash) |
|---|---:|---:|---:|---:|---:|
| Accuracy | **90.40%** (179/198) | **91.7%** | **92.3%** | **91.92%** (182/198) | **91.41% / 91.33%** |
| Output cap (incl. reasoning) | 259,300 (full 262k context) | Not published | Provider max | Not published | Not published |
| Requests hitting cap | 0/198 | — | — | — | — |
| Mean / median tokens per question | 15,224 / 8,673 | — | 17,606 / — | — | — |
| Longest response | 124,295 tokens | — | — | — | — |
| Total generated tokens (198 Q) | 3,014,409 | — | 3,485,931* | — | — |
| NaN / collapsed responses | 0 | — | — | — | — |

*\*Artificial Analysis canonical single-pass token count: 3,432,440 reasoning + 53,491 answer tokens (60,192 input tokens vs. 62,570 in our run).*

- **Protocol (2026-11-18, stock GPU settings):** 1 sample/question, `xhigh` reasoning, temp 1.0, top-k 20, top-p 0.95, seed `20261114 + idx`, 8 concurrent streams ([full record](https://github.com/sachk/qwen-flash-trellis/blob/master/benchmarks/gpqa-diamond-20261118.json)). Questions capped at 32k (32/198) were rerun at 98k with identical seeds, and the 1 remaining item at 259,300 (finished at 124,295). Scores at intermediate caps: **80.30%** (159/198) at 32k, **89.90%** (178/198) at 98k; capped responses count wrong.
- **Comparability:** Harnesses differ in prompt, sampling, repeats, and cap (Qwen's 91.7% is repeated identically on its BF16 and FP8 cards without protocol details; OpenRouter reports two Alibaba Cloud Int. endpoints, 1 run each). A single 198-question pass carries ~±2–3 pp sampling noise, so differences across columns are **not a measured quantisation penalty**.
- **Superseded 2026-11-14 run** (`gpqa-diamond.json`): scored 150/198 (75.76%) at 32k due to rare all-NaN logit steps (~24/h) caused by an unstable non-stock GPU overclock. Stock settings produced 0 NaN events across ~3.5 h and passed a bit-exact GPU stress replay.

#### Same-backbone third-party quants (unmatched protocols, context only)

| Checkpoint | Resident weight footprint | Backend / harness | GPQA Diamond |
|---|---:|---|---:|
| **T3a392 (this repo)** | 46.16 GiB loaded GPU weights | vLLM SM80 (ours) | **90.40%** |
| [turboderp EXL3 SC 2.68 bpw](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) | ~43.2 GiB shards (excl. n-gram) | TabbyAPI / EvalScope (1 run) | 91.4% (2.18: 89.4%, 3.18: 91.4%, 3.68: 91.9%) |
| [ISTA-DASLab GSQ-RCO Q2_0](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | 37.6 GB resident shard | llama.cpp | 89.39% (IQ2_XS: 87.37%, IQ3_XXS: 91.41%, IQ3_S: 92.93%) |

### Speed and memory (64 GiB CMP 170HX)

| Metric | T3a392 |
|---|---:|
| GPU-resident model weights | 46.16 GiB |
| GPU KV pool (FP8) | 14.6 GiB |
| Aggregate pooled KV capacity (native max 262,144 / req) | 906,051 tokens |
| Single-stream decode / step time | 166.5 tok/s / 16.29 ms |
| Four-stream total decode / step time | 352.6 tok/s / 30.64 ms |
| Single-stream prefill | 3,073 tok/s |
| Accepted draft tokens / step (1 / 4 streams, MTP-3) | 1.71 / 1.70 |

Measured 2026-11-18 on a 64 GiB CMP 170HX (74 SMs enabled, SM clock capped at 1400 MHz, stock memory clock; not fully SM- or memory-bus-unlocked). Real replay: eight 4k-token private prompts, 72/36 rounds at 1/4 streams, 400 output tokens, temp 1.0/top-k 20/top-p 0.95, 64K draft vocab. Compare step time first (tok/s moves with draft acceptance). Earlier 2026-11-09 numbers from an unstable GPU config are withdrawn.

### Teacher-forced fidelity & AutoRound comparison

Real-server KL vs. FP32 teacher (top-64 union + tail; lower is better; distribution divergence, not a task benchmark):

| Domain | KL |
|---|---:|
| Own/private replay | 0.0924 (vs. 0.168 for AutoRound base) |
| Nemotron | 0.0391 |
| Multilingual | 0.0365 |
| Medical | 0.0312 |
| Legal | 0.0271 |

T3a392 reuses the dense/PLE layers of [`klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP`](https://huggingface.co/klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP) and replaces its routed experts, cutting own-replay KL nearly in half (0.090 vs. 0.168) while exceeding its throughput (166.5 vs. 152.7 tok/s on matched early replay). Memory stress passed two simultaneous 250k contexts + eight short streams (peak 64,710 MiB).

## Format and references

Routed experts use custom **ring-coded 4-state trellis packets** (mixed 2/3/4-bit payloads, tail-biting loops with zero boundary-state overhead) packed by Hessian-aware Viterbi rounding with sequential error feedback. Decode is bit-exact with tensor-core-friendly layouts. Custom to this repo—not stock QTIP (no RHT or 1MAD/3INST) and no QuIP#/GLQ codebooks. Further efficiency/headroom research: [method review](https://github.com/sachk/qwen-flash-trellis/blob/master/QUANT-METHOD-REVIEW.md), [SM80 testbed plan](https://github.com/sachk/qwen-flash-trellis/blob/master/SM80-REQUANT-PLAN.md).

References: Tseng et al., *QTIP* (NeurIPS 2024, [arXiv:2406.11235](https://arxiv.org/abs/2406.11235)); Frantar et al., *GPTQ* (ICLR 2023, [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)); Marcellin & Fischer, *TCQ* (IEEE TCOMM 1990, [doi:10.1109/26.46532](https://doi.org/10.1109/26.46532)).

## Provenance and licensing

- **FP8 source:** `Qwen/Qwen3.8-Flash-Next-FP8` @ `236dfdf285828023ca3bcd3f37366c58a3469b13`
- **Dense/PLE base:** `klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP` @ `ce0e0b94083895bd836b916f29bf105c40a8162a`
- **vLLM pin:** `a5a30471ff2bb7f0824f2da10e358af98d304472`
- **Files:** 62 safetensors shards (~144.46 GiB / 155 GB decimal incl. ~102.4 GB PLE shard); see `inventory.json`.
- **License:** Weights retain **Qwen Community License 1.0** (`LICENSE`), not Apache-2.0/MIT (commercial MaaS / AI Work Assistant terms apply). Calibration included private traffic; raw private data is not distributed.
