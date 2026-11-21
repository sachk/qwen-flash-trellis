# Qwen Flash trellis

**Weights (Hugging Face):** [sachk0/Qwen3.8-Flash-Next-T3a392-Ampere-SM80-46GiB-GPU-Weights](https://huggingface.co/sachk0/Qwen3.8-Flash-Next-T3a392-Ampere-SM80-46GiB-GPU-Weights)

Standalone quantisation tools and pinned Nix/vLLM runtime for **Qwen3.8-Flash-Next T3a392**: all 512 routed experts per layer in mixed 2/3/4-bit ring-coded trellis packets, **46.16 GiB** GPU-resident weights, with the BF16 PLE table kept on SSD.

Targets **Ampere SM80 (GA100)**; other architectures are unverified. Contains no GPU unlock/overclock tools, firmware, credentials, or private calibration data (`KUDA_*` env vars are loader API names only).

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

- **Protocol (2026-11-18, stock GPU settings):** 1 sample/question, `xhigh` reasoning, temp 1.0, top-k 20, top-p 0.95, seed `20261114 + idx`, 8 concurrent streams ([record](benchmarks/gpqa-diamond-20261118.json)). The 32/198 items capped at 32k were rerun at 98k (same seeds), and the 1 item still capped at 98k was rerun at 259,300 (finished at 124,295). Intermediate caps: **80.30%** (159/198) at 32k, **89.90%** (178/198) at 98k; capped items count wrong.
- **Comparability:** Third-party harnesses differ in prompt, sampling, repeats, and cap (Qwen repeats 91.7% on both BF16 and FP8 cards without protocol details; OpenRouter lists two Alibaba Cloud Int. endpoints, 1 run each). One pass over 198 items carries ~±2–3 pp sampling noise, so differences across columns are **not a measured quantisation penalty**.
- **Superseded 2026-11-14 run** ([record](benchmarks/gpqa-diamond-20261114.json)): 150/198 (75.76%) at 32k due to rare all-NaN logit steps (~24/h) from an unstable GPU overclock. Stock settings produced 0 NaN events in ~3.5 h and passed a bit-exact GPU stress replay.

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

Measured 2026-11-18 ([`benchmarks/speed-20261118.json`](benchmarks/speed-20261118.json); withdrawn 2026-11-09 unstable-GPU figures in [`benchmarks/results.json`](benchmarks/results.json)) on a 64 GiB CMP 170HX (74 SMs enabled, SM clock capped at 1400 MHz, stock memory clock; not yet fully SM- or memory-bus-unlocked). Replay uses eight 4k-token private prompts, 72/36 rounds at 1/4 streams, 400 output tokens, temp 1.0/top-k 20/top-p 0.95, 64K draft vocab. Compare step time first (tok/s moves with draft acceptance).

### Teacher-forced fidelity & AutoRound base

Server KL vs. FP32 teacher (distribution divergence, not a task benchmark): **0.0924** own/private replay, **0.0391** Nemotron, **0.0365** multilingual, **0.0312** medical, **0.0271** legal. Compared with the [`klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP`](https://huggingface.co/klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP) base whose dense/PLE layers T3a392 reuses, own-replay KL drops from **0.168 to 0.090** while matching or beating throughput (166.5 vs. 152.7 tok/s on matched replay after 64K draft-vocab and kernel tuning). Memory stress passed two simultaneous 250k contexts + eight short streams (peak 64,710 MiB).

Further efficiency/headroom research (proposals, not a validated replacement): [method review](QUANT-METHOD-REVIEW.md) and [SM80 testbed plan](SM80-REQUANT-PLAN.md).

## Method and references

Routed-expert weights are packed as **ring-coded 4-state trellis packets** (mixed 2/3/4-bit payloads; tail-biting rings avoid boundary-state overhead) using **Hessian-aware Viterbi rounding** with per-expert second-order costs and sequential error feedback. Kernel decode is bit-exact with tensor-core-friendly packet layouts. Custom to this repo—not stock QTIP (no RHT or 1MAD/3INST) and no QuIP#/GLQ codebooks.

- A. Tseng et al., *QTIP: Quantization with Trellises and Incoherence Processing*, NeurIPS 2024 ([arXiv:2406.11235](https://arxiv.org/abs/2406.11235))
- E. Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, ICLR 2023 ([arXiv:2210.17323](https://arxiv.org/abs/2210.17323))
- M. W. Marcellin & T. R. Fischer, *Trellis Coded Quantization*, IEEE TCOMM 38(1):82–93, 1990 ([doi:10.1109/26.46532](https://doi.org/10.1109/26.46532))

## Requirements

- Linux x86_64 with Nix flakes and an NVIDIA driver supporting CUDA 13.
- **64 GiB-class SM80 GPU** for the measured setup (46.16 GiB weights + 14.6 GiB KV + runtime overhead).
- Fast local SSD for the **~144.46 GiB** checkpoint (62 safetensors shards, incl. ~95.37 GiB BF16 PLE table) plus runtime/download caches. Requantising needs extra scratch disk and ~56 GiB GPU memory (do not run beside a live server).

## Run with Nix

```sh
export T3_WORK_DIR=/path/on/ssd/qwen-trellis
export T3_RUNTIME="$T3_WORK_DIR/runtime"
nix develop
t3 setup
t3 download sachk0/Qwen3.8-Flash-Next-T3a392-Ampere-SM80-46GiB-GPU-Weights
t3 serve --model "$T3_WORK_DIR/model" --kv-gib 14.6
```

`nix run . -- <cmd>` works identically. On non-NixOS Linux, set `LD_LIBRARY_PATH` and `TRITON_LIBCUDA_PATH` to your NVIDIA driver libs. Changing pinned runtime sources requires a fresh `T3_RUNTIME` directory.

The API listens on **127.0.0.1:8099** (model name `qwen3.8-flash-next`); set `VLLM_API_KEY` before exposing beyond loopback. See `t3 serve --help` for memory/context flags.

Pinned stack: vLLM `a5a30471ff2bb7f0824f2da10e358af98d304472` (PLE-SSD offload, software FP8 for Ampere, INT8 head/shared/embed, fused HC, exact verify mask, MTP-3 with 64K draft vocab, 2048-token expert scratch, graphs up to 32 rows) plus 205 locked Python deps in `patches/runtime-lock.txt` (refresh via `t3 python scripts/lock_runtime.py --out patches/runtime-lock.txt`).

### NixOS module

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

Ensure `modelPath` is readable/traversable by `qwen-trellis` (read-only is fine) and `stateDirectory` has room for runtime/caches. API keys load via systemd credentials. Don't run alongside another server on the same GPU.

### Benchmarks and evaluation

**Speed replay** (run against an idle server; pass representative long prompts in JSONL):

```sh
export OPENAI_API_KEY_FILE=/path/to/api-key  # omit if unauthenticated on loopback
t3 benchmark --prompts benchmarks/prompts.jsonl --rounds 72 --output replay.json
```

**GPQA Diamond** ([`scripts/gpqa_eval.py`](scripts/gpqa_eval.py), checks SHA-256 of [OpenAI's GPQA Diamond CSV](https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv)):

```sh
t3 python scripts/gpqa_eval.py --csv /path/to/gpqa_diamond.csv \
  --out local-docs/gpqa/run --base-url http://127.0.0.1:8099 \
  --streams 2 --max-tokens 32768 --reasoning-effort xhigh
```

Supports `--skip-existing` to resume; transport errors invalidate quality results and exit nonzero. `local-docs/` is gitignored. Unit checks: `python3 -m unittest discover -s tests -v`.

**BF16 vs. FP8 source pilot** (streams only selected tensor ranges; runs on an RTX 5070 without SM80 kernels):

```sh
python scripts/source_precision_probe.py --out local-docs/source-precision --device cuda
```

## Requantise from source

Public-only calibration reproduces the procedure (15% agent / 85% chat primary budget; session-disjoint splits), not the original private calibration statistics or exact bytes. After `t3 setup`:

```sh
export T3_WORK_DIR=/path/on/ssd/requant
export T3_BASE_MODEL="$T3_WORK_DIR/base"
export T3_FP8_SOURCE="$T3_WORK_DIR/source"
t3 python scripts/s0_download.py
t3 python -m huggingface_hub.cli.hf download klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP \
  --revision ce0e0b94083895bd836b916f29bf105c40a8162a --local-dir "$T3_BASE_MODEL"
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

Long layer passes support `--resume`. GPU kernel checks (stop serving first):

```sh
t3 python t3/t3_check.py
t3 python t3/topk_check.py --no-bench
t3 python t3/hc_check.py --checkpoint "$T3_WORK_DIR/model"
t3 python t3/t3_roundtrip.py
```

Offline KL: `t3/sim_eval.py`; real-server KL on port 8199: `scripts/s4_eval.py --skip-speed` (defaults to `eval_public`).

## Release verification

Qualification record: [`benchmarks/publication.json`](benchmarks/publication.json) (2026-11-14).

- Isolated Nix build + fresh `t3 setup` installed locked vLLM `0.29.1rc1.dev402+ga5a30471f.ple1`, Torch `2.13.0+cu130`, Transformers `5.17.0`.
- Packaged `t3 serve` (14.6 GiB KV, MTP-3, 8 streams, 262k max context) reported **906,051 pooled KV tokens** and passed chat/completion streaming, 8-stream concurrency, usage accounting, and benchmark smoke.
- All **77 checkpoint files** (62 safetensors shards) matched the Hub by size and digest (LFS SHA-256 / Git SHA-1); `t3 download` resolution/transfer verified.
- Public-data CLI and GPQA scoring unit tests pass (`t3 python -m unittest discover -s tests -v`).

## Files and provenance

- `nix/`: pinned runtime, CUDA toolkit layout, kernel builds, overlay, and NixOS module.
- `t3/`: quantiser, packet format, allocator, assembler, MTP, and arithmetic checkers.
- `scripts/`: download, calibration prep, base/head build, evaluation, serving, and upload.
- `patches/`: PLE-SSD patch, software FP8 / SM80 kernels, and locked dependency constraints.
- `benchmarks/`: measured GPQA/speed/publication records and smoke prompts.
- `MODEL_CARD.md`: Hugging Face model card mirror.

See [NOTICE.md](NOTICE.md) for upstream provenance and licensing. Model weights retain **Qwen Community License 1.0**, not Apache/MIT.
