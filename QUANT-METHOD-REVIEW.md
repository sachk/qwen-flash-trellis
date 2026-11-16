# Fixed-budget Qwen3.8-Flash-Next quantization: method review

Investigation date: 2026-11-17. Companion proposal: [SM80-REQUANT-PLAN.md](SM80-REQUANT-PLAN.md).

## 1. Decision in brief

The goal remains credible: approximately three effective bits per GPU-resident weight, all 512 experts, cheap SM80 depacking, and near-original reasoning quality. It is **not established for T3a392**. Our earlier confidence was too high: teacher-forced KL, short arithmetic checks and memory stress did not establish long-reasoning task quality.

The strongest next candidates are:

1. **Released self-calibrated EXL3 2.68 bpw**, as an external accuracy/performance control before inventing another representation. Its publisher reports 91.4% GPQA-D on this exact backbone. Its procedural decoder is much more relevant than a large vector-codebook decoder, although transforms and SM80 throughput must be measured.
2. **GSQ-RCO Q2_0 expert packing**, as the first new native SM80 scalar-kernel lane. Its publisher reports 89.39% versus its accompanying BF16 91.92%. All routed experts are ordinary Q2_0; expensive IQ2/IQ3 expert lookup is not required.
3. **Format-preserving recovery of existing packets**, only as a bounded control after runtime diagnosis. Update valid codes as well as scales using student-prefix/block or token-level objectives. Repeating already-failed scale-only fitting is not a new experiment.

A concrete numerical/sampling failure must be ruled out first. The observed punctuation collapses cannot presently be attributed to target-weight compression alone. A full new GSQ/QAT training campaign is not a credible promise within 16 total GPU-hours; reusing already-trained checkpoints makes the bounded testbed much more realistic.

## 2. Constraints and accounting

- No increase over the current loaded GPU-weight allocation, historically logged as **46.16 GiB**. This rounded log value is not an exact byte certificate: record an allocation manifest before promoting anything.
- Keep all 512 experts in every layer and the original routing architecture. Top-10 routing is not expert pruning.
- Target expert tensors contain **120,795,955,200 weights**. Current expert packet tensors, including offsets/format metadata, occupy **42,090,938,368 bytes = 39.200241 GiB**.
- Keep PLE/n-gram storage off GPU. Count MTP, head, embeddings, shared experts, alignment and persistent helper tensors in the resident-weight comparison.
- Compare **allocated GPU bytes**, not download size. A metadata-only scan found roughly 49.05 GiB of non-PLE stored tensors in the existing directory; startup packing, language-only loading and unused/redundant tensors make this different from measured GPU residency.
- Branch labels such as EXL3 “2.68 bpw”, GGUF whole-model averages, expert-only bitrates, and our combined target/draft GPU-resident effective bpw have different denominators. Do not rank them as identical accounting units.
- Preserve the current fast serving envelope. Same kernels or fewer packed bytes do not guarantee the same end-to-end speed: routing, dense operations, MTP acceptance and reasoning length matter.

## 3. What the completed GPQA run actually showed

The recovered run has 198 unique completed API responses, 150 correct, no remaining transport errors, and the same temperature 1/top-p .95/top-k 20/min-p 0/xhigh settings throughout. Choice shuffling, answer mappings and persisted correctness were reconciled against the pinned dataset. Six completed wrong answers were not extraction errors.

| Outcome | Questions | Median per-request decode tok/s |
|---|---:|---:|
| Correct, finished before cap | 150 | 133.62 |
| Explicit wrong answer before cap | 6 | 126.20 |
| Other responses reaching 32,768 tokens | 26 | 132.22 |
| Terminal punctuation collapse, reaching cap | 16 | 60.03 |

The collapse diagnostic is intentionally narrow: a terminal run of at least 1,024 identical non-whitespace characters. All 16 detected tails consist of `!`, with lengths 7,667–32,186 characters. No uncapped response meets that diagnostic. The other 26 capped responses must **not** be called degenerate merely because they truncated.

The 16 collapse requests consumed 524,288 generated tokens; the other capped requests consumed 851,968. All scored responses together consumed 2,558,971 output tokens, including 2,511,154 reasoning tokens. Speed in the public summary covers only 184 new recovery requests; 14 completed responses were retained unchanged after an infrastructure interruption. See [the measured summary](benchmarks/gpqa-diamond-20261114.json). Raw traces remain in gitignored `local-docs/gpqa/recovered-20261114/results-recovery/`.

This is a disappointing **deployment** result, not a controlled measurement of intrinsic quantization loss. Qwen’s published 91.7% and the third-party scores below use insufficiently documented protocols. Increasing the cap might help some of the 26 other cases, but is not a convincing cure for thousands of identical punctuation characters. Even making all 16 collapse cases correct would yield only 83.84%; both failure groups matter.

## 4. First diagnostic priority: invalid target or proposal probabilities

### Established observations

- The actual deployed tokenizer maps `!` to **token ID 0**. Saved API text does not preserve original token IDs, so the decoded tails alone do not prove that every emitted token was zero.
- Archived recovery startup logs explicitly say **“Using V2 Model Runner”**, followed by the 46.16-GiB loading measurement. MRV2 is positively identified, not inferred from a current default.
- During overlapping collapse requests, archived logs from 10:13:12 through 10:14:02 UTC on 2026-11-14 report two running requests, approximately 110 generated tok/s, three drafted tokens per replacement, **zero accepted drafts**, and mean acceptance length 1.00. Here 1.00 means the replacement token only, not one accepted draft.
- All 198 scored responses report zero cached prompt tokens. The documented prefix-cache-hit prerequisite of upstream issue 55766 does not match this run.
- Token zero is included in the retained 65,536-token draft vocabulary. Simple omission of `!` from draft support is ruled out.

### Concrete mechanism, not proven cause

The pinned MRV2 rejection implementation computes residual quantities involving `exp(logq - logp)`. Invalid normalization of **either target p or proposal q** can make the comparison against one false for every candidate; residual logits become entirely `-inf`, after which lowest-index recovery can produce zero. A draft-only numerical failure can therefore spoil output even when target weights/logits are otherwise sound.

For valid distributions, the inspected reduced-draft-vocabulary path preserves full target support and caches proposal logits with the relevant sampling transformations. That mathematical property does not cover NaNs. Zero acceptance can also occur with entirely valid but badly divergent p/q; it is not itself proof of numerical corruption.

Primary pinned sources: [acceptance](https://github.com/vllm-project/vllm/blob/a5a30471ff2bb7f0824f2da10e358af98d304472/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py#L628-L691), [recovery](https://github.com/vllm-project/vllm/blob/a5a30471ff2bb7f0824f2da10e358af98d304472/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py#L812-L962), [acceptance metric definition](https://github.com/vllm-project/vllm/blob/a5a30471ff2bb7f0824f2da10e358af98d304472/vllm/v1/spec_decode/metrics.py#L109-L119).

Upstream [55766](https://github.com/vllm-project/vllm/issues/55766) documents GDN/state/cache corruption producing NaN logits and `!` floods on related models. It establishes a relevant failure class, not applicability of that particular trigger. [58231](https://github.com/vllm-project/vllm/pull/58231) documents an all-masked-token-zero failure under thinking-budget forcing, but our requests use active top-k and no explicit thinking-token budget; xhigh is textual instruction, not that budget. Do not cargo-cult either patch.

**Missing causal evidence:** emitted token IDs, onset raw/processed target and proposal logits, their normalization, residual mass/support, and recurrent checkpoint/history. Fresh prefill may erase a lifecycle defect. Do not hide this uncertainty by clamping NaNs, banning punctuation, adding repetition penalties, or selecting a fallback answer.

## 5. The weight-side weaknesses are real, but separate

### Aggressive low-bit allocation

Actual format tables across 24,576 expert/layer pairs:

| Matrix | T2 | T3 | T4 |
|---|---:|---:|---:|
| Fused gate/up | 14,614 | 9,962 | Not supported in current serving dispatch |
| Down | 4,435 | 11,358 | 8,783 |

Thus 59.46% of gate/up matrices and 18.05% of down matrices use T2. Upgrading all T2 to T3 while preserving T4 would add 6.42 GiB: **not an admissible replacement** under the user’s fixed-weight constraint.

### Objective and trajectory mismatch

`t3/t3_pass.py`, `scripts/s2_pass.py` and `t3/t3lib.py` optimize per-matrix relative quadratic reconstruction on source/teacher trajectories. Gate/up rows are optimized independently; down covariances use teacher intermediates. This misses nonlinear SiLU-product sensitivity, co-routed expert error cross terms, downstream/hyperconnection sensitivity and the prefixes produced by the quantized student itself.

A prior paired source-expert substitution, keeping dense/head/KV/embed/shared settings fixed, reduced own-domain simulator KL from .086290 to .028761 and multilingual KL from .036763 to .005472. This establishes substantial expert-compression error. It does not prove GPQA causality or show that the gap is recoverable in the same bitrate.

### Pooling is much broader than “almost unseen experts”

The deployed pass uses NMIN=16 and damping .08. Gate/up pooling starts below 40,960 routes; down below 10,240. Median observed routes are 10,086. Only 13 expert/layer pairs have fewer than 100 routes, but many otherwise well-observed experts still use strongly pooled covariances.

The implementation adds a pooled normalized covariance with coefficient `threshold / route_count` below a hard threshold. It equally averages routed experts into the pool and uses raw counts rather than gate-weighted effective sample size. These are deliberate regularizers, not demonstrated bugs. Their strength, discontinuity and mismatch to reasoning-sensitive experts deserve controlled ablation, not an assumption that removing pooling helps.

### Allocation search is restricted

T4 down candidates are preselected by gate-squared routing mass; some late-layer T2 options are excluded, and other layers have partial candidate sets. A favorable alternative absent from the library cannot be selected. The greedy allocator prices independent relative-error changes using output energy, a uniform floor and empirical layer weights; this does not recover a nonlinear, non-decomposable model objective.

Own/multilingual/medical KL contributed to fitting the layer weights. Those sets are development data for allocation, not untouched final holdouts. The historical calibration used roughly 581k tokens, with short windows and specialist continuation documents; it was not a long scientific-reasoning recovery corpus. The portable public-only builder is not a byte-for-byte reproduction of the historical calibration.

### Encoder and provenance limitations

The four-state additive trellis solver can be exact for its supplied additive costs. The surrounding correlated quadratic Babai search retains limited numerical histories and is not globally exact for the full correlated objective. Ordinary polishing had already provided negligible benefit.

Experts were rounded from dequantized official FP8 weights, while dense/PLE tensors are inherited from another base. The local BF16-source pilot reduced one expert’s weight MSE by 8.34%, but used an identity Hessian. Two sampled dense/control tensors were already BF16 and identical across official releases. BF16-source regeneration is a useful factorial control, not a credible explanation for recovering sixteen GPQA points by itself.

A real order-dependent T2/T4 allocator conflict defect exists in current code. **The deployed plan has no such intersections**, and its formats and shard lengths match: it is not an explanation for this checkpoint. The next testbed must make alternatives mutually exclusive and reject unsupported format IDs rather than inherit default-to-T2 dispatch.

## 6. Approaches already tried: what failed and what was not learned

These are retained historical results, not experiments run for this review. They are summarized in the earlier requant `PROGRESS.md` and `SM80-OPTIMISATION-LOG.md`; relevant portable owners include `t3/t3lib.py`, `t3/t3_alloc.py` and `t3/sim_eval.py`.

| Approach | Observed outcome | Consequence |
|---|---|---|
| Current four-state C2 trellis deployment | GPQA 75.76%, with unresolved runtime contribution | Failed the reasoning-quality goal; not proof that trellises generally fail |
| Full-Hessian fixed-code scale fitting | Example held error .046105 to .046152 | Do not repeat full-model scale-only fitting |
| Teacher-prefix student-aware down rerounding | Development error worsened; best held result merely tied control | Nonlinear compensation alone was insufficient in that setup |
| Nonlinear scale-only optimization | Broader-data improvements approximately 1%, without a strong model-quality lead | Lower local training loss is not promotion evidence |
| Independent/per-row gains and joint routed gains | Some local/development wins; mixed domain/model KL | No all-domain validated replacement |
| Coupled native allocation | Claimed held gain invalidated by source mapping; corrected result retained baseline | Correct development/holdout provenance is a hard gate |
| Up/down diagonal equalization | Pilot selected baseline | No demonstrated free quality gain |
| H128/scalar and E8 screens | No justified rewrite; E8 held proxy error worsened 17.3% | Do not promise Hadamard or lattice coding as an easy fix |
| Alternative computed-codebook decoders | Decoder screens lost to native C2 in that study | Full transform-inclusive speed matters |
| Plain-rounded W4 dense projections | Own KL approximately .09 to .23 | Do not free expert bytes by casually damaging dense/recurrent projections |

Our validation/promotion mistake was broader than choosing a representation: success on replay KL, kernel arithmetic, short completions and memory capacity was treated as stronger evidence of reasoning quality than it was. A four-state custom C2 implementation is not published QTIP or modern self-calibrated EXL3 merely because all use trellises.

## 7. Directly relevant published alternatives

**Published numbers below are not a matched ranking.** Our run used xhigh, one sample/question and a 32,768 total-output cap. The GSQ card does not fully specify its protocol. The EXL3 publisher explicitly describes limited single runs with EvalScope and backend defaults, including a previous sampler-comparability correction.

| Exact-backbone model | GPQA-D (%) | Reported precision / resident-file evidence | Interpretation |
|---|---:|---|---|
| T3a392 deployment | 75.76 | About 46.16 GiB loaded GPU weights | Current measured deployment, unresolved numerical failures |
| GSQ-RCO Q2_0 | 89.39 | 2.40 reported average; resident shard 37,623,740,192 bytes = 35.04 GiB | First regular scalar candidate |
| GSQ-RCO IQ2_XS | 87.37 | Resident shard approximately 39.2 GB decimal | Slower expert codebooks; not the primary port target |
| GSQ-RCO IQ3_XXS | 91.41 | Resident shard approximately 47.0 GB decimal | Quality control, not a promise of cheap SM80 lookup |
| SC EXL3 2.18 / H6, NG8 | 89.4 | Publisher’s 2.18-bpw recipe | Smaller procedural-decoder control |
| SC EXL3 2.68 / H6, NG8 | 91.4 | Six model shards total 46,411,488,305 bytes = 43.22 GiB, excluding n-gram file | Strongest directly relevant external near-quality reference |
| SC EXL3 3.18 / H6, NG8 | 91.4 | Larger recipe; residency not certified here | No clear published GPQA gain over 2.68 |
| SC EXL3 3.68 / H6, NG8 | 91.9 | Larger recipe | Not assumed to fit our budget |
| GSQ study’s BF16 reference | 91.92 | BF16 | Study-specific reference, not a matched local baseline |

Sources: [GSQ release](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), [SC EXL3 release](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3), [EXL3 branch config](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/blob/2.68bpw_h6_ng8/config.json), [EXL3 branch inventory](https://huggingface.co/api/models/turboderp/Qwen3.8-Flash-Next-exl3/tree/2.68bpw_h6_ng8?recursive=true).

The EXL3 config retains 512 experts and specifies mul1, H6, vision6 and MTP2.5. Its evaluated n-gram variant is NG8; replacing it with our BF16 table changes the tested model. File-size fit is not loaded-GPU fit. Previous CMP EXL3 measurements used an older munekazu checkpoint/runtime and different workloads; they cannot determine the current SC checkpoint’s speed or quality.

### 7.1 Why Q2_0 deserves a separate lane

The [complete allocation manifest](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/blob/main/tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.rco-allocation.txt) assigns **all 144 routed gate/up/down tensors across 48 layers to Q2_0**, shared across 512 experts. Format selection is coarse; scales within each matrix remain fine-grained. These are different concepts.

Its decoder contract is 64 weights, one signed FP16 scale d and 16 code bytes: **18 bytes/64 weights = 2.25 bpw**, reconstructing `(code - 1) * d`. The grid is {-1,0,1,2} times d. Preserve the FP16 scale bits and sign; do not infer the exported contract from GSQ’s differently oriented internal integer grid.

A 16x128 packet needs 512 code bytes and 32 FP16 scales: **576 bytes**, versus current T2’s 544. It fits the existing 16-row/128-K tile geometry without padding for both K=2560 gate/up and K=640 down. All routed Q2_0 packets total **31.640625 GiB**, before small indexing metadata, versus current expert tensors’ 39.200241 GiB. This provides room for selective higher precision or preserving dense reconstruction, not permission to add weights beyond the overall cap.

Primary decoder: [pinned Q2_0 reconstruction](https://github.com/ggml-org/llama.cpp/blob/7804be0634c578d1da1604a2f2320b33f5a65377/ggml/src/ggml-quants.c#L442-L462).

The whole Q2_0 checkpoint is **not exclusively Q2_0 or LUT-free**. It also contains Q3_K/Q4_K/Q5/Q6/Q8 and IQ4_XS/IQ4_NL on non-routed tensors. IQ4 uses a tiny 16-entry scalar table, not the large IQ2/IQ3 vector-codebook expert path. Its separate 28,800,138,432-byte n-gram shard is IQ4_NL. Preserve or explicitly account for every dense, PLE, router, head and n-gram tensor.

An expert-only transplant into our existing dense/PLE/MTP is a new hybrid and does not inherit 89.39%. Materializing GGUF dense reconstruction into BF16/FP16 does not restore original weights, can introduce rounding, and may slow dense GEMMs. A full target-weight port is the better quality-preservation baseline, but still requires fresh evaluation on the new arithmetic/runtime.

The card’s 93.79 decode tok/s and 367.49 prompt tok/s are not matched CMP170HX measurements. The user’s observation that IQ variants were slow is valid; it should not be generalized to Q2_0’s ordinary shift/mask/subtract/scale expert decoding.

### 7.2 Why self-calibrated EXL3 is more relevant than generic “trellis” claims

EXL3 uses procedural mul1 decoding, block Hadamard processing/channel scaling and regular 256-weight trellis tiles; it is not a huge arbitrary lookup table. Its implementation is Ampere-aware. Nevertheless, rotations, depacking, tensor shapes and recurrent/PLE serving overhead must be measured together. The publisher’s 5090 speed example is not an SM80 result.

Use the released 2.68 branch as a control rather than spend this experiment rebuilding its encoder. If it already meets residency, quality and speed gates on SM80, using it is preferable to inventing another representation solely to have a new quant. Primary code/documentation: [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [conversion](https://github.com/turboderp-org/exllamav3/blob/master/doc/convert.md), [optimization](https://github.com/turboderp-org/exllamav3/blob/master/doc/optimize.md).

Reproduction caveat: the current optimization documentation calls that pipeline experimental and **untested on sparse models**, despite the exact-backbone card publishing self-calibrated results. The checkpoint's calibration rows/recipe and producer version must be pinned; do not assume the generic current CLI reproduces this MoE release. Its documented approach uses self-sampled in-domain traces and output-sensitivity-shaped noise to choose tensor bitrates, not a claim of full end-to-end QAT. This is another reason to benchmark the released checkpoint before attempting a new conversion within the compute cap.

### 7.3 Format selection is not scale grouping or value assignment

| Candidate | Format-selection unit | Scale/code organization | Serving consequence |
|---|---|---|---|
| Current C2 | Whole expert matrix | 16x128 packets, 128-weight row scales | Regular loads, but constrained neighbor-dependent reconstruction |
| GSQ Q2_0 experts | Whole gate/up/down tensor, shared across all experts | 64-weight scalar scale blocks | Constant scalar type throughout the expert; no per-weight format dispatch |
| Self-calibrated EXL3 | Tensor bitrate recipe | Block Hadamard processing and 256-weight trellis tiles | Coarse regular encoding with procedural reconstruction plus transforms |
| IQ2/IQ3 expert formats | Tensor-level format allocation | Block/vector-codebook structures | Coarse format selection does not remove codebook reconstruction cost |

GSQ's learned per-coordinate assignment chooses a value on the selected grid, not a different unpacking format for every weight. Its published procedure jointly optimizes assignments and scales, stages reconstruction objectives within blocks, and propagates inputs through the already-quantized prefix. RCO targets a non-decomposable model objective under a budget. Those differences from our independent teacher-local rounding and greedy surrogate allocation are plausible quality contributors; scalar versus trellis alone is not a causal explanation.

## 8. Other families: useful evidence, but avoid false comparisons

| Family | Available GPQA evidence | Hardware/layout relevance | Limitation |
|---|---|---|---|
| Regular GGUF Q3_K/Q4_K / dynamic allocation | DeepSeek-R1 study reports DQ3_K_M GPQA 68.95 ±0.65 versus FP8 API 69.58 ±1.65, with a disclosed 32k-cap protocol | Scalar bitplanes/subscales, A100 support; no giant vector LUT | Different model; DQ3_K_M averages 3.59 bpw, not our budget |
| GSQ refinement in an unchanged GGUF format | Qwen3-8B Q3_K_M GPQA 54.65 to 55.15; Q2_K 46.46 to 50.91 | Same inference format after optimization | Different model; full published GSQ compute is not within this campaign |
| Reasoning-aware QAT | Small Qwen/R1-distill studies report GPQA recovery under W2/W3 | Same scalar layouts can be retained | Recovery remains below BF16; not a guarantee on this 125B-class MoE |
| GPTAQ | No qualifying GPQA-D result found in the original paper/code | Student-input/asymmetric calibration ideas suit regular scalar grids | Treat as a bounded algorithmic hypothesis |
| Published QTIP | No GPQA-D result found in the paper | Computed-code trellis, often larger state and Hadamard processing | Released HYB differs from our C2 and includes a small LUT; published quality does not transfer |
| HIGGS/FLUTE | No qualifying GPQA-D result found in the paper | A100 support, Hadamard/Gaussian grids and practical shared-memory lookup | Not a demonstrated lookup-free solution for this model |
| AutoRound exact-backbone W2/W3 | No qualifying exact-backbone GPQA result found | Coarse tensor recipes and integer kernels | Existing tiny inherited smoke report is not GPQA evidence |

Sources: [DeepSeek quantization study](https://arxiv.org/html/2505.02390v1), [GSQ paper](https://arxiv.org/html/2604.18556v2), [GPTAQ](https://arxiv.org/html/2504.02692), [QTIP](https://arxiv.org/html/2406.11235), [HIGGS](https://arxiv.org/html/2411.17525), [FLUTE](https://github.com/HanGuo97/flute).

Uniform Q3_K costs 3.4375 bpw, not three bits including overhead. Its block-256 layout does not divide the 640-wide down projection without padding/rearrangement. Uniform custom Q3/G128 routed packets already occupy 43.95 GiB before non-expert/MTP weights; blanket three-bit experts are not automatically admissible either.

## 9. Post-training: compatible opportunities and compute limits

The important distinction is **learning better parameters within the deployed representation**, not appending a new model component.

- [PV-Tuning](https://arxiv.org/html/2405.14852) alternates continuous and discrete updates while keeping a representation compatible with existing inference. For C2, the discrete update must change valid tail-biting paths/rings, not independently choose impossible per-weight levels. For scalar packets, assignments/scales can change directly within the chosen grid.
- [EfficientQAT](https://aclanthology.org/2025.acl-long.498/) combines blockwise parameter training with end-to-end scale refinement. Its reported 70B/A100-80GB training took 41 hours, not our allowance.
- [Reasoning-QAT](https://arxiv.org/html/2601.14888) supports PTQ initialization followed by knowledge-distillation recovery; direct RL from a severely damaged low-bit policy can fail.
- [OnQ](https://arxiv.org/html/2610.29378v1) specifically supervises the quantized student’s own prefixes, reducing repetition/truncation on 1.7–8B Qwen models. That is a relevant mechanism, not demonstrated feasibility or quality for this much larger MoE.
- [Reasoning-token inflation](https://arxiv.org/html/2606.25519) supports measuring tokens-to-answer and end-to-end cost, rather than claiming success from local kernel tok/s or short-task accuracy.

For a bounded pilot, use a strong existing quantized initialization, fixed export formats, native code/scale updates, and student-prefix teacher supervision on a disjoint multi-domain reasoning set. Train neither on GPQA nor on the saved GPQA answers. Avoid a length-only reward that makes the model confidently terminate incorrectly.

Do not promise a full GSQ/RCO optimization or global QAT run here. The GSQ paper reports 8 H200s for ten hours even for an 8B case. Full decoded target expert weights alone require **225 GiB**. Existing scales number **943,718,400**; FP32 scale parameters, gradients and two Adam moments alone cost **14.06 GiB**, not “tiny adapter” memory. Frozen downstream weights still incur backward activation/compute costs.

Stream one layer or small expert chunks, teacher and student passes separately where necessary. One BF16 expert is 9.375 MiB; one whole layer is 4.6875 GiB. Limit trainable optimizer state to a small chunk. Cache selected teacher logits only with an explicit approximation contract; preserve reasoning-end/EOS information rather than silently omitting it from a top-k target. A paid generic “Qwen Flash” API is not automatically the same teacher checkpoint.

Persistent LoRA/residual adapters violate the no-extra-weights/compute premise unless removed. Merging an adapter and then quantizing again can erase recovery; judge only the exported hard-quantized checkpoint. Arbitrary codebooks or rotations cannot generally be folded into the current C2 decoder for free.

## 10. What this review does and does not establish

Established: the release failed the requested reasoning-quality standard; numerical collapse is a serious runtime hypothesis; independent expert compression creates substantial distributional error; strong exact-backbone scalar/procedural alternatives now have published GPQA evidence; Q2_0 permits a cheap native expert layout with useful byte headroom.

Not established: the first faulty numerical operation, whether a runtime repair alone recovers the 16 cases, how much of the other 26 caps is budget versus quantization, current SC EXL3 speed on this SM80, or a same-byte new quant meeting near-stock quality in less than 16 GPU-hours.

The companion plan therefore prioritizes **diagnosis, existing-checkpoint controls and one scalar port**, with a small recovery experiment only if the earlier gates succeed. It is a bounded, falsifiable testbed—not a promise of a new Pareto frontier.
