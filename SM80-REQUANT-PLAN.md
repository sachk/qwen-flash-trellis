# SM80 fixed-budget requantization testbed plan

Investigation date: 2026-11-17. Read [QUANT-METHOD-REVIEW.md](QUANT-METHOD-REVIEW.md) first for evidence, failed approaches and primary sources.

**Status:** proposal for independent frontier-model review, not authorization to deploy or a claim that a replacement already exists. The investigation used saved outputs, source inspection, metadata and literature; no new model inference/training or serving changes were performed. The public pages now warn that further quality testing was disappointing and a replacement is planned.

## 1. Contract

Develop a testbed of roughly **3,500–7,000 new logical lines**, staying inside the requested 2,000–10,000-line envelope. Reuse the current model/runtime infrastructure or an existing EXL3 backend; do not write a new transformer/recurrent inference engine.

The entire subsequent experimental campaign has a **15.0 aggregate GPU-hour ceiling**, leaving one hour below the user’s strict 16-hour limit. Hours include every GPU used, model warm-up, sampling, teacher passes, training, validation and repeated failed attempts. Two GPUs running for an hour consume two GPU-hours. CPU conversion/download/build time is separate, not a loophole for unmetered GPU work.

Hard deployment constraints:

- Retain **all 512 experts in every layer**, original routing and model architecture. No pruning, zeroed “kept” experts, CPU expert streaming or hidden model substitution.
- No increase in **actual allocated GPU weight bytes** versus the current model, historically logged at approximately 46.16 GiB. Establish the byte-exact baseline at the first isolated launch; include target, MTP, embeddings, head, shared experts, scales, padding and persistent weight-side helpers.
- Preserve native maximum context and do not finance new weights by shrinking the current KV pool. PLE/n-gram remains sparse CPU/SSD storage, not a new GPU-resident table.
- Same or faster **correct** SM80 serving on matched one-/four-stream replay and representative reasoning workloads. Do not retain a numerically invalid optimization merely to meet a speed target.
- Quality goal: approximately **within two percentage points of a matched strong reference** on GPQA-D, normal EOS and identical output budget, without punctuation collapse or pathological token inflation. This is a target, not a promised outcome of fifteen GPU-hours.
- No live-model role/account swapping, production traffic contamination, clock changes, or hardware tuning. Use isolated runtime/port, a finite service window, and automatic restoration. Deploy only after explicit review of results.

The existing checkpoint’s combined GPU effective bpw is roughly three, but model-card bpw labels use different denominators. The authoritative constraints are allocated bytes, all experts, functional equivalence and measured speed.

## 2. Recommended direction

### Control A: released self-calibrated EXL3 2.68

Use [turboderp/Qwen3.8-Flash-Next-exl3, branch `2.68bpw_h6_ng8`](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/2.68bpw_h6_ng8) as the strongest existing procedural-decoder reference. It reports 91.4% GPQA-D on this backbone, has all 512 experts, and six non-ngram model shards total about 43.22 GiB. Loaded GPU residency, SM80 support/performance and exact evaluation parity remain gates.

Use its actual NG8 table and MTP recipe when reproducing the release. Do not swap our BF16 n-gram table into it and call it the same model. Try the existing backend, not a new EXL3 decoder implementation. If this already satisfies all constraints, adopting it beats inventing another quant solely for novelty.

### Main new implementation B: native scalar Q2_0/G64 expert packets

Port the **already learned** [GSQ-RCO Q2_0 release](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) into an SM80-friendly expert layout. Its all-Q2_0 routed experts need no giant expert codebook or activation rotation. Reuse the fast row-compacted MoE scheduling/MMA approach, not the four-state C2 decoder.

First preserve the released quantized target weights, not just expert tensors. This avoids pretending that an expert transplant inherits published quality. Its published 89.39% is 2.53 points below its accompanying BF16 result, so **Q2_0 alone is not yet proof of the desired two-point gate**. It is a credible initialization and performance opportunity, with byte headroom for selective regular higher precision or a small recovery pilot.

### Optional refinement C: spend remaining bytes intelligently

Only after B passes numerical and performance gates, test one bounded refinement: fixed-budget whole-matrix scalar promotions and/or native code/scale recovery on student-generated prefixes. Do not launch full GSQ, global QAT, RL or an entirely new large-state trellis encoder.

Keep current C2 as a diagnostic/control format, not the default new research program. A code-updating, student-prefix/block-aware C2 pilot is a fallback comparison if scalar reuse cannot be made faithful; it must not repeat the rejected scale-only experiments.

## 3. Phase zero: CPU-only preparation and reproducibility

Complete before occupying the serving GPU:

1. Pin checkpoint revisions, hashes, tokenizer/chat template, backend/source revision, runtime flags and format definitions. Record whether the selected model includes vision, draft weights and separate n-gram tensors.
2. Inventory existing downloads and available disk space. Q2_0’s shards total about 66.4 GB; EXL3 model plus NG8 storage is larger than its six model shards. Do not assume both fit simultaneously. Reuse cached objects/hardlinks and stream conversion; never delete unrelated checkpoints or download the full BF16 source just for this experiment.
3. Produce a per-tensor manifest: logical shape/order, source type, byte offsets, scale dtype, loaded target dtype, exact destination bytes and CPU/SSD/GPU placement. Preserve all expert IDs and routing rows.
4. Implement the importer and independent CPU decoder before GPU optimization. Use existing source BF16/FP8 headers and selected byte ranges for any bounded precision comparison.
5. Freeze calibration/development/final datasets by problem/session ID. Deduplicate against GPQA and other final benchmark prompts/solutions. The saved GPQA failures are debugging material, never training examples or hyperparameter-selection labels.
6. Predeclare benchmark settings and the selection rule. Build kernels on the designated CPU build host. Keep every candidate in a separate directory; do not overwrite production artifacts.
7. Implement a persistent GPU-time ledger before model jobs: GPU identity, process/job, start/end time, stage, remaining allowance and stop reason. A controller timeout must include shutdown/restoration overhead and survive a disconnected client.

## 4. Gate A: diagnose the existing degeneration before changing weights

**Maximum 1.0 GPU-hour.** Use one early collapse-onset case and one length-matched successful control, plus a few synthetic state/mask boundaries. The saved traces include an early collapse after only about 1,790 reasoning characters; start there rather than replaying every 32k-token failure.

The archived recovery process positively selected MRV2. The tokenizer’s `!` is token zero, and sustained zero MTP acceptance overlapped collapse. Neither observation proves NaNs. Instrument the earliest divergence, not merely the final all-punctuation output.

### Instrumentation contract

Record, keyed by request and committed position:

- Selected runner/backend and exact source identity; original emitted token IDs.
- Raw and post-processor target logits, proposal logits actually cached before sampling transformations, valid-vocabulary masks, and normalization/log-sum-exp summaries.
- Acceptance decisions, proposal IDs, accepted length, residual support/mass and replacement ID.
- First nonfinite activation/state by layer, active recurrent/checkpoint indices, accepted-token rollback length, block migration/fork metadata and whether the step is prefill/decode/spec-verify.

Do not require every processed logit to be finite: intentional `-inf` masking, including outside draft support, is valid. Require no NaN/+inf, at least one valid candidate per sampled row, finite normalized probabilities, and a valid nonempty residual whenever recovery is invoked. Distinguish valid zero-support edge cases from an invalid normalization. Do not “fix” failure by replacing NaNs with zero, banning `!`, suppressing exceptions or changing the answer budget.

### Paired experiment

At the **same forced committed token prefix** and with the same target weights/packing:

1. MTP off, fresh state: compare target execution with independent decoded-weight reference arithmetic.
2. MTP on: inspect both target p and draft q, plus recovery residual. Compare target verification rows against reference execution of the same forced proposal tokens.
3. If necessary, eager/piecewise versus full graph, and fresh prefill versus incremental/rollback history, changing one variable at a time.
4. Only after localization, isolate FP8 KV/indexer or fused operations. A temporary BF16-cache diagnostic can reduce token capacity while keeping weight bytes unchanged; it is not an approved deployed memory trade.

| Observation | First suspect |
|---|---|
| Target-only valid; speculative target valid; proposal invalid | Draft/MTP numerical path |
| Target-only valid; speculative target invalid | Speculative target/state or multi-row execution |
| Both target paths invalid | Target arithmetic/state, input or reference mismatch |
| Target/proposal valid; residual invalid | Verification/cache/distribution arithmetic |
| All distributions valid and reference-matched | Genuine distributional degeneration or reasoning error |

Seed-identical sampling is not expected across speculative and ordinary generation. Compare distributions and forced prefixes, not independently sampled text. Saved decoded text may not reconstruct the original token segmentation exactly; the diagnostic rerun must capture actual IDs. Fresh prefill can erase a history-dependent corruption and is not sufficient to clear state lifecycle handling.

A reference can stream one expert/layer at a time without loading the full BF16 model. One BF16 expert is 9.375 MiB, one expert layer 4.6875 GiB. Use CPU FP32/FP64 for captured sampler/GDN subproblems and the workstation GPU for small arithmetic cases. Do not try to hold full source and student models together.

**Exit rule:** if a numerical/state bug is identified, repair and prove that operation before attributing quality to the quant. If diagnosis is not localized within the budget, record the unresolved failure and do not run recovery training through a corrupt engine. Existing EXL3 can still serve as an independent reference; the old fork cannot be treated as trusted merely because its small checkers passed.

## 5. Gate B: existing-reference feasibility

**Maximum 0.5 GPU-hour.** Load the released SC EXL3 control in its existing backend, with a bounded KV configuration matching the eventual comparison. Check actual allocated weight bytes, experts, source table, a short reasoning panel, output accounting and real one-/four-stream timing.

This is not its full GPQA evaluation. Its purpose is to establish whether a near-quality checkpoint can already run in the hardware envelope and whether its transforms/procedural depack are actually a bottleneck. Historical old-munekazu EXL3 results do not substitute for this version.

If it meets speed/residency and the main scalar port is not competitive, stop inventing formats and spend the remaining allowance validating the existing candidate. If it fails to load or exceeds memory, record the exact obstacle rather than spending this campaign implementing a replacement EXL3 engine.

## 6. Native Q2_0 packet contract

### Representation

The released GGUF contract is `(q - 1) * d`, q in 0..3, **signed FP16 d per 64 consecutive weights**. Preserve original scale bits and sign. Do not replace d by abs(d), BF16, a new maximum, or a 128-weight scale. Repacking must not introduce a second quantization pass.

For a 16-row by 128-K tile:

- 2,048 logical weights.
- 512 code bytes, reordered for coalesced lane/MMA consumption.
- 32 FP16 scales: two per row, 64 scale bytes.
- **576 total bytes**, naturally aligned to 32 bytes.

The current T2 packet is 544 bytes and cannot preserve Q2_0’s two scales losslessly. Use an explicit new format ID, not existing trellis T2 ID 2. Unknown IDs must fail, not silently dispatch to T2. Change every narrow/wide gate-up/down and prefill/decode dispatch path that consumes that ID.

### Shapes and byte ledger

| Per-expert matrix | Shape | 16x128 packets |
|---|---|---:|
| Fused gate/up | 1280x2560 | 1,600 |
| Down | 2560x640 | 800 |

All dimensions fit this geometry without padding. Across all target experts, code/scale storage is **33,973,862,400 bytes = 31.640625 GiB**. Index/format metadata is additional; the current two offsets/two format arrays total 442,368 bytes across 48 layers.

Compared with current expert tensor storage, a similarly indexed all-Q2_0 expert pack frees approximately **7.5592 GiB**. This is not a free increase in the total GPU allowance: dense/MTP preservation and any higher-precision promotions must fit the original allocation manifest.

### Kernel design

- Reuse row compaction, routed-row scheduling, tensor-core MMA shapes and fusion boundaries already exercised on SM80.
- Convert coalesced 2-bit fields with integer mask/shift/subtract and packed FP16/BF16 arithmetic. No dynamic per-weight format branch, large vector LUT, runtime rotation or whole-expert expansion in the inner loop.
- Make the two 64-weight scale regions explicit; amortize scale loading over the 128-K tile. Measure register pressure, occupancy, scale traffic and actual routed batches.
- Preserve the decoder’s intended arithmetic: compare against FP32 `(q-1)*fp16(d)` followed by the **declared destination rounding**. A BF16 serving path and the original GGUF backend may round differently; lossless code/scale repack is not a promise of bit-identical backend logits or the same GPQA score.
- Do not add per-request allocation/copying or retain expanded BF16 expert tensors. Conversion belongs offline; serving consumes the packed representation directly.

## 7. Whole-model import, not an expert-only quality claim

The Q2_0 model’s routed tensors are uniform Q2_0, but non-experts include regular Q3_K/Q4_K/Q5/Q6/Q8 and small-table IQ4_XS/IQ4_NL. Its separate n-gram shard is IQ4_NL, not BF16.

Maintain three distinct manifests/experiment names:

1. **Q2 experts + current nonexperts/PLE/MTP:** cheapest hybrid diagnostic; no inherited published accuracy claim.
2. **Published Q2 target reconstruction + separately declared MTP:** preferred quality-preservation port; preserve every target tensor, including PLE auxiliaries, head, router and n-gram lookup semantics. The draft remains separately sourced and independently qualified.
3. **New mixed scalar candidate:** any higher-precision substitutions or recovery updates; requires its own quality measurement.

Materialize selected small dense tensors once to a declared dense operand type as a correctness-first option. This decodes the released quant, not original BF16 weights, and can add rounding. A metadata-derived estimate suggests Q2 experts plus expanded nonexperts and a separately counted draft may fit the ceiling, but **there is no loaded residency certificate yet**. Count F32 exceptions and never keep resident source and expanded copies simultaneously.

Dense BF16 expansion may lose the speed goal because it increases bandwidth. Preserve exact representable regular formats through existing optimized linear kernels where possible; a format conversion that cannot preserve scales/affine offsets must be labelled requantization. IQ4’s 16-entry scalar table is not the forbidden giant expert lookup. Decode CPU/SSD n-gram rows sparsely and cache them through the existing PLE offload interface; verify row ordering, special-token handling and prefill/decode equality.

Do not implement a general multi-format GGUF CUDA runtime in this campaign. If whole-target conversion needs too many new dense kernels or violates bytes/speed, fall back to the labelled hybrid experiment and report that the published-quality port was not achieved. Do not silently advertise the hybrid as GSQ-equivalent.

## 8. Gate C: decoder, MoE and state qualification

**Maximum 2.0 GPU-hours**, including initial kernel tuning.

Before model-level timing:

- Independent CPU reconstruction covers all four Q2 codes, both scale signs, subnormal/normal scale boundaries, zero scales and both 64-weight halves. Compare actual scale bits and source code ordering, not repacker output against itself.
- Verify real gate/up/down shapes, expert IDs 0 and 511, all format dispatch paths and mixed-format batches. Check M=1,2,4,8,16,64 and a representative prefill batch, including partial row tiles.
- Compare kernel outputs against independently decoded dense operands with a tolerance justified by accumulation order/dtype. Bit-exact decode and approximate GEMM agreement are distinct contracts.
- Exercise fresh and reused graph buffers, long sequential generation, accepted lengths 0..3, multi-request interleaving, prefill/decode transitions and cache/state migration. Reuse existing valuable checks; add only missing consumer-visible regressions for actual new failure modes.
- Confirm all 512 experts remain present and usable, n-gram tensors stay off GPU, and loaded weight bytes remain below baseline. Count temporary/scratch/graphs separately and preserve the KV pool.
- First smoke is MTP off; enabling MTP requires valid p/q/residuals and matching target distributions on forced prefixes. A speculative speed win is invalid if it changes the intended distribution or hides corruption.

A small standalone depack benchmark is a screening tool only. Promotion depends on the complete routed MoE and end-to-end runtime.

## 9. Optional bounded fixed-byte refinement

**Maximum 2.0 GPU-hours. Skip entirely if numerical or speed gates fail, or if final evaluation needs the reserve.** Select at most one of the following, not an unbounded search.

### Option 1: regular scalar precision allocation

Keep format choice per whole expert matrix/layer, with fixed group-64 or group-128 scales inside. Use Q2_0 plus a small number of regular Q3/G128 or Q4/G128 matrices where held nonlinear/block loss justifies their bytes and latency.

| Format | 16x128 packet bytes | All target experts, before indexing |
|---|---:|---:|
| Q2_0/G64 | 576 | 31.640625 GiB |
| New regular Q3/G128 | 800 | 43.945313 GiB |
| New regular Q4/G128 | 1,056 | 58.007813 GiB |

These Q3/Q4 formats require new quantization; they are not lossless remaps of IQ3 codebooks. Blanket Q3 exceeds the current expert allowance; blanket Q4 is far outside it. RCO-style constrained search is inspiration, not a requirement to reproduce RCO here. Begin with a few byte-neutral, mutually exclusive matrix alternatives, evaluated on actual routed/block output using student-prefix activations. Generate only shortlisted candidates.

Construct promoted candidates from the original BF16 tensors, fetched as selected ranges where necessary, with the same frozen calibration/development contract. Merely expanding an existing two-bit reconstruction into a three-bit container cannot recover information already discarded. Preserve the released Q2_0 candidate as the control and charge any teacher/calibration GPU work to this campaign.

Use a latency surrogate from real SM80 shape/mix measurements in addition to the exact storage budget. A global byte saving that promotes frequently routed matrices can still slow serving. Keep the better initialization unless the export improves session-disjoint development metrics and preserves the declared per-domain guards.

### Option 2: format-preserving recovery

Fix format tables and optimize only existing scalar assignments/scales or valid C2 paths/scales. Start from a strong quantized checkpoint; use a small math/code/science/general calibration mixture disjoint from final benchmarks. Compare equal-compute teacher-prefix versus student-prefix/block-aware supervision, not training loss alone.

For C2, use a PV-style gradient-informed target and a valid native ring/path projection, retaining actual decoder levels and scale precision. Independent per-weight softmax assignments generally violate the trellis constraint. For scalar packets, hard forward codes and stored scales must define the executed model throughout optimization. Export and re-evaluate after every selected checkpoint; do not report fake-quant or unmerged-adapter results.

Prefer a short distillation/reconstruction pilot to RL. On-policy recovery is attractive, but producing teacher supervision at the student’s prefixes is expensive. Cache/chunk selected teacher targets with an explicit tail/support/EOS approximation contract; do not claim exact full-vocabulary KL from a truncated cache. Preserve natural reasoning-end behavior rather than rewarding short wrong answers.

Use small active expert chunks and offload optimizer state. Full-model FP32 scale Adam state is about 14.06 GiB, and full decoded expert weights are 225 GiB. Those are incompatible with a casual “just fine-tune it” assumption. The workstation GPU is useful for one/few-expert arithmetic or training pilots, not a full-model quality certificate. A pilot win must survive exported model-level evaluation; otherwise retain the baseline and stop.

## 10. Development panel and selection

**Maximum 1.5 GPU-hours**, including source/teacher work charged to this stage if not already charged to refinement.

Use a frozen, session/problem-disjoint development panel spanning scientific reasoning, math, code and ordinary instruction following. Include both short and long trajectories, EOS transitions and recurrent-state boundaries. Do not select the winner using the saved 42 GPQA caps or their answers.

Measure accuracy/functional correctness, terminal collapse, cap rate, generated/reasoning tokens, mean/p95 tokens-to-answer, correct answers per GPU-hour, and one-/four-stream latency. Also record p/q validity, MTP acceptance by position and routed format mix. The old KL sets can be regression guards, but sets used to fit layer weights are not pristine holdouts.

Choose **one** finalist before final GPQA. Selection order: numerical correctness; resident bytes/all experts; development quality/termination; matched end-to-end speed. Do not average incomparable published scores into the selection rule.

## 11. Final benchmark protocol and honest certification limits

**Reserve up to 7.5 GPU-hours.** The amount of final coverage is determined by measured throughput/token usage and the remaining ledger, not a promise that arbitrary workloads finish.

Freeze:

- Pinned 198-question GPQA Diamond CSV, answer extraction, normal EOS and deterministic choice shuffle.
- Same prompts, thinking instructions, sampler and **32,768 total output-token cap** for every local comparison arm. Do not silently use a larger budget for the new model or copy a published 131k-cap result into this comparison.
- Primary seed and one sample/question. Predeclare whether the comparison reference is diagnosed T3a392, released SC EXL3, or a pinned-source teacher available elsewhere. A generic commercial Qwen endpoint is not presumed identical to released BF16.
- Same source n-gram table when claiming checkpoint reproduction; otherwise label the substitution. Quality measurement is initially target-only; retain a paired MTP-enabled correctness/speed check before promotion.
- Isolated traffic, complete terminal SSE/usage/finish metadata and persisted per-item outcomes. Infrastructure or numerical failures are separately reported; they cannot be silently dropped from the denominator.

Report raw correct/198, cap/collapse counts, all API-reported input/output/reasoning tokens, median/p95 latency, aggregate throughput and correct answers per GPU-hour. Report statistical uncertainty and seed limitations. A single 198-question run changes by 0.505 percentage points per question and cannot establish a tight universal noninferiority claim. Multiple seeds and a matched BF16 reference remain necessary before claiming broadly near-stock quality.

### Budget feasibility rule

One full capped run can emit **6,488,064 tokens**. At the historical approximately 210 aggregate tok/s, generation alone could take **8.58 hours**. Therefore two worst-case full runs exceed the entire campaign. Do not pretend that “15 GPU-hours” guarantees full multi-model, multi-seed GPQA certification.

Use measured development throughput/token lengths to project completion, including warm-up and headroom. Prefer one complete finalist run to several incomplete scores; reuse the historical T3 result only as a labelled historical reference, not as a clean runtime-matched control. Run a complete matched reference as well only if the ledger supports it. If not, return the complete finalist result against explicitly unmatched published references and state that the near-stock gate is not certified.

If a run reaches the hard experiment deadline, persist every attempt and mark the benchmark incomplete. Do not shorten the remaining questions, lower their cap, omit difficult items, count partial accuracy as a full score, or rerun only failures until they become correct. The campaign can legitimately reject a candidate or produce an inconclusive quality result; it cannot manufacture certainty to fit the allowance.

A useful provisional finalist is within two points of the selected matched reference at the declared budget, with no numerical collapse and no material token inflation. “Within two points of published 91.7/91.92” without a matched protocol is only an exploratory target. Q2_0’s published 89.39 is evidence for a strong baseline, not an automatic success certificate.

## 12. End-to-end speed and memory gate

**Maximum 0.5 GPU-hour**, with shorter timing already collected in previous stages.

Interleave finalist/control replay runs at one and four streams on the same hardware configuration and fixed cache/prefill conditions. Reuse representative retained prompts; public smoke prompts are not equivalent to the historical replay. Include full prefill and real completion accounting, not SSE chunk counts.

Record ms per target verification step, accepted tokens per step, tok/s, prompt throughput, p50/p95 request latency, peak scratch/graphs, actual persistent weight bytes and pooled KV capacity. For reasoning, compare time-to-correct-answer and correct answers/GPU-hour as well as tok/s.

The speed target is **no measured regression**, not “same decoder source”. Use repeated paired measurements and show noise; do not call overlapping intervals a demonstrated speedup. Any actual slowdown fails the requested same-or-faster target even if a standalone unpack kernel wins. Avoid drawing an SM80 conclusion from the GSQ card’s unspecified hardware or an EXL3 5090 example.

## 13. Aggregate GPU-hour ledger

| Stage | Hard GPU-hour allocation |
|---|---:|
| A: numerical/state diagnosis | 1.0 |
| B: existing EXL3 feasibility control | 0.5 |
| C: scalar kernel qualification and initial tuning | 2.0 |
| Optional refinement, one method only | 2.0 |
| Development panel and finalist selection | 1.5 |
| Final quality evaluation reserve | 7.5 |
| Final matched speed/memory replay | 0.5 |
| **Total maximum** | **15.0** |

Every retry and teacher pass debits one of these allocations. Transfer unused allocations to final evaluation, never increase the total. Stop an optional stage early to preserve the final reserve. A global elapsed-GPU meter, finite deadlines and automatic service restoration are required. A long-lived second GPU or an idle resident experimental process still needs explicit accounting; do not hide its occupancy from the experiment log.

## 14. Proposed code footprint

These are design estimates, not code quotas or permission to pad the implementation.

| Component | Approximate new logical lines |
|---|---:|
| Manifest, GGUF reader/repacker and sparse PLE mapping | 600–1,000 |
| Native SM80 Q2_0 packet/MoE kernels | 800–1,600 |
| Loader, explicit format dispatch and dense integration | 400–900 |
| Numerical/state probes and independent reference checks | 600–1,000 |
| Optional scalar Q3/Q4 or native-code recovery pilot | 500–1,500 |
| Evaluation, ledger/deadline controller and valuable regression checks | 600–1,000 |
| **Total** | **3,500–7,000** |

Reuse existing CUDA scheduling, tensor-core placement, PLE offload and benchmark clients. Keep import/reconstruction, serving-format implementation, and calibration/optimization separate. No new general quantization framework, standalone model server or copied vendor runtime is needed.

## 15. Deliverables and promotion decision

The campaign should leave:

- Exact source/destination manifests and checksums, resident-byte report and proof of all experts.
- The numerical failure localization or an explicit unresolved-cause record; retained target/draft/residual diagnostic inputs without public/private data leakage.
- One runnable packed candidate and the independent decoder/kernel validation outputs, or a clear stage rejection with the last known-good checkpoint retained.
- Reproducible isolated launch/profile, finite GPU-hour ledger, benchmark settings and complete per-item outcome artifacts.
- A comparison separating published scores, historical local data and new matched measurements. No new public near-original claim without the corresponding quality evidence.

**Promote only if** numerical correctness, byte ceiling, all-expert retention, quality/termination and no-speed-regression gates all pass. Otherwise keep the warning, keep production/restoration safe, and submit the measured rejection to the next reviewer. Do not promote a lower local loss, a smaller file, an attractive microbenchmark or an incomplete GPQA score.

## 16. Questions for the next frontier-model review

1. Does the existing SC EXL3 2.68 checkpoint already dominate the proposed new implementation on SM80 once its current backend is measured?
2. Should the first new candidate preserve the entire Q2_0 release or intentionally use a labelled expert transplant to avoid expensive dense-format integration? Which exact allocated bytes and arithmetic changes follow?
3. Are the target/proposal numerical probes sufficient to separate draft-only failure from target quantization, recurrent rollback and valid-distribution degeneration?
4. Can the proposed 576-byte packet retain enough occupancy/scale reuse to beat current C2 on the real routed shapes, including prefill and wide batches?
5. Within the hard fifteen-hour ledger, which single optional refinement has the highest expected value: regular mixed-bit allocation or native code-updating student-prefix recovery?
6. What final reference and benchmark coverage can actually certify the desired quality claim without assuming unpublished protocols or silently expanding GPU compute?

The next iteration should answer these questions before authorizing implementation or changing production. The best outcome may be adopting a proven checkpoint with a faster native path, not discovering a new quantization family.
