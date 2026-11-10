#!/usr/bin/env python3
"""Applies the T3 hooks to a private copy of the vLLM source. Exits non-zero if any anchor text is missing."""
import re, shutil, sys
from pathlib import Path
root = Path(sys.argv[1])
fm = root / "vllm/model_executor/layers/fused_moe"
shutil.copy(Path(__file__).parent / "t3_method.py", fm / "t3_method.py")
shutil.copy(Path(__file__).parent.parent / "t3gemm.py", fm / "t3gemm.py")
p = fm / "routed_experts.py"
s = p.read_text()
a = '''        quant_method = None
        if quant_config is not None:
            quant_method = resolve_quant_method(quant_config, self, prefix)
'''
assert a in s
s = s.replace(a, '''        from vllm.model_executor.layers.fused_moe.t3_method import (
            T3MoEMethod,
            t3_layer_index,
        )

        t3_index = t3_layer_index(prefix)
        if t3_index is not None:
            return T3MoEMethod(moe_config, t3_index)
        quant_method = None
        if quant_config is not None:
            quant_method = resolve_quant_method(quant_config, self, prefix)
''')
a = '''        expert_mapping = self.get_expert_mapping(include_fused=True)
        for expert_name, loaded_weight in weights:
'''
assert a in s
s = s.replace(a, '''        if hasattr(self.quant_method, "load_weights"):
            yield from self.quant_method.load_weights(self, weights)
            return
        expert_mapping = self.get_expert_mapping(include_fused=True)
        for expert_name, loaded_weight in weights:
''')
p.write_text(s)
# Optional architecture-level QSA launch overrides; no device-specific table is installed.
p = root / "vllm/models/qwen4_exp/nvidia/ops/qsa.py"
s = p.read_text()
a = """    elif base_programs > 2048:
        BLOCK_N, target_splits, num_warps = (
"""
assert a in s
s = s.replace(a, """    elif _sm80_override(base_programs) is not None:
        BLOCK_N, target_splits, num_warps = _sm80_override(base_programs)
""" + a)
a = "def _select_config(\n"
assert a in s
s = s.replace(a, """def _sm80_override(base_programs: int):
    table = os.environ.get("KUDA_QSA_TABLE")
    if not table:
        return None
    for entry in table.split(","):
        limit, block_n, splits, warps = map(int, entry.split(":"))
        if base_programs <= limit:
            return block_n, splits, warps
    return None


""" + a, 1)
if "\nimport os\n" not in s:
    s = s.replace("\nimport torch\n", "\nimport os\n\nimport torch\n", 1)
p.write_text(s)
p = root / "vllm/model_executor/layers/fused_moe/router/fused_topk_router.py"
s = p.read_text()
a = "            padding is not None, num_warps=4,\n"
assert a in s
s = s.replace(a, "            padding is not None, num_warps=1,\n")
p.write_text(s)
p = root / "vllm/v1/worker/gpu/spec_decode/speculator.py"
s = p.read_text()
a = """    def load_model(self, target_model: nn.Module) -> None:
"""
assert a in s
s = s.replace(a, """    def load_model(self, target_model: nn.Module) -> None:
        self._load_model_impl(target_model)
        from vllm.model_executor.layers.fused_moe.t3_method import install_draft_vocab

        install_draft_vocab(self, target_model)

    def _load_model_impl(self, target_model: nn.Module) -> None:
""", 1)
a = """        if draft_logits is not None:
            logits = self.model.compute_logits(hidden_states)
            sampled = gumbel_sample(
"""
assert a in s
s = s.replace(a, """        if draft_logits is not None:
            logits = self.model.compute_logits(hidden_states)
            table = getattr(self, "draft_vocab_table", None)
            if table is not None:
                # Back to the full vocabulary: q is zero outside the table.
                full = torch.full((logits.shape[0], self.draft_vocab_size_full), float("-inf"), dtype=logits.dtype, device=logits.device)
                full[:, table] = logits
                logits = full
            sampled = gumbel_sample(
""", 1)
p.write_text(s)
# Draft distribution filtered like the target's. Rejection sampling stays exact with any q, so only acceptance changes.
p = root / "vllm/v1/worker/gpu/spec_decode/speculator.py"
s = p.read_text()
a = """            table = getattr(self, "draft_vocab_table", None)
            if table is not None:
"""
assert a in s
# Before expanding to the full vocabulary, so top-k scans only the draft table.
s = s.replace(a, """            from vllm.model_executor.layers.fused_moe.t3_method import filter_draft_logits

            logits = filter_draft_logits(logits, idx_mapping, temperature)
""" + a)
p.write_text(s)
p = root / "vllm/models/qwen4_exp/nvidia/model.py"
s = p.read_text()
a = """            self.lm_head.quant_method = Sm80Int8Head()
"""
assert a in s
s = s.replace(a, a + """            from vllm.model_executor.layers.fused_moe.t3_method import apply_int8_layers

            apply_int8_layers(self)
            from vllm.model_executor.layers.fused_moe.t3_method import apply_t3_gemm

            apply_t3_gemm(self)
""")
p.write_text(s)
p = root / "vllm/models/qwen4_exp/nvidia/mtp.py"
s = p.read_text()
a = """        self.set_moe_parameters(self.model.layers)
"""
assert a in s or "set_moe_parameters(" in s
s, n = re.subn(r"(\n        enable_qwen4_exp_low_latency_gemm\(self[^\n]*\n)", r"\1        from vllm.model_executor.layers.fused_moe.t3_method import apply_t3_gemm\n\n        apply_t3_gemm(self)\n", s, count=1)
assert n == 1
p.write_text(s)
# FP8 compressed indexer keys on SM80: Triton has no FP8 type here, so raw bytes are encoded and decoded in software.
d = root / "vllm/models/qwen4_exp/nvidia"
p = d / "ops/qsa_indexer.py"
s = p.read_text()
a = """@triton.jit
def _qsa_mqa_paged_uniform_kernel("""
assert a in s
s = s.replace(a, """@triton.jit
def _decode_e4m3fn(raw):
    raw = raw.to(tl.uint32)
    half_bits = (((raw & 128) << 8) | ((raw & 127) << 7)).to(tl.uint16)
    return half_bits.to(tl.float16, bitcast=True).to(tl.float32) * 256.0


""" + a, 1)
assert s.count("    STAGES: tl.constexpr,\n) -> None:") == 2
s = s.replace("    STAGES: tl.constexpr,\n) -> None:", "    STAGES: tl.constexpr,\n    SOFTWARE_FP8: tl.constexpr = False,\n) -> None:")
a = """            mask=live[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
"""
assert s.count(a) == 2
s = s.replace(a, """            mask=live[:, None] & (dims[None, :] < HEAD_DIM),
            other=0,
            eviction_policy="evict_first",
        )
        if SOFTWARE_FP8:
            keys = _decode_e4m3fn(keys).to(query.dtype)
""")
assert s.count("num_warps=1 if k_cache.dtype == torch.float8_e4m3fn else 2,") == 2
s = s.replace("num_warps=1 if k_cache.dtype == torch.float8_e4m3fn else 2,", "num_warps=1 if k_cache.dtype in (torch.float8_e4m3fn, torch.uint8) else 2,\n            SOFTWARE_FP8=k_cache.dtype == torch.uint8,")
a = "    if k_cache.dtype == torch.float8_e4m3fn:\n        TILE_R, STAGES, num_warps = 32, 2, 8"
assert a in s
s = s.replace(a, "    if k_cache.dtype in (torch.float8_e4m3fn, torch.uint8):\n        TILE_R, STAGES, num_warps = 32, 2, 8")
# Software FP8 scores a BF16 query against keys viewed as uint8: the type check, prefill and warmup all have to allow that.
a = '''    assert q.dtype == k_cache.dtype, "Q and the compressed K cache must match"\n'''
assert s.count(a) == 2
s = s.replace(a, '''    # Software FP8 on SM80 scores a BF16 query against keys viewed as uint8.\n    assert q.dtype == k_cache.dtype or (q.dtype == torch.bfloat16 and k_cache.dtype == torch.uint8), "Q and the compressed K cache must match"\n''')
start = s.index("def _prefill_logits(")
end = s.index("\ndef ", start + 10)
body = s[start:end]
assert "SOFTWARE_FP8" not in body
body, n = re.subn(r"(\n\s+)num_warps=num_warps,", r"\g<0>\1SOFTWARE_FP8=k_cache.dtype == torch.uint8,", body, count=1)
assert n == 1
s = s[:start] + body + s[end:]
start = s.index("def warmup_qsa_mqa_paged_decode(")
end = s.index("    return profiles\n", start)
body = s[start:end]
a = "    k_cache_ptr = TritonWarmupTensor(k_cache.dtype, shape=tuple(k_cache.shape))\n"
assert a in body
body = body.replace(a, "    software_fp8 = k_cache.dtype == torch.float8_e4m3fn\n    cache_dtype = torch.uint8 if software_fp8 else k_cache.dtype\n    k_cache_ptr = TritonWarmupTensor(cache_dtype, shape=tuple(k_cache.shape))\n")
a = "        q_ptr = TritonWarmupTensor(\n            k_cache.dtype,\n"
assert a in body
body = body.replace(a, "        q_ptr = TritonWarmupTensor(\n            torch.bfloat16 if software_fp8 else k_cache.dtype,\n")
a = "            num_warps=1 if k_cache.dtype in (torch.float8_e4m3fn, torch.uint8) else 2,\n            SOFTWARE_FP8=k_cache.dtype == torch.uint8,\n"
assert a in body
body = body.replace(a, "            num_warps=1 if cache_dtype in (torch.float8_e4m3fn, torch.uint8) else 2,\n            SOFTWARE_FP8=cache_dtype == torch.uint8,\n")
s = s[:start] + body + s[end:]
p.write_text(s)
p = d / "ops/qsa_pre_indexer.py"
s = p.read_text()
s = s.replace("@triton.jit", """@triton.jit
def _encode_e4m3fn(x):
    bits = (x.to(tl.float32) / 256.0).to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
    mag = bits & 32767
    rounded = tl.minimum((mag + 63 + ((mag >> 7) & 1)) >> 7, 126)
    return (((bits >> 15) << 7) | rounded).to(tl.uint8)


@triton.jit""", 1)
a = """                + compressed_row * compressed_cache_stride_token
                + dims,
                tl.reshape(y, (D,)),
                mask=valid,
"""
assert a in s
s = s.replace(a, """                + compressed_row * compressed_cache_stride_token
                + dims,
                _encode_e4m3fn(tl.reshape(y, (D,))) if compressed_cache_ptr.dtype.element_ty == tl.uint8 else tl.reshape(y, (D,)),
                mask=valid,
""")
p.write_text(s)
p = d / "indexer_qsa.py"
s = p.read_text()
a = """        compressed_key_cache = self.compressed_key_cache.kv_cache

"""
assert a in s
s = s.replace(a, """        compressed_key_cache = self.compressed_key_cache.kv_cache
        software_fp8 = self.indexer_dtype == torch.float8_e4m3fn
        if software_fp8:
            compressed_key_cache = compressed_key_cache.view(torch.uint8)

""")
a = """            q = projected_q.new_empty(
                num_tokens,
                self.index_n_heads,
                self.index_head_dim,
                dtype=self.indexer_dtype,
            )
"""
assert a in s
s = s.replace(a, """            q = projected_q.new_empty(
                num_tokens,
                self.index_n_heads,
                self.index_head_dim,
                dtype=torch.bfloat16,
            )
""")
p.write_text(s)
# KUDA_HC_FUSE: optional SM80 HC consumer custom op, applied last.
# Normalization, down projection and injection remain in GatedResidual.
p = root / "vllm/models/qwen4_exp/nvidia/hyperconnection.py"
s = p.read_text()
a = """from .ops.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)
"""
assert s.count(a) == 1
s = s.replace(a, """from .ops.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
)
from .ops.hc_fuse import hc_consumer
""", 1)
a = """        lora = hc_silu(lora, self.hc_count)
        gate = self.input_mix_weight_up(lora)  # [M, D]
        block_input = hc_gate_mix(xn, gate, self.hc_count)
"""
assert s.count(a) == 2
s = s.replace(a, """        block_input = hc_consumer(lora, self.input_mix_weight_up, xn, self.hc_count)
""")
shutil.copy(Path(__file__).with_name("hc_fuse.py"), p.parent / "ops/hc_fuse.py")
p.write_text(s)
print("patched", root)

# KUDA_TOPK_MASK: parallel passes with the pinned Triton search, applied last.
# Do not replace this with sorted top-k: the capped searches have different
# edge semantics. Extract their arithmetic rather than maintaining a fork.
import ast
import textwrap

topk_ops = root / "vllm/v1/sample/ops"
topk_source = (topk_ops / "topk_topp_triton.py").read_text()
topk_ast = ast.parse(topk_source)

def topk_function_source(name):
    matches = [node for node in topk_ast.body
               if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(matches) == 1, name
    node = matches[0]
    first = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return "\n".join(topk_source.splitlines()[first - 1:node.end_lineno]) + "\n"

topk_helper = topk_function_source("_update_min_larger_stats")
topk_kernel = topk_function_source("_topk_topp_kernel")
topk_begin = "                # Zeroth pass: Compute avg and std from a sample block\n"
topk_end = "                # Second passes: Ternary search for pivot\n"
topk_branch = "            if k < VOCAB_SIZE:\n"
assert topk_kernel.count(topk_branch + topk_begin) == topk_kernel.count(topk_end) == 1
topk_first = topk_kernel.index(topk_branch + topk_begin) + len(topk_branch)
topk_last = topk_kernel.index(topk_end)
topk_prefill = topk_kernel[topk_first:topk_last]
topk_seed_end = "                num_outliers = tl.zeros((), dtype=tl.uint32)\n"
assert topk_prefill.count(topk_seed_end) == 1
topk_seed_body = topk_prefill[:topk_prefill.index(topk_seed_end)]
topk_seed = """@triton.jit
def _topkmask_seed_kernel(
    LOGITS, LOGITS_STRIDE_0, PERCENTILE_TO_STD_TABLE, K, SEED,
    VOCAB_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    LOGITS_ROW = LOGITS + row_id.to(tl.int64) * LOGITS_STRIDE_0
    k = tl.load(K + row_id)
    outlier_pivot = -float("inf")
    if k > 0 and k < VOCAB_SIZE:
"""
topk_seed += textwrap.indent(textwrap.dedent(topk_seed_body), "        ")
topk_seed += "    tl.store(SEED + row_id, outlier_pivot)\n"

topk_fast = """                if tl.load(ROW_META + row_id * 5) != 0:
                    min_logit = tl.load(ROW_META + row_id * 5 + 1).to(tl.float32, bitcast=True)
                    max_logit = tl.load(ROW_META + row_id * 5 + 2).to(tl.float32, bitcast=True)
                    num_finite_total = tl.load(ROW_META + row_id * 5 + 3).to(tl.uint32)
                    num_outliers = tl.load(ROW_META + row_id * 5 + 4).to(tl.uint32)
                    outlier_pivot = tl.load(SEED + row_id)
                else:
"""
topk_fast += textwrap.indent(topk_prefill, "    ")
topk_kernel = topk_kernel[:topk_first] + topk_fast + topk_kernel[topk_last:]
assert topk_kernel.count("def _topk_topp_kernel(") == 1
topk_kernel = topk_kernel.replace("def _topk_topp_kernel(", "def _topkmask_pivot_kernel(", 1)
a = "    BATCH_SIZE,\n"
assert topk_kernel.count(a) == 1
topk_kernel = topk_kernel.replace(a, a + "    ROW_META,\n    SEED,\n    DECISIONS,\n", 1)
a = "        BUFFER_ROW = BUFFER + pid * VOCAB_SIZE\n"
assert topk_kernel.count(a) == 1
topk_kernel = topk_kernel.replace(a, "        BUFFER_ROW = BUFFER + row_id * VOCAB_SIZE\n", 1)

# Replace only the final store pass. The exact pivot/boundary state drives the
# parallel CUDA mask; integer tie counts retain their original uint32 bits.
topk_mask_begin = "        # Sixth pass: Apply mask and store final output.\n"
assert topk_kernel.count(topk_mask_begin) == 1
topk_mask_at = topk_kernel.index(topk_mask_begin)
topk_old_mask = topk_kernel[topk_mask_at:]
assert topk_old_mask.count("if not (final_pivot < max_logit):") == 1
assert topk_old_mask.count("elif final_pivot != -float(\"inf\"):") == 1
assert topk_old_mask.count("tl.abs(logits_blk - duplicate_logit) < 1e-9") == 1
assert topk_old_mask.rstrip().endswith("tl.store(LOGITS_ROW + offs_n, logits_blk, mask=mask_n)")
topk_kernel = topk_kernel[:topk_mask_at] + """        active = (final_pivot < max_logit) & (final_pivot != -float("inf"))
        tl.store(DECISIONS + row_id * 5, active.to(tl.int32))
        tl.store(DECISIONS + row_id * 5 + 1, final_pivot.to(tl.int32, bitcast=True))
        tl.store(DECISIONS + row_id * 5 + 2, duplicate_logit.to(tl.int32, bitcast=True))
        tl.store(DECISIONS + row_id * 5 + 3, num_duplicate_logit.to(tl.int32, bitcast=True))
        tl.store(DECISIONS + row_id * 5 + 4, num_keep.to(tl.int32, bitcast=True))
"""

# Fail early if upstream changes the assumed dispatch/split protocol.
assert "block_size, block_size_trunc, num_warps = 8192, 4096, 8" in topk_source
for topk_constant in ("_SPLIT_MAX_BATCH = 64", "_SPLIT_MAX_SPLITS = 32",
                      "_SPLIT_FANOUT = 8", "_SPLIT_ROUNDS = 5"):
    assert topk_constant in topk_source
topk_generated = (
    "# SPDX-License-Identifier: Apache-2.0\n"
    "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
    "# Generated by KUDA_TOPK_MASK apply.py; arithmetic extracted from the pinned reference.\n"
    "from vllm.triton_utils import tl, triton\n\n"
    + topk_helper + "\n" + topk_seed + "\n" + topk_kernel
)
compile(topk_generated, "topk_mask_reference.py", "exec")
(topk_ops / "topk_mask_reference.py").write_text(topk_generated)
shutil.copy(Path(__file__).with_name("topk_mask.py"), topk_ops / "topk_mask.py")

# Both legacy and GPU-worker speculative verification reach this function.
# Other native callers at the same choke point inherit the same guards; the
# FlashInfer direct-sampling route is untouched.
p = topk_ops / "topk_topp_sampler.py"
s = p.read_text()
a = """        apply_top_k_top_p_triton,
    )
"""
assert s.count(a) == 1
s = s.replace(a, a + "    from vllm.v1.sample.ops.topk_mask import apply_topk_mask\n", 1)
a = """    if HAS_TRITON:
        return apply_top_k_top_p_triton(logits, k, p)
"""
assert s.count(a) == 1
s = s.replace(a, """    if HAS_TRITON:
        return apply_topk_mask(logits, k, p)
""", 1)
p.write_text(s)
print("patched exact-compatible top-k mask", root)

# Prefix-cache CoW: bounded device copies for disjoint contiguous blocks.
p = root / "vllm/v1/worker/utils.py"
s = p.read_text()
a = "from vllm.config import CacheConfig, VllmConfig\n"
assert s.count(a) == 1
s = s.replace(a, "from vllm import _custom_ops as ops\n" + a, 1)
a = "    indices_np = np.array(kv_cache_block_copies, dtype=np.int64)\n"
assert s.count(a) == 1
s = s.replace(a, a + """    source_ids = set(indices_np[:, 0])
    destination_ids = set(indices_np[:, 1])
    direct = (indices_np.shape[1] == 2 and indices_np.min() >= 0
              and indices_np.max() < num_blocks
              and source_ids.isdisjoint(destination_ids)
              and len(destination_ids) == len(kv_cache_block_copies))
""", 1)
a = """        if indices is None:
            indices = async_tensor_h2d(indices_np, device=cache.device)
        assert cache.device == indices.device
        src, dst = indices.unbind(dim=1)

"""
assert s.count(a) == 1
s = s.replace(a, "", 1)
a = "        blocks[dst] = blocks[src]\n"
assert s.count(a) == 1
s = s.replace(a, """        if direct and blocks.is_cuda and blocks.is_contiguous():
            # A batched gather materializes every source block. On the hybrid
            # cache a 28-block fork needs 1.15 GiB, even with little KV in use.
            # CPU pointer metadata plus ordered DMA copies need no GPU staging.
            block_bytes = blocks.stride(0) * blocks.element_size()
            base = blocks.data_ptr()
            source_ptrs = torch.from_numpy(base + indices_np[:, 0] * block_bytes)
            destination_ptrs = torch.from_numpy(base + indices_np[:, 1] * block_bytes)
            sizes = torch.full((len(indices_np),), block_bytes, dtype=torch.int64, device="cpu")
            with torch.cuda.device(cache.device):
                ops.swap_blocks_batch(source_ptrs, destination_ptrs, sizes)
        else:
            # Preserve simultaneous-read semantics for chains/cycles and the
            # existing strided/virtual-block behavior. Direct DMA is unsafe here.
            if indices is None:
                indices = async_tensor_h2d(indices_np, device=cache.device)
            assert cache.device == indices.device
            src, dst = indices.unbind(dim=1)
            blocks[dst] = blocks[src]
""", 1)
p.write_text(s)
print("patched bounded prefix-cache CoW copies", root)
