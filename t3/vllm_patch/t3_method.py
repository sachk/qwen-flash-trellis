"""Routed experts in the ring-coded trellis packet format, run by libt3moe.so on SM80.

Active when KUDA_T3MOE points at a model directory with t3layers.json. Target-model layers only: the MTP draft
keeps its own path. Shared experts stay with the runner, because this method reports no internal MK.
"""
import ctypes
import json
import os
from collections import Counter
import re
from pathlib import Path

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.t3gemm import Layout, pack
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.utils.torch_utils import current_stream, direct_register_custom_op

logger = init_logger(__name__)
LAYER = re.compile(r"(?:^|\.)model\.layers\.(\d+)\.mlp")


def t3_layer_index(prefix: str) -> int | str | None:
    if not os.environ.get("KUDA_T3MOE"):
        return None
    if "mtp" in prefix:
        return "mtp" if os.environ.get("KUDA_T3MTP") else None
    match = LAYER.search(prefix)
    return int(match.group(1)) if match else None


class _Runtime:
    """Library plus ordered scratch shared by all target/draft expert layers."""

    lib = None
    tensors = None
    capacity = 0

    @classmethod
    def library(cls):
        if cls.lib is None:
            path = os.environ.get("KUDA_T3MOE_LIB", "libt3moe.so")
            lib = ctypes.CDLL(path)
            lib.t3_init()
            lib.t3_forward.argtypes = [ctypes.c_void_p] * 19 + [ctypes.c_int, ctypes.c_void_p, ctypes.c_long]
            cls.lib = lib
            logger.info("t3moe kernels loaded from %s", path)
        return cls.lib

    @classmethod
    def get(cls, device, capacity):
        lib = cls.library()
        if cls.tensors is None:
            rows = capacity * 10 + 512 * 16
            cls.tensors = [
                torch.zeros(512, dtype=torch.int32, device=device),
                torch.zeros(512, dtype=torch.int32, device=device),
                torch.zeros(512, dtype=torch.int32, device=device),
                torch.zeros(rows, dtype=torch.int32, device=device),
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.zeros(rows, dtype=torch.int32, device=device),
                torch.zeros(rows, dtype=torch.float32, device=device),
                torch.zeros((rows, 640), dtype=torch.bfloat16, device=device),
                torch.zeros((capacity, 2560), dtype=torch.float32, device=device),
            ]
            cls.capacity = capacity
            logger.info("t3moe workspace: max_tokens=%d bytes=%d",
                        capacity, sum(t.nbytes for t in cls.tensors))
        elif capacity > cls.capacity:
            # Never invalidate workspace addresses held by captured graphs.
            raise ValueError(f"t3moe: workspace has {cls.capacity} rows, requested {capacity}")
        return lib, cls.tensors


class T3MoEMethod(FusedMoEMethodBase):
    def __init__(self, moe, layer_index):
        super().__init__(moe)
        self.index = layer_index

    def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype, **extra):
        assert num_experts == 512 and hidden_size == 2560 and intermediate_size_per_partition == 640
        sizes = json.loads((Path(os.environ["KUDA_T3MOE"]) / "t3layers.json").read_text())[str(self.index)]
        layer.num_experts = num_experts
        for name, shape, dtype in (
            ("t3_w13", (sizes["w13"],), torch.uint8), ("t3_w2", (sizes["w2"],), torch.uint8),
            ("t3_off13", (512,), torch.int64), ("t3_off2", (512,), torch.int64),
            ("t3_fmt13", (512,), torch.uint8), ("t3_fmt2", (512,), torch.uint8),
        ):
            setattr(layer, name, torch.nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False))

    def load_weights(self, layer, weights):
        for name, tensor in weights:
            key = name.rsplit(".", 1)[-1]
            if key.startswith("t3_"):       # The draft's old W4 tensors may sit beside these in its shards.
                getattr(layer, key).data.copy_(tensor)
                yield f"{layer.layer_name}.{key}"

    def get_fused_moe_quant_config(self, layer):
        return None

    def process_weights_after_loading(self, layer):
        pass

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        M = x.shape[0]
        capacity = self.moe.max_num_tokens
        if M > capacity:
            raise ValueError(f"t3moe: {M} rows exceeds scheduler capacity {capacity}")
        lib, work = _Runtime.get(x.device, capacity)
        if x.stride(0) != 2560:
            x = x.contiguous()
        out = torch.empty((M, 2560), dtype=torch.bfloat16, device=x.device)
        pointers = [layer.t3_w13, layer.t3_w2, layer.t3_off13, layer.t3_off2, layer.t3_fmt13, layer.t3_fmt2, *work, x, topk_ids, topk_weights]
        lib.t3_forward(
            ctypes.c_void_p(int(current_stream().cuda_stream)),
            *[ctypes.c_void_p(t.data_ptr()) for t in pointers],
            ctypes.c_int(M), ctypes.c_void_p(out.data_ptr()), ctypes.c_long(2560),
        )
        return out


# ---------------------------------------------------------------- draft vocabulary
class _DraftHead(torch.nn.Module):
    """The target's INT8 head restricted to the draft vocabulary, so each draft step reads about a quarter of the rows."""

    def __init__(self, weight, quant_method):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.quant_method = quant_method
        self.tp_size = 1
        self.tp_rank = 0
        self.quant_method.process_weights_after_loading(self)

    def forward(self, x):
        return self.quant_method.apply(self, x)


def install_draft_vocab(proposer, target_model) -> None:
    """Rejection sampling stays exact: q is zero outside the table, and p's mass there falls to the residual."""
    path = os.environ.get("KUDA_DRAFT_VOCAB")
    if not path:
        return
    from safetensors import safe_open

    from vllm.models.qwen4_exp.nvidia.sm80_head import Sm80Int8Head

    device = torch.device("cuda")
    table = torch.tensor(json.loads(Path(path).read_text()), dtype=torch.long, device=device)
    model_dir = Path(os.environ["KUDA_T3MOE"])
    shard = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]["lm_head.weight"]
    with safe_open(model_dir / shard, framework="pt") as f:
        full = f.get_tensor("lm_head.weight")
    rows = full[table.cpu()].to(device).to(torch.bfloat16)
    del full
    # KUDA_DRAFT_HEAD_BITS=4 halves the head read on every draft step. Only
    # acceptance can change: rejection sampling is exact for any proposal q.
    head = _int4_head() if os.environ.get("KUDA_DRAFT_HEAD_BITS") == "4" else Sm80Int8Head()
    proposer.model.lm_head = _DraftHead(rows, head)
    proposer.draft_vocab_table = table
    proposer.draft_vocab_size_full = 248320
    logger.info("draft vocabulary: %d rows from %s (%s head)", len(table), path, type(head).__name__)


def _int4_head():
    """Draft head in Marlin W4 (uint4b8), symmetric, group size 128, round to nearest."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_gptq_marlin_linear,
        marlin_make_workspace_new,
        marlin_permute_scales,
    )
    from vllm.models.qwen4_exp.nvidia.sm80_head import Sm80Int8Head
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types

    class Int4Head(Sm80Int8Head):
        group = 128

        @torch.no_grad()
        def process_weights_after_loading(self, layer):
            weight = layer.weight
            n, k = weight.shape
            self.size_n, self.size_k = n, k
            assert k % self.group == 0 and n % 64 == 0, (n, k)
            packed = torch.empty((k // 8, n), device=weight.device, dtype=torch.int32)
            scales = torch.empty((k // self.group, n), device=weight.device, dtype=weight.dtype)
            shifts = torch.arange(0, 32, 4, device=weight.device, dtype=torch.int32)
            for start in range(0, n, 1024):
                rows = weight[start:start + 1024].float()
                grouped = rows.view(rows.shape[0], k // self.group, self.group)
                scale = (grouped.abs().amax(-1) / 7).clamp_min(1e-12).to(weight.dtype)
                codes = (grouped / scale[..., None].float()).round().clamp(-8, 7).add_(8).to(torch.int32)
                words = (codes.view(rows.shape[0], k // 8, 8) << shifts).sum(-1, dtype=torch.int32)
                packed[:, start:start + rows.shape[0]] = words.T
                scales[:, start:start + rows.shape[0]] = scale.T
            device = weight.device
            del layer.weight
            repacked = ops.gptq_marlin_repack(b_q_weight=packed, size_k=k, size_n=n, num_bits=4)
            self.scales = marlin_permute_scales(scales, k, n, self.group)
            self.workspace = marlin_make_workspace_new(device, 4)
            self.zero_points = torch.empty(0, device=device, dtype=torch.int32)
            layer.weight = torch.nn.Parameter(repacked, requires_grad=False)

        def apply(self, layer, x, bias=None):
            if bias is not None:
                raise ValueError("The INT4 draft head does not support bias")
            return apply_gptq_marlin_linear(
                x, layer.weight, self.scales, self.zero_points, self.workspace,
                scalar_types.uint4b8, self.size_n, self.size_k,
            )

    return Int4Head()


# ---------------------------------------------------------------- INT8 for the small BF16 tensors
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod


class _Int8Embedding(UnquantizedEmbeddingMethod):
    """Per-row INT8 embedding. One row per token, dequantised on the fly."""

    def process_weights_after_loading(self, layer):
        weight = layer.weight
        rows = weight.shape[0]
        codes = torch.empty(weight.shape, dtype=torch.int8, device=weight.device)
        scales = torch.empty(rows, dtype=torch.float32, device=weight.device)
        for start in range(0, rows, 8192):
            part = weight[start:start + 8192].float()
            scale = (part.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-12)
            codes[start:start + 8192] = (part / scale).round().clamp(-127, 127).to(torch.int8)
            scales[start:start + 8192] = scale[:, 0]
        del layer.weight
        layer.t3_codes = torch.nn.Parameter(codes, requires_grad=False)
        layer.t3_scales = torch.nn.Parameter(scales, requires_grad=False)

    def embedding(self, layer, ids):
        rows = layer.t3_codes[ids].float() * layer.t3_scales[ids].unsqueeze(-1)
        return rows.to(torch.bfloat16)


def _padded_int8():
    """The head packer for shapes Marlin can't tile: zero rows up to a multiple of 64, sliced off after."""
    from vllm.models.qwen4_exp.nvidia.sm80_head import Sm80Int8Head

    class Padded(Sm80Int8Head):
        def process_weights_after_loading(self, layer):
            weight = layer.weight
            self.size_true = weight.shape[0]
            pad = (-self.size_true) % 64
            if pad:
                layer.weight = torch.nn.Parameter(
                    torch.cat([weight.data, torch.zeros(pad, weight.shape[1], dtype=weight.dtype, device=weight.device)]), requires_grad=False)
            super().process_weights_after_loading(layer)

        def apply(self, layer, x, bias=None):
            return super().apply(layer, x, bias)[..., : self.size_true]

    return Padded()


def apply_int8_layers(model) -> None:
    """Selected by KUDA_INT8, a comma list of embed, hc, shared. Uses the same startup packer as the output head."""
    wanted = set(filter(None, os.environ.get("KUDA_INT8", "").split(",")))
    if not wanted:
        return
    count = 0
    for name, module in model.named_modules():
        if "embed" in wanted and name.endswith("embed_tokens") and hasattr(module, "quant_method"):
            module.quant_method = _Int8Embedding()
            count += 1
        elif "hc" in wanted and name.rsplit(".", 1)[-1] in ("input_mix_weight_up", "input_mix_weight_down_block_inject", "input_mix_weight_down"):
            module.quant_method = _padded_int8()
            count += 1
        elif "shared" in wanted and ".shared_expert." in name and name.rsplit(".", 1)[-1] in ("gate_up_proj", "down_proj"):
            module.quant_method = _padded_int8()
            count += 1
    logger.info("INT8 packing on %d layers: %s", count, sorted(wanted))


# ---------------------------------------------------------------- skinny BF16 multiply
def _t3_gemm(x: torch.Tensor, packed: torch.Tensor, rts: int, kps: int, K: int, N: int) -> torch.Tensor:
    lib = _Runtime.library()
    shape = x.shape
    if x.dim() != 2:
        x = x.reshape(-1, K)
    if x.stride(0) != K:
        x = x.contiguous()
    y = torch.empty((x.shape[0], N), dtype=torch.bfloat16, device=x.device)
    lib.t3_gemm(
        ctypes.c_void_p(int(current_stream().cuda_stream)), ctypes.c_void_p(packed.data_ptr()),
        ctypes.c_void_p(x.data_ptr()), ctypes.c_long(K), ctypes.c_void_p(y.data_ptr()), ctypes.c_long(N),
        ctypes.c_int(rts), ctypes.c_int(kps), ctypes.c_int(K), ctypes.c_int(N), ctypes.c_int(x.shape[0]),
    )
    return y.reshape(*shape[:-1], N)


def _t3_gemm_fake(x: torch.Tensor, packed: torch.Tensor, rts: int, kps: int, K: int, N: int) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], N))


direct_register_custom_op(op_name="t3_gemm", op_func=_t3_gemm, fake_impl=_t3_gemm_fake)


class _T3Gemm(UnquantizedLinearMethod):
    """A skinny BF16 projection on SM80, in one launch. The packed weights replace the originals, so VRAM doesn't grow."""

    def process_weights_after_loading(self, layer):
        weight = layer.weight.data
        self.layout = Layout(weight.shape[0], weight.shape[1])
        packed = pack(weight, self.layout)
        del layer.weight
        layer.t3_packed = torch.nn.Parameter(packed, requires_grad=False)

    def apply(self, layer, x, bias=None):
        if bias is not None or x.dtype != torch.bfloat16:
            raise ValueError("t3 gemm takes BF16 input and no bias")
        o = self.layout
        return torch.ops.vllm.t3_gemm(x, layer.t3_packed, o.rts, o.kps, o.K, o.N)


def apply_t3_gemm(model) -> None:
    """KUDA_T3GEMM moves plain BF16 projections off cuBLAS: K at most this value, or 0 for all of them."""
    limit = os.environ.get("KUDA_T3GEMM")
    if not limit:
        return
    limit = int(limit) or 1 << 30
    # Measured 2026-11-07: cuBLAS is faster in the real model on (10240, 320) and (336, 10240).
    least = int(os.environ.get("KUDA_T3GEMM_MIN_K", "0"))
    shapes = Counter()
    for name, module in model.named_modules():
        if not isinstance(module, LinearBase) or type(module.quant_method) is not UnquantizedLinearMethod:
            continue
        weight = getattr(module, "weight", None)
        if weight is None or weight.dim() != 2 or weight.shape[0] < 16 or weight.shape[1] % 8 or not least <= weight.shape[1] <= limit:
            continue
        module.quant_method = _T3Gemm()
        shapes[tuple(weight.shape)] += 1
    logger.info("t3 gemm on %d layers: %s", sum(shapes.values()), dict(shapes))


# ---------------------------------------------------------------- draft sampling
_DRAFT_TOPK = int(os.environ.get("KUDA_DRAFT_TOPK", "0"))
_DRAFT_TOPP = float(os.environ.get("KUDA_DRAFT_TOPP", "1"))


def filter_draft_logits(logits, idx_mapping, temperature):
    """Top-k then top-p on draft logits, in the target's order. KUDA_DRAFT_TOPK=0 leaves them alone.

    The switch is fixed for the server's life, for testing against traffic that uses these settings. For mixed clients it
    should take per-request values from SamplingStates. Exactness doesn't depend on it: a rejected draft is resampled from
    the residual, so q may be zero anywhere.
    """
    if not _DRAFT_TOPK:
        return logits
    values, _ = logits.topk(_DRAFT_TOPK, dim=-1)
    floor = values[:, -1:]
    if _DRAFT_TOPP < 1:
        rows = idx_mapping[: logits.shape[0]].clamp_min(0)
        scale = temperature[rows].clamp_min(1e-4).unsqueeze(-1)
        probabilities = torch.softmax(values.float() / scale, dim=-1)
        # Keep each token whose preceding mass is under p, so the set always covers at least p.
        keep = probabilities.cumsum(-1) - probabilities < _DRAFT_TOPP
        floor = values.masked_fill(~keep, float("inf")).amin(-1, keepdim=True)
    return logits.masked_fill(logits < floor, float("-inf"))
