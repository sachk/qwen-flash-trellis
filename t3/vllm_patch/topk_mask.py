"""KUDA_TOPK_MASK=1: parallel CUDA passes around the pinned Triton search.

Build topkmask.cu, then apply vllm_patch/apply.py to a private vLLM copy; it
also generates topk_mask_reference.py from that copy's asserted source anchors.
KUDA_TOPK_MASK_LIB defaults to libtopkmask.so on the library path. No production
wiring is included. See topkmask.cu for the complete observed mask semantics.

The shared apply_top_k_top_p choke point serves legacy rejection_sampler's
apply_sampling_constraints and GPU rejection_sampler._verify -> Sampler ->
SamplingStates. The supplied tree omits runner selection, so it cannot identify
which runner is instantiated at runtime. Hooking the shared function covers
both without changing the FlashInfer sampling path or probability/RNG code.

Exact compatibility requires preserving reference arithmetic, not sorting.
The generated pivot kernel retains all original searches and exceptional paths;
CUDA supplies ordered outliers only when the reference would use that buffer.
A separate CUDA mask consumes its exact pivot/boundary decisions. P-only rows
use the original split kernels with private workspace. Scratch is allocated per
invocation from PyTorch (graph pool during capture), never cached across streams.
Only lookup tables/device properties are cached, initialized outside capture.
"""
import ctypes
import os

import torch

from vllm.logger import init_logger
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import direct_register_custom_op
from . import topk_topp_triton as reference
from .topk_mask_reference import _topkmask_seed_kernel, _topkmask_pivot_kernel

logger = init_logger(__name__)
_LIB_PATH = os.environ.get("KUDA_TOPK_MASK_LIB", "libtopkmask.so")
_library = None
_gather = _compact = _counts = _mask = None
if os.environ.get("KUDA_TOPK_MASK") == "1":
    try:
        _library = ctypes.CDLL(_LIB_PATH)
        _gather = _library.topkmask_gather
        _gather.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 2 + [ctypes.c_int64]
        _compact = _library.topkmask_compact
        _compact.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 2
        _counts = _library.topkmask_counts
        _mask = _library.topkmask_mask
        for function in (_counts, _mask):
            function.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2 + [ctypes.c_int64]
        for function in (_gather, _compact, _counts, _mask):
            function.restype = ctypes.c_int
    except (OSError, AttributeError) as error:
        logger.warning("KUDA_TOPK_MASK: cannot load %s; retaining Triton: %s", _LIB_PATH, error)
        _gather = _compact = _counts = _mask = None
_AVAILABLE = _gather is not None
_tables = {}
_logged = False


def prepare_topk_mask(device):
    """Call during eager warmup, before capturing the same device's first op."""
    device = torch.device(device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device in _tables:
        return _tables[device]
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("KUDA_TOPK_MASK needs one eager warmup before capture")
        cdf = torch.tensor(reference._NORMAL_CDF_TO_SIGMA_TABLE, dtype=torch.float32, device=device)
        percentile = torch.tensor(reference._PERCENTILE_TO_STD_TABLE, dtype=torch.float32, device=device)
        sms = num_compute_units(device.index)
        capability = torch.cuda.get_device_capability(device)
        # Lookup tables are visible to subsequent captures on other streams.
        torch.cuda.current_stream(device).synchronize()
    _tables[device] = (cdf, percentile, sms, capability)
    return _tables[device]


def _ptr(t):
    return ctypes.c_void_p(0 if t is None else t.data_ptr())


def _launch(function, *args):
    status = function(*args)
    if status:
        raise RuntimeError(f"{function.__name__} returned cudaError_t={status}")


def _run(logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor | None) -> None:
    global _logged
    if not _AVAILABLE:
        raise RuntimeError("top-k mask custom op called without its native library")
    with torch.cuda.device(logits.device):
        cdf, percentile, sms, capability = prepare_topk_mask(logits.device)
        if capability != (8, 0):
            reference.apply_top_k_top_p_triton(logits, k, p)
            return
        rows, vocab = logits.shape
        tiles = (vocab + 8191) // 8192
        seed = logits.new_empty((rows,))
        local = logits.new_empty((rows, tiles, 8192))
        buffer = logits.new_empty((rows, vocab))
        tilemeta = torch.empty((rows, tiles, 5), dtype=torch.int32, device=logits.device)
        rowmeta = torch.empty((rows, 5), dtype=torch.int32, device=logits.device)
        decisions = torch.empty((rows, 5), dtype=torch.int32, device=logits.device)
        counts = torch.empty((rows, tiles), dtype=torch.int32, device=logits.device)
        stream = ctypes.c_void_p(torch.cuda.current_stream(logits.device).cuda_stream)
        _topkmask_seed_kernel[(rows,)](
            logits, logits.stride(0), percentile, k, seed,
            VOCAB_SIZE=vocab, BLOCK_SIZE=8192, num_warps=8,
        )
        _launch(_gather, stream, _ptr(logits), _ptr(k), _ptr(seed), _ptr(local),
                _ptr(tilemeta), rows, vocab, logits.stride(0))
        _launch(_compact, stream, _ptr(k), _ptr(local), _ptr(tilemeta),
                _ptr(buffer), _ptr(rowmeta), rows, vocab)
        _topkmask_pivot_kernel[(rows,)](
            logits, logits.stride(0), buffer, percentile, cdf, k,
            p if p is not None else logits, rows, rowmeta, seed, decisions,
            VOCAB_SIZE=vocab, MASK_VALUE=float("-inf"), BLOCK_SIZE=8192,
            BLOCK_SIZE_TRUNC=4096, TOPK_ENABLED=True, TOPP_ENABLED=p is not None,
            SPLIT_COVERS_PONLY=p is not None, num_warps=8,
        )
        _launch(_counts, stream, _ptr(logits), _ptr(decisions), _ptr(counts),
                rows, vocab, logits.stride(0))
        _launch(_mask, stream, _ptr(logits), _ptr(decisions), _ptr(counts),
                rows, vocab, logits.stride(0))
        if p is not None:
            # Same dispatch/splits/rounds as apply_top_k_top_p_triton, without
            # its global scratch cache (safe for independently captured graphs).
            stats = logits.new_empty((64 * 32, 4))
            parts = logits.new_empty((64, 5, 32, 8, 3))
            reference._topp_split_stats(logits, stats, k, p, sms)
            for round_i in range(5):
                reference._topp_split_step(logits, stats, parts, k, p, round_i, sms)
            reference._topp_split_mask(logits, stats, parts, k, p, float("-inf"), sms)
    if not _logged:
        logger.info("KUDA_TOPK_MASK: parallel exact-compatibility mask taken (library=%s)", _LIB_PATH)
        _logged = True


def _fake(logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor | None) -> None:
    pass  # Mutation-only schema; shape and dtype do not change.


direct_register_custom_op(
    op_name="kuda_topk_mask", op_func=_run, fake_impl=_fake, mutates_args=["logits"],
)


def apply_topk_mask(logits, k, p):
    """Same return/alias contract as apply_top_k_top_p_triton; metadata guards."""
    if (
        _AVAILABLE and logits.is_cuda and logits.dtype == torch.float32
        and logits.ndim == 2 and 1 <= logits.shape[0] <= 64
        and 1 <= logits.shape[1] <= 262144 and logits.stride(1) == 1
        and logits.stride(0) >= logits.shape[1]
        and isinstance(k, torch.Tensor) and k.device == logits.device
        and k.dtype == torch.int32 and k.shape == (logits.shape[0],) and k.is_contiguous()
        and (p is None or (p.device == logits.device and p.dtype == torch.float32
                          and p.shape == (logits.shape[0],) and p.is_contiguous()))
        and not (torch.is_grad_enabled() and logits.requires_grad)
    ):
        torch.ops.vllm.kuda_topk_mask(logits, k, p)
        return logits
    return reference.apply_top_k_top_p_triton(logits, k, p)
