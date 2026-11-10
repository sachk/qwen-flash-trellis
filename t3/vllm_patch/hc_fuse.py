"""Optional SM80 HC consumer fusion; set KUDA_HC_FUSE=1 before worker startup.

KUDA_HC_FUSE_LIB defaults to libhcfuse.so on the library path, built from
hcfuse.cu without HC_V1/HC_V2/HC_V3. Shape/layout/method mismatches retain the
original module call, not a substitute GEMM. Library loading happens outside
Dynamo/CUDA graph capture; the native call remains inside an opaque custom op.
"""
import ctypes
import os
from functools import cache

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.utils.torch_utils import direct_register_custom_op

from .hc import hc_gate_mix, hc_silu

logger = init_logger(__name__)
_LIB_PATH = os.environ.get("KUDA_HC_FUSE_LIB", "libhcfuse.so")
_library = None
_forward = None
if os.environ.get("KUDA_HC_FUSE") == "1":
    try:
        _library = ctypes.CDLL(_LIB_PATH)
        _forward = _library.hc_fuse_forward
        _forward.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4
        _forward.restype = ctypes.c_int
    except (OSError, AttributeError) as error:
        logger.warning("KUDA_HC_FUSE: cannot load %s; keeping the three-op path: %s", _LIB_PATH, error)
        _forward = None
_AVAILABLE = _forward is not None
_logged_fused = False


@torch.compiler.assume_constant_result
@cache
def _is_sm80(device_index: int) -> bool:
    # Device capability is constant per worker/device. Never initialize CUDA
    # during module import or inspect tensor values to make the dispatch choice.
    return torch.cuda.get_device_capability(device_index) == (8, 0)


def _hc_fuse(lora: torch.Tensor, weight: torch.Tensor, xn: torch.Tensor) -> torch.Tensor:
    global _logged_fused
    if _forward is None:
        raise RuntimeError("HC fusion custom op called without an available library")
    with torch.cuda.device(xn.device):
        out = xn.new_empty((xn.shape[0], 2560))
        status = _forward(
            ctypes.c_void_p(torch.cuda.current_stream(xn.device).cuda_stream),
            *[ctypes.c_void_p(t.data_ptr()) for t in (lora, weight, xn, out)],
            xn.shape[0], lora.stride(0), xn.stride(0), out.stride(0),
        )
    if status:
        # A launch failure is not an eligibility miss; do not mask CUDA errors
        # or retry on a stream that may already have failed.
        raise RuntimeError(f"hc_fuse_forward returned cudaError_t={status}")
    if not _logged_fused:
        logger.info("KUDA_HC_FUSE: fused HC consumer taken (library=%s)", _LIB_PATH)
        _logged_fused = True
    return out


def _hc_fuse_fake(lora: torch.Tensor, weight: torch.Tensor, xn: torch.Tensor) -> torch.Tensor:
    return xn.new_empty((xn.shape[0], 2560))


direct_register_custom_op(
    op_name="qwen4_exp_hc_fuse",
    op_func=_hc_fuse,
    fake_impl=_hc_fuse_fake,
)


def hc_consumer(lora: torch.Tensor, up: ReplicatedLinear, xn: torch.Tensor, hc_count: int) -> torch.Tensor:
    # All guards are Python configuration or tensor metadata. Dynamo can
    # specialize the <=8 row range; neither guards nor the fake op read data.
    weight = getattr(up, "weight", None)
    if (
        _AVAILABLE
        and not envs.VLLM_BATCH_INVARIANT
        and hc_count == 4
        and type(up) is ReplicatedLinear
        and type(up.quant_method) is UnquantizedLinearMethod
        and up.bias is None
        and not up.return_bias
        and isinstance(weight, torch.Tensor)
        and lora.ndim == xn.ndim == weight.ndim == 2
        and lora.dtype == xn.dtype == weight.dtype == torch.bfloat16
        and xn.is_cuda
        and lora.device == xn.device == weight.device
        and 1 <= xn.shape[0] <= 8
        and lora.shape == (xn.shape[0], 320)
        and xn.shape[1] == 10240
        and weight.shape == (10240, 320)
        and weight.is_contiguous()
        and weight.storage_offset() % 8 == 0  # cp.async needs 16-byte alignment
        and lora.stride(1) == xn.stride(1) == 1
        and 320 <= lora.stride(0) <= 2147483647
        and 10240 <= xn.stride(0) <= 2147483647
        and not (torch.is_grad_enabled() and (lora.requires_grad or xn.requires_grad or weight.requires_grad))
        and _is_sm80(xn.device.index)
    ):
        return torch.ops.vllm.qwen4_exp_hc_fuse(lora, weight, xn)
    return hc_gate_mix(xn, up(hc_silu(lora, hc_count)), hc_count)
