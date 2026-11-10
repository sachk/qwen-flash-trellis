"""Static INT8 output-head packing for the pinned SM80 deployment.

The checkpoint is unchanged. Packing happens once, before KV allocation and
CUDA graph capture; target and MTP share the resulting head. Activations and
outputs remain BF16. This does not change FP8 KV or the BF16 SSD PLE table.
"""

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_gptq_marlin_linear,
    marlin_make_workspace_new,
    marlin_permute_scales,
)
from vllm.scalar_type import scalar_types


class Sm80Int8Head(LinearMethodBase):
    def create_weights(self, *args, **kwargs):
        return UnquantizedLinearMethod().create_weights(*args, **kwargs)

    @torch.no_grad()
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight
        self.size_n, self.size_k = weight.shape
        # Work a bounded row tile at a time: never allocate a full FP32 head.
        packed = torch.empty(
            (self.size_k // 4, self.size_n),
            device=weight.device,
            dtype=torch.int32,
        )
        scales = torch.empty(self.size_n, device=weight.device, dtype=weight.dtype)
        for start in range(0, self.size_n, 1024):
            rows = weight[start : start + 1024].float()
            scale = (rows.abs().amax(-1) / 127).clamp_min(1e-12).to(weight.dtype)
            codes = (
                (rows / scale[:, None].float())
                .round()
                .clamp(-127, 127)
                .add_(128)
                .to(torch.uint8)
            )
            packed[:, start : start + rows.shape[0]] = codes.view(torch.int32).T
            scales[start : start + rows.shape[0]] = scale
        device = weight.device
        del layer.weight
        del weight, rows, codes
        repacked = ops.gptq_marlin_repack(
            b_q_weight=packed,
            size_k=self.size_k,
            size_n=self.size_n,
            num_bits=8,
        )
        self.scales = marlin_permute_scales(
            scales[None, :], self.size_k, self.size_n, -1
        )
        self.workspace = marlin_make_workspace_new(device, 4)
        self.zero_points = torch.empty(0, device=device, dtype=torch.int32)
        layer.weight = torch.nn.Parameter(repacked, requires_grad=False)

    def apply(self, layer, x, bias=None):
        if bias is not None:
            raise ValueError("The SM80 INT8 output head does not support bias")
        return apply_gptq_marlin_linear(
            x,
            layer.weight,
            self.scales,
            self.zero_points,
            self.workspace,
            scalar_types.uint8b128,
            self.size_n,
            self.size_k,
        )
