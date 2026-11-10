#!/usr/bin/env python3
"""Check hcfuse against the existing vLLM hc_silu -> F.linear -> hc_gate_mix.

Run through the Nix runtime (does execute GPU work):
  t3 python t3/hc_check.py --checkpoint /path/to/model --weights 8 --bench
  t3 python t3/hc_check.py --checkpoint /path/to/model --weights 16 \
      --export-weights hc-up.bf16

The optional raw export feeds hc-bench --weights without packing/transposing.
Default tolerance: atol=2^-9, rtol=2^-7 (one BF16 ULP plus a cancellation floor
for normalized inputs). This is not a bit-exact contract: cuBLAS may split K,
and CUDA/Triton sigmoid approximations can differ at BF16 rounding boundaries.
"""
import argparse
import ctypes
import json
import math
import sys
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib", type=Path, default=Path(os.environ["KUDA_HC_FUSE_LIB"]))
    parser.add_argument("--vllm-path", type=Path, default=Path(os.environ["T3_RUNTIME"]) / "src")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--weights", type=int, default=8)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--export-weights", type=Path)
    parser.add_argument("--bench", action="store_true", help="graph-time the literal vLLM three-op reference and fused op")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--atol", type=float, default=1 / 512)
    parser.add_argument("--rtol", type=float, default=1 / 128)
    args = parser.parse_args()
    if args.weights < 1 or args.iters < 1 or any(m < 1 for m in args.rows):
        parser.error("weights, iters and rows must be positive")
    if args.atol < 0 or args.rtol < 0:
        parser.error("tolerances must be nonnegative")

    import torch
    import torch.nn.functional as F

    sys.path.insert(0, str(args.vllm_path))
    from vllm.models.qwen4_exp.nvidia.ops.hc import hc_gate_mix, hc_silu

    torch.set_grad_enabled(False)
    torch.manual_seed(3821)
    H, HC, K = 2560, 4, 320
    D = H * HC
    names, host_weights = [], []
    if args.checkpoint is not None:
        from safetensors import safe_open

        index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        names = sorted(n for n in index if n.endswith("input_mix_weight_up.weight"))[:args.weights]
        if not names:
            raise ValueError("checkpoint has no input_mix_weight_up.weight tensors")
        for name in names:
            with safe_open(args.checkpoint / index[name], framework="pt", device="cpu") as shard:
                weight = shard.get_tensor(name)
            if weight.dtype != torch.bfloat16 or tuple(weight.shape) != (D, K):
                raise ValueError(f"{name}: expected BF16 {(D, K)}, got {weight.dtype} {tuple(weight.shape)}")
            host_weights.append(weight.contiguous())
    else:
        names = [f"synthetic-{i}" for i in range(args.weights)]
        host_weights = [(torch.randn(D, K) / math.sqrt(K)).bfloat16() for _ in names]
    if args.export_weights is not None:
        with args.export_weights.open("wb") as output:
            for weight in host_weights:
                output.write(weight.view(torch.uint16).numpy().tobytes())
        print(f"exported {len(host_weights)} raw BF16 [10240,320] matrices to {args.export_weights}", flush=True)
    weights = [w.cuda() for w in host_weights]
    del host_weights

    lib = ctypes.CDLL(str(args.lib.resolve()))
    lib.hc_fuse_forward.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4
    lib.hc_fuse_forward.restype = ctypes.c_int

    def fused(lora, weight, xn, out):
        status = lib.hc_fuse_forward(
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
            *[ctypes.c_void_p(t.data_ptr()) for t in (lora, weight, xn, out)],
            lora.shape[0], lora.stride(0), xn.stride(0), out.stride(0),
        )
        if status:
            raise RuntimeError(f"hc_fuse_forward returned cudaError_t={status}")
        return out

    def reference(lora, weight, xn):
        return hc_gate_mix(xn, F.linear(hc_silu(lora, HC), weight), HC)

    def inputs(rows, stride, scale=4.0):
        # Poison injection/padding columns: the fused op must read only rank=320.
        storage = torch.full((rows, stride), float("nan"), dtype=torch.bfloat16, device="cuda")
        storage[:, :K] = torch.randn(rows, K, device="cuda") * scale
        lora = storage[:, :K]
        pad = 16 if stride != K else 0
        xs = torch.full((rows, D + pad), float("nan"), dtype=torch.bfloat16, device="cuda")
        xs[:, :D] = torch.randn(rows, D, device="cuda")
        ys = torch.full((rows, H + pad), float("nan"), dtype=torch.bfloat16, device="cuda")
        return lora, xs[:, :D], ys[:, :H], ys[:, H:]

    def compare(lora, weight, xn, out, guard, label):
        expected = reference(lora, weight, xn)
        actual = fused(lora, weight, xn, out)
        torch.testing.assert_close(actual, expected, atol=args.atol, rtol=args.rtol, equal_nan=False, msg=label)
        if not torch.isnan(guard).all().item():
            raise AssertionError(f"{label}: output padding overwritten")
        delta = actual.float() - expected.float()
        return (
            delta.abs().max().item(),
            (delta.norm() / expected.float().norm().clamp_min(1e-12)).item(),
            (actual == expected).float().mean().item(),
        )

    def capture(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        return graph

    def graph_us(graph, count):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            graph.replay()
        stop.record()
        stop.synchronize()
        return start.elapsed_time(stop) * 1000 / args.iters / count

    print(json.dumps({"weights": names, "atol": args.atol, "rtol": args.rtol,
                      "torch_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}), flush=True)
    for rows in args.rows:
        for stride in (K, 336):
            cases = [inputs(rows, stride) for _ in weights]
            errors = [compare(l, w, x, y, guard, f"M={rows}, stride={stride}, {name}")
                      for (l, x, y, guard), w, name in zip(cases, weights, names)]
            record = dict(M=rows, lora_stride=stride, max_abs=max(e[0] for e in errors),
                          max_relative_l2=max(e[1] for e in errors), min_exact_fraction=min(e[2] for e in errors))
            if args.bench and stride == 336:
                # Use real contiguous xn/output layouts for timing, independent of the strided checks.
                bench_cases = [(l, x.contiguous(), torch.empty(rows, H, dtype=torch.bfloat16, device="cuda"))
                               for l, x, _, _ in cases]
                def baseline():
                    for (l, x, _), w in zip(bench_cases, weights):
                        reference(l, w, x)
                def candidate():
                    for (l, x, y), w in zip(bench_cases, weights):
                        fused(l, w, x, y)
                bg, fg = capture(baseline), capture(candidate)
                for _ in range(5):
                    bg.replay()
                    fg.replay()
                bt, ft = [], []
                for trial in range(5):
                    if trial & 1:
                        ft.append(graph_us(fg, len(weights)))
                        bt.append(graph_us(bg, len(weights)))
                    else:
                        bt.append(graph_us(bg, len(weights)))
                        ft.append(graph_us(fg, len(weights)))
                record.update(three_op_us=sorted(bt)[2], fused_us=sorted(ft)[2])
                del bg, fg
            print(json.dumps(record), flush=True)

    # Exercise the 8->16 row-tile tail, zero gates, and sigmoid saturation.
    for scale in (0.0, 80.0):
        lora, xn, out, guard = inputs(9, 336, scale)
        compare(lora, weights[0], xn, out, guard, f"M=9, scale={scale}")
    torch.cuda.synchronize()
    print("PASS: actual vLLM HC ops matched within tolerance; stride guards and row tails intact", flush=True)


if __name__ == "__main__":
    main()
