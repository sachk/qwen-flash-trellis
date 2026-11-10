#!/usr/bin/env python3
"""S3.1 head: GPTQ-rounded INT8 per-row head, written back as BF16 q*s for sm80_head.py."""
import argparse
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from requant_common import FP8_SRC, LIVE, ROOT, Checkpoint, Job

DEVICE = "cuda"
ROW_CHUNK = 32768


def startup_packer(weight):
    """The integers vLLM's sm80_head.py derives from a BF16 head at startup."""
    out = []
    for start in range(0, weight.shape[0], 1024):
        rows = weight[start:start + 1024].float()
        scale = (rows.abs().amax(-1) / 127).clamp_min(1e-12).to(weight.dtype)
        out.append((rows / scale.unsqueeze(-1).float()).round().clamp(-127, 127).to(torch.int16))
    return torch.cat(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--s2", default=str(ROOT / "s2"))
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    job = Job("s3-head")
    source = Checkpoint(FP8_SRC if (FP8_SRC / Checkpoint(FP8_SRC).weight_map["lm_head.weight"]).exists() else LIVE)
    head = source.read("lm_head.weight", DEVICE).float()
    with safe_open(f"{args.s2}/head_hessian.safetensors", framework="pt") as f:
        H = f.get_tensor("h_head").to(DEVICE).double()
    H = H / H.diagonal().mean()
    H.diagonal().add_(0.01)
    upper = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True).float()
    K = head.shape[1]
    result = torch.empty(head.shape, dtype=torch.bfloat16, device="cpu")
    numerator, denominator = 0.0, 0.0
    for start in range(0, head.shape[0], ROW_CHUNK):
        W0 = head[start:start + ROW_CHUNK]
        W = W0.clone()
        amax, peak = W0.abs().max(1)
        scale = (amax / 127).clamp_min(1e-12).to(torch.bfloat16).float().unsqueeze(1)
        Q = torch.empty_like(W)
        for i1 in range(0, K, 128):
            i2 = min(i1 + 128, K)
            R1 = W[:, i1:i2].clone()
            U11 = upper[i1:i2, i1:i2]
            E1 = torch.zeros_like(R1)
            for i in range(i2 - i1):
                q = (R1[:, i] / scale[:, 0]).round().clamp(-127, 127)
                Q[:, i1 + i] = q
                error = (R1[:, i] - q * scale[:, 0]) / U11[i, i]
                E1[:, i] = error
                R1[:, i:] -= error.unsqueeze(1) * U11[i, i:].unsqueeze(0)
            if i2 < K:
                W[:, i2:] -= E1 @ upper[i1:i2, i2:]
        # A rounded-down peak would make the packer re-derive a different scale for the row.
        rows = torch.arange(W0.shape[0], device=DEVICE)
        Q[rows, peak] = 127 * W0[rows, peak].sign()
        delta = Q * scale - W0
        numerator += float(((delta @ H.float()) * delta).sum())
        denominator += float(((W0 @ H.float()) * W0).sum())
        result[start:start + ROW_CHUNK] = (Q * scale).to(torch.bfloat16).cpu()
        if job.stop:
            job.log("stopped; head restarts on resume")
            raise SystemExit(0)
    derived = startup_packer(result.to(DEVICE))
    expected_scale = (head.abs().amax(1) / 127).clamp_min(1e-12).to(torch.bfloat16).float().unsqueeze(1)
    expected = (result.float().to(DEVICE) / expected_scale).round()
    mismatch = float((derived.float() != expected).float().mean())
    error = (numerator / denominator) ** 0.5
    plain = startup_packer(head.to(torch.bfloat16))
    plain_delta = plain.float() * (head.abs().amax(1) / 127).unsqueeze(1) - head
    plain_error = (float(((plain_delta @ H.float()) * plain_delta).sum()) / denominator) ** 0.5
    job.log(f"weighted head error: GPTQ {error:.5f} vs plain rounding {plain_error:.5f}; integer mismatch {mismatch:.2e}")
    save_file({"lm_head.weight": result}, job.dir / "lm_head.safetensors")
    job.done({"gptq_error": error, "plain_error": plain_error, "packer_mismatch": mismatch})


if __name__ == "__main__":
    main()
