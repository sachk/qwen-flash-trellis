"""Packs BF16 weights into gemm_bf16's layout (see t3moe.cu). The packed copy replaces the original, so VRAM doesn't grow."""
import torch


class Layout:
    def __init__(self, N, K):
        self.N, self.K = N, K
        self.rts = (N + 15) // 16
        self.kps = (K + 127) // 128


def pack(w, layout, chunk=64):
    """[row tile, packet, lane, P, half] as BF16, zero outside the matrix."""
    device = w.device
    lane = torch.arange(32, device=device).view(-1, 1, 1)
    P = torch.arange(32, device=device).view(1, -1, 1)
    half = torch.arange(2, device=device).view(1, 1, -1)
    row_in = (lane >> 2) + 8 * (P & 1) + 0 * half
    col_in = 16 * (P >> 2) + 2 * (lane & 3) + 8 * ((P >> 1) & 1) + half
    out = torch.zeros(layout.rts, layout.kps, 32, 32, 2, dtype=torch.bfloat16, device=device)
    for start in range(0, layout.rts, chunk):
        rts = torch.arange(start, min(start + chunk, layout.rts), device=device).view(-1, 1, 1, 1, 1)
        kps = torch.arange(layout.kps, device=device).view(1, -1, 1, 1, 1)
        row = rts * 16 + row_in[None, None]
        col = kps * 128 + col_in[None, None]
        valid = (row < layout.N) & (col < layout.K)
        out[rts[:, 0, 0, 0, 0]] = torch.where(valid, w[row.clamp_max(layout.N - 1), col.clamp_max(layout.K - 1)], torch.zeros((), dtype=torch.bfloat16, device=device))
    return out.flatten()
