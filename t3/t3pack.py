"""Packets for the ring-coded trellis formats: pack from rounded indices, decode back to weights.

Decoding here follows the kernel's bit operations exactly, so decode(pack(q)) == hat checks the packer, and
the kernels are checked against this decode. Per-layer file:
  L{n}.safetensors: w13, w2 (uint8), off13, off2 (int64 [512]), fmt13, fmt2 (uint8 [512])
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

import t3lib as t

PKT = {2: 544, 3: 800, 4: 1056}
SOFF = {2: 512, 3: 768, 4: 1024}
F = {2: 8, 3: 5, 4: 4}
W = {2: 4, 3: 6, 4: 8}


class Geometry:
    """Row and column of every field: tensors over [rt, kp, lane, word, half, field], low half first."""

    def __init__(self, matrix, bits, device):
        self.matrix, self.bits = matrix, bits
        self.rts, self.kps = (80, 20) if matrix == "w13" else (160, 5)
        f, w = F[bits], W[bits]
        rt = torch.arange(self.rts, device=device).view(-1, 1, 1, 1, 1, 1)
        kp = torch.arange(self.kps, device=device).view(1, -1, 1, 1, 1, 1)
        lane = torch.arange(32, device=device).view(1, 1, -1, 1, 1, 1)
        word = torch.arange(w, device=device).view(1, 1, 1, -1, 1, 1)
        half = torch.arange(2, device=device).view(1, 1, 1, 1, -1, 1)
        field = torch.arange(f, device=device).view(1, 1, 1, 1, 1, -1)
        g, tt = lane >> 2, lane & 3
        if bits == 3:
            P = 10 * (word // 2) + 2 * field + (word & 1)
        else:
            P = f * word + field
        k = kp * 128 + 16 * (P >> 2) + 2 * tt + 8 * ((P >> 1) & 1) + half
        high = P & 1
        if matrix == "w13":
            row = rt * 8 + g + 640 * high
        else:
            row = rt * 16 + g + 8 * high
        self.row, self.col = row, k
        self.P = P
        # Spare pairs are T3's 30, 31.
        if bits == 3:
            spare_P = torch.tensor([30, 31], device=device).view(1, 1, 1, 2, 1)
            rt5, kp5, lane5 = rt[..., 0], kp[..., 0], lane[..., 0]
            half5 = torch.arange(2, device=device).view(1, 1, 1, 1, 2)
            g5, t5 = lane5 >> 2, lane5 & 3
            self.spare_col = kp5 * 128 + 16 * (spare_P >> 2) + 2 * t5 + 8 * ((spare_P >> 1) & 1) + half5
            high5 = spare_P & 1
            self.spare_row = rt5 * 8 + g5 + 640 * high5 if matrix == "w13" else rt5 * 16 + g5 + 8 * high5
        # Which packet follows in the ring.
        if matrix == "w13":
            self.next_kp = torch.tensor([(k_ - 3 if k_ % 4 == 3 else k_ + 1) for k_ in range(20)], device=device)
        else:
            self.next_rt = torch.arange(160, device=device) ^ 1
        # Field shifts within a 32-bit word.
        self.shift = field * bits + 16 * half


_geo = {}


def geometry(matrix, bits, device="cuda"):
    key = (matrix, bits, str(device))
    if key not in _geo:
        _geo[key] = Geometry(matrix, bits, device)
    return _geo[key]


def pack(q, s, matrix, bits):
    """q: int8 [E, R, K], level index (trellis) or scalar q at spare positions. s: scales [E, R, G]. Returns int64 words [E, rt, kp, PKT/4], each holding a uint32."""
    device = q.device
    geo = geometry(matrix, bits, device)
    E = q.shape[0]
    rows, cols = geo.row, geo.col
    idx = q[:, rows, cols].long()                                     # [E, rt, kp, lane, word, half, field]
    # u at word i is c0 of the next position in the ring.
    w = W[bits]
    c0 = idx & 1
    next_c0 = torch.empty_like(c0)
    next_c0[:, :, :, :, :-1] = c0[:, :, :, :, 1:]
    if matrix == "w13":
        next_c0[:, :, :, :, -1] = c0[:, :, geo.next_kp, :, 0]
    else:
        next_c0[:, :, :, :, -1] = c0[:, geo.next_rt, :, :, 0]
    free = idx >> 2                                                    # bits above the two label bits
    field = (free << 1) | next_c0
    words = (field.to(torch.int64) << geo.shift.unsqueeze(0)).sum(-1).sum(-1)     # [E, rt, kp, lane, word]
    if bits == 3:
        sq = q[:, geo.spare_row, geo.spare_col].long()                 # [E, rt, kp, lane, 2 pairs, 2 halves]
        for pair in range(2):
            for b in range(3):
                bit = (sq[:, :, :, :, pair] >> b) & 1
                words[:, :, :, :, pair * 3 + b] |= (bit[..., 0] << 15) | (bit[..., 1] << 31)
    rts, kps = geo.rts, geo.kps
    # Write through a 32-bit view of each packet, in lane layout.
    buf = torch.zeros((E, rts, kps, PKT[bits] // 4), dtype=torch.int64, device=device)
    n_a = min(w, 4)
    for i in range(n_a):
        buf[:, :, :, torch.arange(32, device=device) * 4 + i] = words[:, :, :, :, i]
    if bits == 3:
        base = 512 // 4
        for i in (4, 5):
            buf[:, :, :, base + torch.arange(32, device=device) * 2 + (i - 4)] = words[:, :, :, :, i]
    elif bits == 4:
        base = 512 // 4
        for i in range(4, 8):
            buf[:, :, :, base + torch.arange(32, device=device) * 4 + (i - 4)] = words[:, :, :, :, i]
    # Scales: row g in the low half, row g+8 in the high half, per k-group.
    bf = s.to(torch.bfloat16).view(torch.int16).long() & 0xFFFF         # [E, R, G]
    rt_ = torch.arange(rts, device=device).view(-1, 1, 1)
    kp_ = torch.arange(kps, device=device).view(1, -1, 1)
    g_ = torch.arange(8, device=device).view(1, 1, -1)
    if matrix == "w13":
        row_lo, row_hi = rt_ * 8 + g_, 640 + rt_ * 8 + g_
    else:
        row_lo, row_hi = rt_ * 16 + g_, rt_ * 16 + g_ + 8
    buf[:, :, :, SOFF[bits] // 4 + torch.arange(8, device=device)] = bf[:, row_lo, kp_] | (bf[:, row_hi, kp_] << 16)
    return buf & 0xFFFFFFFF


def decode(packed, matrix, bits, cubic="c2" if os.environ.get("T3_LEVELS") == "c2" else 0.0, device="cuda"):
    """Weights from packets, with the kernel's operations. packed: uint32 [E, rt, kp, PKT/4]. Returns float32 [E, R, K]."""
    if not torch.is_tensor(packed):
        packed = torch.from_numpy(packed.astype(np.int64)).to(device)
    E, rts, kps, _ = packed.shape
    w = W[bits]
    f = F[bits]
    lanes = torch.arange(32, device=device)
    words = []
    for i in range(min(w, 4)):
        words.append(packed[:, :, :, lanes * 4 + i])
    if bits == 3:
        words += [packed[:, :, :, 128 + lanes * 2 + (i - 4)] for i in (4, 5)]
    elif bits == 4:
        words += [packed[:, :, :, 128 + lanes * 4 + (i - 4)] for i in range(4, 8)]
    words = torch.stack(words, 4)                                       # [E, rt, kp, lane, word]
    if matrix == "w13":
        next_kp = torch.tensor([(k - 3 if k % 4 == 3 else k + 1) for k in range(20)], device=device)
        prev_kp = torch.argsort(next_kp)
        prev_w = words[:, :, prev_kp]
    else:
        prev_w = words[:, torch.arange(160, device=device) ^ 1]
    full = torch.cat([prev_w[:, :, :, :, -2:], words], 4)               # w[-2], w[-1], w[0], ...
    v = full[:, :, :, :, 2:]
    u2 = full[:, :, :, :, :-2]
    z = full[:, :, :, :, 1:-1]
    U = sum(1 << (bits * j) for j in range(f))
    U = U | (U << 16)
    y = v ^ (u2 & U)
    geo = geometry(matrix, bits, device)
    shift = geo.shift[0, 0, 0, 0]                                       # [half, field]
    ys = (y.unsqueeze(-1).unsqueeze(-1) >> shift) & ((1 << bits) - 1)
    cs = (z.unsqueeze(-1).unsqueeze(-1) >> shift) & 1
    idx = (2 * ys + cs).long()                                          # [E, rt, kp, lane, word, half, field]
    lev = t.trellis_levels(bits, cubic, device)
    R = 1280 if matrix == "w13" else 2560
    K = 2560 if matrix == "w13" else 640
    scale_words = packed[:, :, :, SOFF[bits] // 4 + torch.arange(8, device=device)]
    lo = (scale_words & 0xFFFF).to(torch.int16)
    hi = ((scale_words >> 16) & 0xFFFF).to(torch.int16)
    s_lo, s_hi = lo.view(torch.bfloat16).float(), hi.view(torch.bfloat16).float()
    S = torch.zeros((E, R, K // 128), device=device)
    rt_ = torch.arange(rts, device=device).view(-1, 1, 1)
    kp_ = torch.arange(kps, device=device).view(1, -1, 1)
    g_ = torch.arange(8, device=device).view(1, 1, -1)
    if matrix == "w13":
        row_lo, row_hi = rt_ * 8 + g_, 640 + rt_ * 8 + g_
    else:
        row_lo, row_hi = rt_ * 16 + g_, rt_ * 16 + g_ + 8
    S[:, row_lo, kp_] = s_lo
    S[:, row_hi, kp_] = s_hi
    hat = torch.empty((E, R, K), device=device)
    hat[:, geo.row, geo.col] = lev[idx] * S[:, geo.row, geo.col // 128]
    if bits == 3:
        lev_s = t.scalar_levels(3, cubic, device)
        sq = torch.zeros((E, rts, kps, 32, 2, 2), dtype=torch.long, device=device)
        for pair in range(2):
            for b in range(3):
                word = words[:, :, :, :, pair * 3 + b]
                sq[:, :, :, :, pair, 0] |= ((word >> 15) & 1) << b
                sq[:, :, :, :, pair, 1] |= ((word >> 31) & 1) << b
        hat[:, geo.spare_row, geo.spare_col] = lev_s[sq] * S[:, geo.spare_row, geo.spare_col // 128]
    return hat


def plan_tensors(directory, index):
    """A layer's tensors, read from the library through a t3_alloc --plan file, so scoring writes no copy."""
    plan = json.loads((Path(directory) / "plan.json").read_text())
    lib, n = Path(plan["lib"]) / f"L{index}", str(index)
    result = []
    for key, rts, kps, base, alternatives in (
        ("13", 80, 20, "t3_13", {2: "t2_13"}),
        ("2", 160, 5, "t3_2", {2: "t2_2", 4: "t4_2"}),
    ):
        fmt = np.full(512, 3, dtype=np.uint8)
        positions = {}
        for bits, name in alternatives.items():
            for e, pos in plan[name].get(n, {}).items():
                fmt[int(e)] = bits
                positions[int(e)] = (name, pos)
        sizes = rts * kps * np.array([PKT[int(b)] for b in fmt], dtype=np.int64)
        off = np.concatenate(([0], np.cumsum(sizes)[:-1]))
        data = np.empty(int(sizes.sum()), dtype=np.uint8)
        files = {name: np.memmap(lib / f"{name}.bin", dtype=np.uint8, mode="r") for name in [base, *alternatives.values()] if (lib / f"{name}.bin").exists()}
        for e in range(512):
            name, pos = positions.get(e, (base, e))
            start = int(pos) * int(rts * kps * PKT[int(fmt[e])])
            data[off[e]:off[e] + sizes[e]] = files[name][start:start + int(sizes[e])]
        result += [data, off, fmt]
    return result


def load_layer(directory, index, device):
    """Decoded gate_up [512, 1280, 2560] and down [512, 2560, 640], BF16, for the simulator."""
    if (Path(directory) / "plan.json").exists():
        w13, off13, fmt13, w2, off2, fmt2 = plan_tensors(directory, index)
    else:
        with safe_open(f"{directory}/L{index}.safetensors", framework="np") as f:
            prefix = f"model.language_model.layers.{index}.mlp.experts."
            w13, w2 = f.get_tensor(prefix + "t3_w13"), f.get_tensor(prefix + "t3_w2")
            off13, off2 = f.get_tensor(prefix + "t3_off13"), f.get_tensor(prefix + "t3_off2")
            fmt13, fmt2 = f.get_tensor(prefix + "t3_fmt13"), f.get_tensor(prefix + "t3_fmt2")
    gate_up = torch.empty((512, 1280, 2560), dtype=torch.bfloat16, device=device)
    down = torch.empty((512, 2560, 640), dtype=torch.bfloat16, device=device)
    for out, data, off, fmt, matrix, rts, kps in ((gate_up, w13, off13, fmt13, "w13", 80, 20), (down, w2, off2, fmt2, "w2", 160, 5)):
        for bits in np.unique(fmt).tolist():
            members = np.flatnonzero(fmt == bits)
            for start in range(0, len(members), 64):
                part = members[start:start + 64]
                size = rts * kps * PKT[bits]
                raw = np.stack([np.frombuffer(data[off[e]:off[e] + size].tobytes(), dtype=np.uint32) for e in part])
                packed = torch.from_numpy(raw.reshape(len(part), rts, kps, PKT[bits] // 4).astype(np.int64)).to(device)
                out[torch.from_numpy(part).to(device)] = decode(packed, matrix, int(bits), device=device).to(torch.bfloat16)
    return gate_up, down
