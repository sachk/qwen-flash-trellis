#!/usr/bin/env python3
"""Kernel output vs. a dense reference built from t3pack.decode, on random packets, with padding rows and mixed formats."""
import ctypes, os, sys
import numpy as np, torch, torch.nn.functional as F
import t3pack as p

torch.set_grad_enabled(False)
D = "cuda"
lib = ctypes.CDLL(os.environ["T3_TESTS_LIB"])
lib.t3_init()
lib.t3_forward.argtypes = [ctypes.c_void_p] * 19 + [ctypes.c_int, ctypes.c_void_p, ctypes.c_long]
E_USED = 64
rng = torch.Generator(device=D).manual_seed(7)
fmt13 = torch.tensor([3, 2] * 256, dtype=torch.uint8, device=D)
fmt2 = torch.tensor([3, 4, 2, 3] * 128, dtype=torch.uint8, device=D)


def random_matrix(fmt, matrix):
    rts, kps = (80, 20) if matrix == "w13" else (160, 5)
    out, offsets, off = [], [], 0
    for e in range(512):
        bits = int(fmt[e])
        size = rts * kps * p.PKT[bits]
        offsets.append(off)
        if e < E_USED:
            words = torch.randint(0, 2**32, (rts, kps, p.PKT[bits] // 4), generator=rng, device=D, dtype=torch.int64)
            scale = torch.randint(0x3700, 0x3840, (rts, kps, 8), generator=rng, device=D, dtype=torch.int64)
            words[:, :, p.SOFF[bits] // 4 + torch.arange(8, device=D)] = scale | (scale << 16)
            out.append(words.flatten().to(torch.uint32 if hasattr(torch, "uint32") else torch.int64))
            off += size
        else:
            offsets[-1] = 0
    data = torch.cat([x.to(torch.int64) for x in out])
    raw = data.cpu().numpy().astype(np.uint32).tobytes()
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(D), torch.tensor(offsets, dtype=torch.int64, device=D)


w13, off13 = random_matrix(fmt13, "w13")
w2, off2 = random_matrix(fmt2, "w2")


def expert_dense(e):
    b13, b2 = int(fmt13[e]), int(fmt2[e])
    s13 = 80 * 20 * p.PKT[b13]
    s2 = 160 * 5 * p.PKT[b2]
    a = np.frombuffer(w13[int(off13[e]):int(off13[e]) + s13].cpu().numpy().tobytes(), dtype=np.uint32).astype(np.int64)
    c = np.frombuffer(w2[int(off2[e]):int(off2[e]) + s2].cpu().numpy().tobytes(), dtype=np.uint32).astype(np.int64)
    gu = p.decode(torch.from_numpy(a).to(D).view(1, 80, 20, p.PKT[b13] // 4), "w13", b13, "c2")[0]
    dn = p.decode(torch.from_numpy(c).to(D).view(1, 160, 5, p.PKT[b2] // 4), "w2", b2, "c2")[0]
    return gu, dn


dense = [expert_dense(e) for e in range(E_USED)]
MAXROWS = 20480 + 512 * 16
cnt, nxt, base = [torch.zeros(512, dtype=torch.int32, device=D) for _ in range(3)]
blk = torch.zeros(MAXROWS, dtype=torch.int32, device=D)
nb = torch.zeros(1, dtype=torch.int32, device=D)
rid = torch.zeros(MAXROWS, dtype=torch.int32, device=D)
rw = torch.zeros(MAXROWS, dtype=torch.float32, device=D)
hB = torch.zeros((MAXROWS, 640), dtype=torch.bfloat16, device=D)
acc = torch.zeros((2048, 2560), dtype=torch.float32, device=D)
stream = torch.cuda.current_stream().cuda_stream
worst = 0.0
failed = False
for M, pad in ((1, 0), (2, 0), (4, 0), (4, 1), (8, 0), (9, 0), (16, 0), (16, 3), (32, 0), (47, 0), (48, 0), (64, 2), (128, 0), (256, 0), (2048, 0)):
    x = (torch.randn(M, 2560, device=D, generator=rng)).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E_USED, device=D, generator=rng)[:10] for _ in range(M)]).int()
    tw = F.softmax(torch.randn(M, 10, device=D, generator=rng), -1).float()
    if pad:
        ids[-pad:] = -1
    out = torch.full((M, 2560), float("nan"), dtype=torch.bfloat16, device=D)
    args = [w13, w2, off13, off2, fmt13, fmt2, cnt, nxt, base, blk, nb, rid, rw, hB, acc, x, ids, tw]
    lib.t3_forward(ctypes.c_void_p(int(stream)), *[ctypes.c_void_p(t.data_ptr()) for t in args], ctypes.c_int(M), ctypes.c_void_p(out.data_ptr()), ctypes.c_long(2560))
    torch.cuda.synchronize()
    ref = torch.zeros(M, 2560, dtype=torch.float64, device=D)
    xf = x.float()
    for row in range(M):
        if ids[row, 0] < 0:
            continue
        for k in range(10):
            gu, dn = dense[int(ids[row, k])]
            g, u = (xf[row] @ gu.T).chunk(2)
            h = (F.silu(g) * u).to(torch.bfloat16).float()
            ref[row] += float(tw[row, k]) * (h @ dn.T).double()
    got = out.float().double()
    err = float((got - ref).norm() / ref.norm().clamp_min(1e-9))
    nans = int(torch.isnan(got).sum())
    worst = max(worst, err)
    failed |= not bool(torch.isfinite(got).all()) or not err < 2e-3
    if pad:
        failed |= bool(got[-pad:].ne(0).any())
    print(f"M={M} pad={pad}: relative error {err:.2e}, n_blocks {int(nb)}, NaNs {nans}, padded row max {float(got[-pad:].abs().max()) if pad else 0:.2e}", flush=True)
print("FAIL" if failed else "PASS", worst)
sys.exit(int(failed))
