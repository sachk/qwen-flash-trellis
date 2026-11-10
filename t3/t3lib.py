"""Hessian-aware rounding for the ring-coded trellis formats (T2/T3/T4) and plain W3, batched over experts.

Objective, as in GPTQ: per row r, (w_hat_r - w_r) H (w_hat_r - w_r)^T, with H over the K dimension.
Rows don't interact, so all rows run in parallel. Weights within a row interact through H, so positions are
split into colours: all rings of one colour are solved by a Viterbi pass with everything else held fixed, then
the gradient G = delta H is updated. A colour holds about 1/40 of the columns, so a sweep costs one full
delta @ H multiply plus the Viterbi passes. Sweep 1 from delta = 0 is sequential error feedback in colour order.
Later sweeps are block coordinate descent.

Layout (see w3p_format.h, w3t.cu): lane (g, t), pair P = F*i + j in word i and field j, half h.
  row in tile = g + 8*(P&1); k in packet = 16*(P>>2) + 2t + 8*((P>>1)&1) + h.
Each (lane, j, h) is a tail-biting ring through the words of the unit's packets, with the 4-state code:
  c1 = u[i] ^ u[i-2], c0 = u[i-1], idx = free_bits*4 + 2*c1 + c0, value = levels[idx].
Spare pairs (F*W <= P < 32) are scalar.
"""
import numpy as np
import torch

BF16 = torch.bfloat16
FORMATS = {
    "T2": dict(bits=2, F=8, W=4),
    "T3": dict(bits=3, F=5, W=6),
    "T4": dict(bits=4, F=4, W=8),
    "S3": dict(bits=3, F=5, W=6, scalar=True),   # plain W3 in the same packet layout
}


def pair_of(fmt, i, j):
    """Fragment pair of word i, field j. In T3, consecutive words of a ring hold rows g and g+8 at the same k,
    so all rows have the same column order and the solver needs only one factorisation."""
    F, W = FORMATS[fmt]["F"], FORMATS[fmt]["W"]
    if fmt in ("T3", "S3"):
        return 10 * (i // 2) + 2 * j + (i & 1)
    return F * i + j




SHAPE_B = {2: 250.0, 3: 1000.0, 4: 4000.0}


def shaped(v, bits):
    """The kernel's L = (v*v + B) * v, with BF16 rounding after each FMA."""
    t = (v.double() * v.double() + SHAPE_B[bits]).to(BF16).double()
    return (t * v.double()).to(BF16).float()


def trellis_levels(bits, cubic=0.0, device="cuda"):
    n = 2 ** (bits + 1)
    v = torch.arange(n, dtype=torch.float32, device=device) * 2 - (n - 1)
    if cubic == "c2":
        return shaped(v, bits)
    if cubic < 0:                                    # v(1 + b|v|)
        return v * (1 - cubic * v.abs())
    return v * (1 + cubic * v * v)


def scalar_levels(bits, cubic=0.0, device="cuda"):
    n = 2 ** bits
    centre = (n - 1) / 2
    v = (torch.arange(n, dtype=torch.float32, device=device) - centre) * 4
    v = v.round()
    if cubic == "c2":
        return shaped(v, 3)
    if cubic < 0:
        return v * (1 - cubic * v.abs())
    return v * (1 + cubic * v * v)


class Layout:
    """Column and row sets per colour, and each ring's positions in them."""

    def __init__(self, matrix, fmt):
        info = FORMATS[fmt]
        F, W = info["F"], info["W"]
        self.matrix, self.fmt, self.F, self.W = matrix, fmt, F, W
        self.scalar = info.get("scalar", False)
        if matrix == "w13":
            self.R, self.K = 1280, 2560
            tiles, kps, ppw = 80, 20, 4
        else:
            self.R, self.K = 2560, 640
            tiles, kps, ppw = 160, 5, 2
        self.kp_of_col = None
        self.colours = []
        rows_of = {}
        for c in (0, 1):
            g = np.arange(8)
            if matrix == "w13":
                rt = np.arange(80)
                rows_of[c] = (rt[:, None] * 8 + g[None, :]).ravel() + (640 if c else 0)
            else:
                rt = np.arange(160)
                rows_of[c] = (rt[:, None] * 16 + g[None, :] + (8 if c else 0)).ravel()
        self.rows_of = rows_of

        def col(kp, t, P, h):
            return kp * 128 + 16 * (P >> 2) + 2 * t + (8 if P & 2 else 0) + h

        n_dense = F * W
        for t in range(4):
            for j in range(F):
                for h in range(2):
                    self.colours.append(self._ring_colour(matrix, F, W, t, j, h, kps, ppw, rows_of, col))
            if n_dense < 32:
                for h in range(2):
                    self.colours.append(self._spare_colour(t, h, kps, n_dense, rows_of, col))

    def _ring_colour(self, matrix, F, W, t, j, h, kps, ppw, rows_of, col):
        T = ppw * W
        cols = {0: [], 1: []}
        for kp in range(kps):
            for i in range(W):
                P = pair_of(self.fmt, i, j)
                cols[P & 1].append(col(kp, t, P, h))
        pos = {c: {v: n for n, v in enumerate(cols[c])} for c in cols}
        n = {c: len(cols[c]) for c in cols}
        offset = {0: 0, 1: len(rows_of[0]) * n[0]}
        row_pos = {c: {int(r): idx for idx, r in enumerate(rows_of[c])} for c in rows_of}
        rings = []
        if matrix == "w13":
            units = [(rt, group) for rt in range(80) for group in range(kps // ppw)]
        else:
            units = [(rt, kp) for rt in range(0, 160, 2) for kp in range(kps)]
        for unit in units:
            for g in range(8):
                ring = []
                for s in range(T):
                    p, i = s // W, s % W
                    P = pair_of(self.fmt, i, j)
                    c = P & 1
                    if matrix == "w13":
                        rt, group = unit
                        kp = group * ppw + p
                        row = rt * 8 + g + (640 if c else 0)
                    else:
                        base, kp = unit
                        rt = base + p
                        row = rt * 16 + g + (8 if c else 0)
                    ring.append(offset[c] + row_pos[c][row] * n[c] + pos[c][col(kp, t, P, h)])
                rings.append(ring)
        return dict(kind="ring", cols=cols, rings=np.array(rings, dtype=np.int64), rows=rows_of)

    def _spare_colour(self, t, h, kps, n_dense, rows_of, col):
        cols = {0: [], 1: []}
        for kp in range(kps):
            for P in range(n_dense, 32):
                cols[P & 1].append(col(kp, t, P, h))
        return dict(kind="spare", cols=cols, rows=rows_of)

    def to(self, device):
        for colour in self.colours:
            colour["cols_t"] = {c: torch.tensor(v, device=device) for c, v in colour["cols"].items() if len(v)}
            colour["rows_t"] = {c: torch.from_numpy(colour["rows"][c]).to(device) for c in colour["cols_t"]}
            if colour["kind"] == "ring":
                colour["rings_t"] = torch.from_numpy(colour["rings"]).to(device)
        return self


_cache = {}


def layout(matrix, fmt, device="cuda"):
    key = (matrix, fmt, str(device))
    if key not in _cache:
        _cache[key] = Layout(matrix, fmt).to(device)
    return _cache[key]


def coset_tables(levels):
    """For each of 4 cosets: its level indices [4, n] and the midpoints between its levels [4, n-1]."""
    n = len(levels) // 4
    idx = torch.stack([torch.arange(c, len(levels), 4, device=levels.device) for c in range(4)])
    values = levels[idx]
    mids = (values[:, :-1] + values[:, 1:]) / 2
    return idx, values, mids, n


def tb_viterbi(cost):
    """Exact tail-biting min-sum for the 4-state code. cost: [B, T, 4] by label. Returns path bits u [B, T]."""
    B, T, _ = cost.shape
    device = cost.device
    states = torch.arange(4, device=device)
    u, u1 = states >> 1, states & 1
    p0, p1 = u1 * 2, u1 * 2 + 1              # predecessors (u1, u2=0/1) of (u, u1)
    lab0, lab1 = ((u ^ 0) * 2 + u1), ((u ^ 1) * 2 + u1)
    J = torch.full((B, 4, 4), float("inf"), device=device, dtype=cost.dtype)
    J[:, states, states] = 0
    back = torch.empty((T, B, 4, 4), dtype=torch.bool, device=device)
    for t in range(T):
        c = cost[:, t]
        a = J[:, :, p0] + c[:, lab0].unsqueeze(1)
        b = J[:, :, p1] + c[:, lab1].unsqueeze(1)
        take = b < a
        back[t] = take
        J = torch.where(take, b, a)
    final = J[:, states, states]
    start = final.argmin(1)
    state = start.clone()
    bits = torch.empty((B, T), dtype=torch.long, device=device)
    rows = torch.arange(B, device=device)
    for t in range(T - 1, -1, -1):
        current = state
        bits[:, t] = current >> 1
        u2 = back[t][rows, start, current].long()
        state = (current & 1) * 2 + u2
    return bits


def labels_from_bits(u):
    return (u ^ u.roll(2, 1)) * 2 + u.roll(1, 1)


def condition(H, gate2, routed, pool, n_min_scale=2.0, damp=0.01):
    """As s2_pass.condition, for a batch. H is normalised by gate^2; cold experts blend toward the pool."""
    K = H.shape[-1]
    H = H / gate2.to(H.device).float().clamp_min(1e-12).view(-1, 1, 1)
    n_min = n_min_scale * K
    alpha = (n_min / routed.to(H.device).float().clamp_min(1)).clamp_max(1e6)
    cold = routed.to(H.device) < n_min
    if cold.any():
        H[cold] += alpha[cold].view(-1, 1, 1) * pool.to(H.device)
    mean = H.diagonal(dim1=1, dim2=2).mean(1).clamp_min(1e-8)
    H.diagonal(dim1=1, dim2=2).add_(damp * mean.unsqueeze(1))
    return H


def group_sigma(w, diag):
    """H-diagonal-weighted rms per row and 128-group. w: [E, R, K], diag: [E, K]."""
    E, R, K = w.shape
    G = K // 128
    grouped = w.reshape(E, R, G, 128)
    d = diag.reshape(E, 1, G, 128)
    return ((grouped * grouped * d).sum(-1) / d.sum(-1).clamp_min(1e-12)).sqrt().clamp_min(1e-7)


def to_bf16(x):
    return x.to(BF16).float()


def round_batch(w, H, fmt, alpha, cubic=0.0, sweeps=4, refine=0, order_seed=0, callback=None):
    """Returns (w_hat, s, idx) for a batch of experts. w: [E, R, K] float32; H conditioned [E, K, K]."""
    device = w.device
    E, R, K = w.shape
    lay = layout("w13" if K == 2560 else "w2", fmt, device)
    info = FORMATS[fmt]
    bits = info["bits"]
    scalar = info.get("scalar", False)
    lev_t = trellis_levels(bits, cubic, device)
    lev_s = scalar_levels(bits if bits == 3 else 3, cubic, device)
    idx_t, val_t, mid_t, _ = coset_tables(lev_t)
    mid_s = (lev_s[:-1] + lev_s[1:]) / 2
    diag = H.diagonal(dim1=1, dim2=2)
    s = to_bf16(alpha * group_sigma(w, diag)).clamp_min(1e-6)
    kp_map = torch.arange(K, device=device) // 128
    hat = w.clone()
    G = torch.zeros_like(w)
    q_store = torch.zeros((E, R, K), dtype=torch.int8, device=device)
    is_scalar_pos = torch.zeros((E, R, K), dtype=torch.bool, device=device)
    rng = np.random.default_rng(order_seed)
    colours = list(range(len(lay.colours)))

    for sweep in range(sweeps):
        rng.shuffle(colours)
        for ci in colours:
            colour = lay.colours[ci]
            parts = []
            for c in colour["cols_t"]:
                rows, cols = colour["rows_t"][c], colour["cols_t"][c]
                rr, cc = rows.unsqueeze(1), cols.unsqueeze(0)
                d = diag[:, cols].unsqueeze(1)
                s_pos = s[:, rows.unsqueeze(1), kp_map[cols].unsqueeze(0)]
                cur = hat[:, rr, cc]
                res = G[:, rr, cc] - (cur - w[:, rr, cc]) * d
                tau = (w[:, rr, cc] - res / d) / s_pos
                parts.append((c, rows, cols, rr, cc, d, s_pos, cur, tau))
            if colour["kind"] == "spare" or scalar:
                for c, rows, cols, rr, cc, d, s_pos, cur, tau in parts:
                    q = (tau.unsqueeze(-1) > mid_s).sum(-1)
                    new = lev_s[q] * s_pos
                    q_store[:, rr, cc] = q.to(torch.int8)
                    is_scalar_pos[:, rr, cc] = True
                    delta = new - cur
                    hat[:, rr, cc] = new
                    G[:, rows] += delta @ H[:, cols]
                continue
            # Trellis colour: costs per label, [E, positions, 4], then ring order.
            flat_tau = torch.cat([p[8].flatten(1) for p in parts], 1)
            flat_d = torch.cat([(p[5] * p[6] * p[6]).flatten(1) for p in parts], 1)
            m = (flat_tau.unsqueeze(-1).unsqueeze(-1) > mid_t).sum(-1)      # [E, N, 4] index within coset
            values = val_t[torch.arange(4, device=device).view(1, 1, 4), m]
            cost = flat_d.unsqueeze(-1) * (values - flat_tau.unsqueeze(-1)) ** 2
            rings = colour["rings_t"]
            ring_cost = cost[:, rings]                                       # [E, rings, T, 4]
            N, T = rings.shape
            bits_u = tb_viterbi(ring_cost.reshape(E * N, T, 4)).reshape(E, N, T)
            lab = labels_from_bits(bits_u.reshape(E * N, T)).reshape(E, N, T)
            # Back to flat positions.
            flat_lab = torch.empty_like(m[:, :, 0])
            flat_lab[:, rings] = lab
            chosen_m = m.gather(2, flat_lab.unsqueeze(-1)).squeeze(-1)
            chosen = idx_t[flat_lab, chosen_m]
            start = 0
            for c, rows, cols, rr, cc, d, s_pos, cur, tau in parts:
                n = tau.shape[1] * tau.shape[2]
                q = chosen[:, start:start + n].reshape(tau.shape)
                start += n
                new = lev_t[q] * s_pos
                q_store[:, rr, cc] = q.to(torch.int8)
                delta = new - cur
                hat[:, rr, cc] = new
                G[:, rows] += delta @ H[:, cols]
        if refine and sweep < refine:
            s, hat, G = refine_scales(w, hat, s, H, diag)
        if callback:
            callback(sweep, hat)
    return hat, s, q_store, is_scalar_pos


def refine_scales(w, hat, s, H, diag):
    """Least-squares group scale for the current integers, against the current target."""
    E, R, K = w.shape
    G_cur = (hat - w) @ H
    target = hat - G_cur / diag.unsqueeze(1)
    v = hat / s.repeat_interleave(128, 2)
    d = diag.unsqueeze(1)
    num = ((target * v * d).reshape(E, R, K // 128, 128)).sum(-1)
    den = ((v * v * d).reshape(E, R, K // 128, 128)).sum(-1).clamp_min(1e-12)
    new_s = to_bf16((num / den).clamp_min(1e-6))
    # Damped toward the old scale, since the target moves with the scale.
    new_s = to_bf16(s + 0.5 * (new_s - s))
    new_hat = v * new_s.repeat_interleave(128, 2)
    return new_s, new_hat, (new_hat - w) @ H


def objective(w, hat, H):
    d = hat - w
    return (((d @ H) * d).sum((1, 2)) / ((w @ H) * w).sum((1, 2)).clamp_min(1e-12))


# ---------------------------------------------------------------- sequential (Babai) rounding with survivors
class Plan:
    """Column order and ring steps for the sequential solver, one per (matrix, format)."""

    def __init__(self, matrix, fmt, device):
        info = FORMATS[fmt]
        F, W = info["F"], info["W"]
        self.matrix, self.fmt, self.F, self.W = matrix, fmt, F, W
        if matrix == "w13":
            self.R, self.K, kps, ppw = 1280, 2560, 20, 4
            self.groups = [[g * 4 + p for p in range(4)] for g in range(5)]
            self.orig_rows = {0: np.arange(640), 1: 640 + np.arange(640)}
            self.n_rings = 640
        else:
            self.R, self.K, kps, ppw = 2560, 640, 5, 2
            self.groups = [[kp] for kp in range(5)]
            rows = np.arange(2560)
            self.orig_rows = {0: rows[(rows & 8) == 0], 1: rows[(rows & 8) != 0]}
            self.n_rings = 640
        T = ppw * W

        def col(kp, t, P, h):
            return kp * 128 + 16 * (P >> 2) + 2 * t + (8 if P & 2 else 0) + h

        order = {0: [], 1: []}
        n_dense = F * W
        self.spare = {0: [], 1: []}
        if n_dense < 32:
            for kp in range(kps):
                for t in range(4):
                    for h in range(2):
                        for P in range(n_dense, 32):
                            self.spare[P & 1].append(col(kp, t, P, h))
            order[0].extend(self.spare[0])
            order[1].extend(self.spare[1])
        self.blocks = []
        for t in range(4):
            for j in range(F):
                for h in range(2):
                    for group in self.groups:
                        steps = []
                        for s in range(T):
                            p, i = s // W, s % W
                            P = pair_of(fmt, i, j)
                            c = P & 1
                            if matrix == "w13":
                                kp, row_page = group[p], 0
                            else:
                                kp, row_page = group[0], p
                            steps.append(dict(s=s, c=c, kp=kp, page=row_page, col=col(kp, t, P, h)))
                        block_cols = {0: [], 1: []}
                        for st in steps:
                            if st["col"] not in block_cols[st["c"]]:
                                block_cols[st["c"]].append(st["col"])
                        self.blocks.append(dict(steps=steps, cols=block_cols))
                        for c in (0, 1):
                            order[c].extend(block_cols[c])
        self.perm = {}
        pos = {}
        for c in (0, 1):
            assert sorted(order[c]) == sorted(set(order[c]))
            self.perm[c] = torch.tensor(order[c], device=device)
            pos[c] = {v: n for n, v in enumerate(order[c])}
        self.K_c = {c: len(order[c]) for c in (0, 1)}
        # Rows in each class are laid out so that ring m on page p is row m + page * 640 (w2) or m (w13).
        for b in self.blocks:
            b["c"] = torch.tensor([st["c"] for st in b["steps"]], device=device)
            b["kp"] = torch.tensor([st["kp"] for st in b["steps"]], device=device)
            b["page"] = torch.tensor([st["page"] for st in b["steps"]], device=device)
            b["pos"] = torch.tensor([pos[st["c"]][st["col"]] for st in b["steps"]], device=device)
            b["start"] = {c: (pos[c][b["cols"][c][0]] if b["cols"][c] else None) for c in (0, 1)}
            b["n"] = {c: len(b["cols"][c]) for c in (0, 1)}
            prevs = []
            for s, st in enumerate(b["steps"]):
                prevs.append([q for q in range(s) if b["steps"][q]["c"] == st["c"] and b["steps"][q]["page"] == st["page"]])
            b["prevs"] = prevs
        self.T = T
        # Class row arrays: for w13, m -> m. For w2, rows in class order are rt*8+g; ring m = a*8+g on page p is rt=2a+p.
        if matrix == "w2":
            m = torch.arange(640, device=device)
            a, g = m // 8, m % 8
            self.row_of_page = torch.stack([(2 * a + p) * 8 + g for p in range(2)])   # [2, 640]
        else:
            self.row_of_page = torch.arange(640, device=device).unsqueeze(0)
        self.orig_rows_t = {c: torch.from_numpy(v).to(device) for c, v in self.orig_rows.items()}


_plans = {}


def plan(matrix, fmt, device="cuda"):
    key = (matrix, fmt, str(device))
    if key not in _plans:
        _plans[key] = Plan(matrix, fmt, device)
    return _plans[key]


# Branch tables. Predecessor p = (u1, u2), branch u: label = 2*(u^u2) + u1, next = (u, u1).
_P = torch.arange(4)
_U1, _U2 = _P >> 1, _P & 1
_LAB = torch.stack([2 * (0 ^ _U2) + _U1, 2 * (1 ^ _U2) + _U1], 1)          # [4 preds, 2 branches]
_N = torch.arange(4)
_NU, _NU1 = _N >> 1, _N & 1
_P0, _P1 = 2 * _NU1, 2 * _NU1 + 1


def inverse_factor(H):
    factor, info = torch.linalg.cholesky_ex(H)
    damp = 0.1
    while info.any():
        for e in torch.nonzero(info).flatten().tolist():
            H[e].diagonal().add_(damp * H[e].diagonal().mean())
        factor, info = torch.linalg.cholesky_ex(H)
        damp *= 10
    return torch.linalg.cholesky(torch.cholesky_inverse(factor), upper=True)


def group_amax(w):
    E, R, K = w.shape
    return w.reshape(E, R, K // 128, 128).abs().amax(-1)


def babai_round(w, H, fmt, alpha, cubic=0.0, starts=4, check=False, beta=0.0):
    """Sequential rounding with per-survivor feedback inside each ring. Returns (hat, s, q, spare_mask)."""
    device = w.device
    E, R, K = w.shape
    pl = plan("w13" if K == 2560 else "w2", fmt, device)
    info = FORMATS[fmt]
    bits = info["bits"]
    lev_t = trellis_levels(bits, cubic, device)
    lev_s = scalar_levels(3, cubic, device)
    idx_t, val_t, mid_t, _ = coset_tables(lev_t)
    mid_s = (lev_s[:-1] + lev_s[1:]) / 2
    lab = _LAB.to(device)
    p0, p1, nu = _P0.to(device), _P1.to(device), _NU.to(device)
    diag = H.diagonal(dim1=1, dim2=2)
    raw = torch.maximum(alpha * group_sigma(w, diag), beta * group_amax(w) / lev_t.max())
    # A scale past this puts the innermost level beyond the group's largest weight. L32 of pass 2 produced an infinite scale.
    cap = group_amax(w) / lev_t.abs()[lev_t != 0].min()
    s = to_bf16(torch.where(torch.isfinite(raw), torch.minimum(raw, cap), cap).clamp_min(1e-10))

    classes = {}
    Uc = None
    for c in (0, 1):
        perm = pl.perm[c]
        rows = pl.orig_rows_t[c]
        Wc = w[:, rows][:, :, perm].clone()
        if c == 0 or not torch.equal(perm, pl.perm[0]):
            Hc = H[:, perm][:, :, perm]
            Uc = inverse_factor(Hc)
            del Hc
        scale_map = (perm // 128)
        Sc = s[:, rows][:, :, scale_map]                                   # [E, R_c, K], scale per position
        classes[c] = dict(W=Wc, U=Uc, S=Sc, hat=torch.empty_like(Wc), q=torch.zeros(Wc.shape, dtype=torch.int8, device=device), done=0)

    # Spare scalar columns first, one at a time, as GPTQ.
    for c in (0, 1):
        C = classes[c]
        n = len(pl.spare[c])
        for i in range(n):
            tau = C["W"][:, :, i] / C["S"][:, :, i]
            q = (tau.unsqueeze(-1) > mid_s).sum(-1)
            v = lev_s[q] * C["S"][:, :, i]
            C["hat"][:, :, i] = v
            C["q"][:, :, i] = q.to(torch.int8)
            err = (C["W"][:, :, i] - v) / C["U"][:, i, i].unsqueeze(1)
            C["W"][:, :, i + 1:] -= err.unsqueeze(2) * C["U"][:, i, i + 1:].unsqueeze(1)
        C["done"] = n

    m = pl.n_rings
    B = E * m
    T = pl.T
    rows_e = torch.arange(E, device=device).view(E, 1)
    state_ids = torch.arange(4, device=device)
    start_ids = state_ids if starts == 4 else state_ids[:starts]
    St = len(start_ids)
    for block in pl.blocks:
        ring_rows = [pl.row_of_page[block["page"][t]] for t in range(T)]
        tau_base = torch.stack([classes[int(block["c"][t])]["W"][:, ring_rows[t], block["pos"][t]] for t in range(T)], 2)   # [E, m, T]
        scale = torch.stack([classes[int(block["c"][t])]["S"][:, ring_rows[t], block["pos"][t]] for t in range(T)], 2)
        tau_base, scale = tau_base.reshape(B, T), scale.reshape(B, T)
        U_diag = torch.stack([classes[int(block["c"][t])]["U"][:, block["pos"][t], block["pos"][t]] for t in range(T)], 1)  # [E, T]
        U_diag = U_diag.repeat_interleave(m, 0)                              # [B, T]
        J = torch.full((B, St, 4), float("inf"), device=device)
        J[:, torch.arange(St, device=device), start_ids] = 0
        hist = torch.zeros((B, St, 4, T), device=device)
        hist_idx = torch.zeros((B, St, 4, T), dtype=torch.int8, device=device)
        for t in range(T):
            prev = block["prevs"][t]
            tau = tau_base[:, t].view(B, 1, 1).expand(B, St, 4)
            if prev:
                c = int(block["c"][t])
                U = classes[c]["U"]
                coefficient = U[:, block["pos"][prev], block["pos"][t]]        # [E, n_prev]
                coefficient = coefficient.repeat_interleave(m, 0).view(B, 1, 1, len(prev))
                tau = tau - (hist[:, :, :, prev] * coefficient).sum(-1)
            scaled = tau / scale[:, t].view(B, 1, 1)                           # [B, St, 4]
            m_idx = (scaled[..., None, None] > mid_t[lab]).sum(-1)              # [B, St, 4 preds, 2]
            values = val_t[lab, m_idx]
            eps = (tau.unsqueeze(-1) - values * scale[:, t].view(B, 1, 1, 1)) / U_diag[:, t].view(B, 1, 1, 1)
            cost = eps * eps
            a = J[:, :, p0] + cost[:, :, p0, nu]
            b = J[:, :, p1] + cost[:, :, p1, nu]
            take = b < a
            pred = torch.where(take, p1, p0)                                   # [B, St, 4]
            J = torch.where(take, b, a)
            gather = pred.unsqueeze(-1).expand(B, St, 4, T)
            hist = hist.gather(2, gather)
            hist_idx = hist_idx.gather(2, gather)
            brow = torch.arange(B, device=device).view(B, 1, 1)
            bst = torch.arange(St, device=device).view(1, St, 1)
            br = nu.view(1, 1, 4)
            hist[:, :, :, t] = eps[brow, bst, pred, br]
            chosen_m = m_idx[brow, bst, pred, br]
            chosen_lab = lab[pred, br.expand_as(pred)]
            hist_idx[:, :, :, t] = idx_t[chosen_lab, chosen_m].to(torch.int8)
        final = J[:, torch.arange(St, device=device), start_ids]              # [B, St]
        best = final.argmin(1)
        rows_b = torch.arange(B, device=device)
        end = start_ids[best]
        eps_best = hist[rows_b, best, end].reshape(E, m, T)
        idx_best = hist_idx[rows_b, best, end].long().reshape(E, m, T)
        if check:
            c0 = idx_best & 1
            c1 = (idx_best >> 1) & 1
            # u[s-1] = c0[s]; so u[s] = c0[s+1], and c1[s] = u[s] ^ u[s-2].
            u = c0.roll(-1, 2)
            assert bool((c1 == (u ^ u.roll(2, 2))).all()), "path inconsistent"
        for c in (0, 1):
            n = block["n"][c]
            if not n:
                continue
            C = classes[c]
            start = block["start"][c]
            E1 = torch.zeros((E, C["W"].shape[1], n), device=device)
            for t in range(T):
                if int(block["c"][t]) != c:
                    continue
                row = ring_rows[t]
                pos = block["pos"][t]
                v = lev_t[idx_best[:, :, t]] * C["S"][:, row, pos]
                C["hat"][:, row, pos] = v
                C["q"][:, row, pos] = idx_best[:, :, t].to(torch.int8)
                E1[:, row, int(pos) - start] = eps_best[:, :, t]
            if start + n < C["W"].shape[2]:
                C["W"][:, :, start + n:] -= E1 @ C["U"][:, start:start + n, start + n:]
        del J, hist, hist_idx, E1

    hat = torch.empty_like(w)
    q_out = torch.zeros((E, R, K), dtype=torch.int8, device=device)
    for c in (0, 1):
        rows = pl.orig_rows_t[c]
        perm = pl.perm[c]
        rr, cc = rows.unsqueeze(1), perm.unsqueeze(0)
        hat[:, rr, cc] = classes[c]["hat"]
        q_out[:, rr, cc] = classes[c]["q"]
    return hat, s, q_out


def polish(w, H, s, q, hat, fmt, sweeps=2, cubic=0.0, seed=0):
    """Block coordinate descent after babai_round. Per colour, all rings get a new Viterbi path, and a row group
    keeps its change only if the exact objective falls, so this can't diverge the way a plain sweep does."""
    device = w.device
    E, R, K = w.shape
    matrix = "w13" if K == 2560 else "w2"
    lay = layout(matrix, fmt, device)
    bits = FORMATS[fmt]["bits"]
    lev_t = trellis_levels(bits, cubic, device)
    lev_s = scalar_levels(3, cubic, device)
    idx_t, val_t, mid_t, _ = coset_tables(lev_t)
    mid_s = (lev_s[:-1] + lev_s[1:]) / 2
    diag = H.diagonal(dim1=1, dim2=2)
    kp_map = torch.arange(K, device=device) // 128
    G = (hat - w) @ H
    rng = np.random.default_rng(seed)
    colours = list(range(len(lay.colours)))
    n_groups = 640

    def group_of(rows, c):
        # Rows of one row group share rings. w13: gate row r and up row r. w2: rows of a pair of row tiles.
        if matrix == "w13":
            return rows - (640 if c else 0)
        rt, g = rows // 16, rows % 8
        return (rt // 2) * 8 + g

    for _ in range(sweeps):
        rng.shuffle(colours)
        for ci in colours:
            colour = lay.colours[ci]
            parts = []
            for c in colour["cols_t"]:
                rows, cols = colour["rows_t"][c], colour["cols_t"][c]
                rr, cc = rows.unsqueeze(1), cols.unsqueeze(0)
                d = diag[:, cols].unsqueeze(1)
                s_pos = s[:, rows.unsqueeze(1), kp_map[cols].unsqueeze(0)]
                cur = hat[:, rr, cc]
                g_pos = G[:, rr, cc]
                tau = (w[:, rr, cc] - (g_pos - (cur - w[:, rr, cc]) * d) / d) / s_pos
                parts.append(dict(c=c, rows=rows, cols=cols, rr=rr, cc=cc, d=d, s=s_pos, cur=cur, g=g_pos, tau=tau))
            if colour["kind"] == "spare":
                for part in parts:
                    q_new = (part["tau"].unsqueeze(-1) > mid_s).sum(-1)
                    part["q"], part["new"] = q_new, lev_s[q_new] * part["s"]
            else:
                flat_tau = torch.cat([part["tau"].flatten(1) for part in parts], 1)
                flat_d = torch.cat([(part["d"] * part["s"] * part["s"]).flatten(1) for part in parts], 1)
                m = (flat_tau.unsqueeze(-1).unsqueeze(-1) > mid_t).sum(-1)
                values = val_t[torch.arange(4, device=device).view(1, 1, 4), m]
                cost = flat_d.unsqueeze(-1) * (values - flat_tau.unsqueeze(-1)) ** 2
                rings = colour["rings_t"]
                N, T = rings.shape
                bits_u = tb_viterbi(cost[:, rings].reshape(E * N, T, 4))
                lab = labels_from_bits(bits_u).reshape(E, N, T)
                flat_lab = torch.empty_like(m[:, :, 0])
                flat_lab[:, rings] = lab
                chosen = idx_t[flat_lab, m.gather(2, flat_lab.unsqueeze(-1)).squeeze(-1)]
                start = 0
                for part in parts:
                    n = part["tau"].shape[1] * part["tau"].shape[2]
                    part["q"] = chosen[:, start:start + n].reshape(part["tau"].shape)
                    part["new"] = lev_t[part["q"]] * part["s"]
                    start += n
            change = torch.zeros(E, n_groups, device=device)
            for part in parts:
                delta = part["new"] - part["cur"]
                part["delta"] = delta
                part["group"] = group_of(part["rows"], part["c"])
                Hcc = H[:, part["cols"]][:, :, part["cols"]]
                per_row = 2 * (delta * part["g"]).sum(-1) + ((delta @ Hcc) * delta).sum(-1)
                change.index_add_(1, part["group"], per_row)
            keep = change < 0
            for part in parts:
                mask = keep[:, part["group"]].unsqueeze(-1)
                delta = part["delta"] * mask
                hat[:, part["rr"], part["cc"]] = part["cur"] + delta
                q[:, part["rr"], part["cc"]] = torch.where(mask, part["q"].to(torch.int8), q[:, part["rr"], part["cc"]])
                G[:, part["rows"]] += delta @ H[:, part["cols"]]
    return hat, q
