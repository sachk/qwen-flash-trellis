#!/usr/bin/env python3
"""S2: one teacher pass over the FP8 source, producing everything downstream needs.

Per layer: routing statistics, per-expert GPTQ library, QSA K/V absmax. At the end:
teacher top-64 logits for the eval sets and the head Hessian. The forward pass stays on
the unquantised weights, so quantisation never feeds into later layers.

Attention runs per sequence. Everything else has no sequence dependence and runs over
all tokens at once, which is what makes 512 experts x 48 layers tractable.
"""
import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoConfig
from transformers.models.qwen4_exp import modeling_qwen4_exp as model_code

from requant_common import FP8_SRC, LIVE, ROOT, Checkpoint, Job, pack

DEVICE = "cuda"
BF16 = torch.bfloat16
NUM_EXPERTS = 512
PREFIX = "model.language_model."
SETS = ROOT / "data/sets"
EDGE_LAYERS = {*range(12), *range(36, 48)}
CLIP_RATIOS = (1.0, 0.85, 0.7, 0.6, 0.5, 0.42, 0.35, 0.3)
HIDDEN_CHECKPOINT_EVERY = 8
QUERY_CHUNK = 512
TOKEN_CHUNK = 65536
TOP_LOGPROBS = 64
PLE_THREADS = 32
HOST_CHUNK = 1 << 16


class Stop(Exception):
    """SIGTERM arrived: leave at a checkpoint and exit cleanly."""


class Sequences:
    """All token sequences in one tensor, with their set, session and scored spans."""

    def __init__(self, smoke):
        self.ids, self.names, self.calib, self.spans = [], [], [], []
        for name in ("calib", "eval_public", "eval_nemotron", "eval_multilingual"):
            path = SETS / f"{name}.safetensors"
            with safe_open(path, framework="np") as f:
                ids, offsets, meta = f.get_tensor("input_ids"), f.get_tensor("offsets"), f.metadata()
            spans = json.loads(meta["score_spans"]) if "score_spans" in meta else None
            limit = {"calib": 16_000, "eval_public": 4_000, "eval_nemotron": 0, "eval_multilingual": 0}[name] if smoke else None
            used = 0
            for i in range(len(offsets) - 1):
                sequence = ids[offsets[i]:offsets[i + 1]]
                if limit is not None and used + len(sequence) > limit:
                    if used > 0 or limit == 0:
                        break
                    sequence = sequence[:limit]
                used += len(sequence)
                self.ids.append(torch.from_numpy(sequence.astype(np.int64)))
                self.names.append(name)
                self.calib.append(name == "calib")
                self.spans.append(spans[i] if spans else [[1, len(sequence)]])
        self.lengths = [len(s) for s in self.ids]
        self.offsets = np.cumsum([0, *self.lengths])
        self.total = int(self.offsets[-1])
        self.calib_mask = torch.cat([torch.full((n,), c, dtype=torch.bool) for n, c in zip(self.lengths, self.calib)]).to(DEVICE)
        self.eval_mask = torch.cat([torch.full((n,), name == "eval_public", dtype=torch.bool) for n, name in zip(self.lengths, self.names)]).to(DEVICE)


class PleLookup(torch.nn.Module):
    """The n-gram table, read from the live snapshot's BF16 shards instead of 95 GiB of VRAM."""

    def __init__(self, live, layer, device):
        super().__init__()
        self.register_buffer("weight", torch.empty(0, device=device), persistent=False)
        self.shards = []
        base = f"{PREFIX}layers.{layer}.ple.ple_embedding.ngram_embedding.shard_"
        index = 0
        fds = {}
        while f"{base}{index}.weight" in live:
            path, offset, length, entry = live.location(f"{base}{index}.weight")
            self.rows, self.width = entry["shape"]
            if path not in fds:
                fds[path] = os.open(path, os.O_RDONLY)
                os.posix_fadvise(fds[path], 0, 0, os.POSIX_FADV_RANDOM)
            self.shards.append((fds[path], offset))
            index += 1
        self.device = device
        self.pool = ThreadPoolExecutor(PLE_THREADS)

    def _read(self, rows, out):
        row_bytes = self.width * 2
        for position, row in enumerate(rows.tolist()):
            fd, offset = self.shards[row // self.rows]
            os.preadv(fd, [out[position]], offset + (row % self.rows) * row_bytes)

    def forward(self, ids):
        unique, inverse = torch.unique(ids.reshape(-1), return_inverse=True)
        rows = unique.cpu().numpy()
        out = np.empty((len(rows), self.width), dtype=np.int16)
        parts = np.array_split(np.arange(len(rows)), PLE_THREADS)
        list(self.pool.map(lambda part: self._read(rows[part], out[part[0]:part[-1] + 1]) if len(part) else None, parts))
        values = torch.from_numpy(out).to(self.device).view(BF16)
        return values[inverse].reshape(*ids.shape, self.width)


def tensor_for(name, source, live, device):
    if name in source and (FP8_SRC / source.weight_map[name]).exists():
        return source.weight(name, device).to(BF16)
    return live.read(name, device).to(BF16) if name in live else None


def build_layer(config, index, source, live):
    with torch.device("meta"):
        layer = model_code.Qwen4ExpTextDecoderLayer(config, index)
    if layer.ple is not None:
        layer.ple.ple_embedding.ngram_embedding = torch.nn.Identity()
    layer.mlp.experts._parameters.pop("gate_up_proj")
    layer.mlp.experts._parameters.pop("down_proj")
    layer.to_empty(device=DEVICE)
    prefix = f"{PREFIX}layers.{index}."
    for name, parameter in layer.named_parameters():
        if ".experts." in name:
            continue
        value = tensor_for(prefix + name, source, live, DEVICE)
        if value is None:
            raise KeyError(prefix + name)
        parameter.data = value.reshape(parameter.shape)
    if layer.ple is not None:
        embedding = layer.ple.ple_embedding
        embedding.ngram_embedding = PleLookup(live, index, DEVICE)
        embedding.layer_multipliers = live.read(f"{prefix}ple.ple_embedding.layer_multipliers", DEVICE).long()
        embedding.ngram_heads_vocab_sizes = torch.tensor(embedding.head_vocab_sizes, dtype=torch.long, device=DEVICE)
        embedding.ngram_heads_offsets = torch.tensor(embedding.head_offsets, dtype=torch.long, device=DEVICE)
    return layer.requires_grad_(False).eval()


def load_experts(index, source):
    """Dequantised weights: gate_up [E, 1280, 2560] and down [E, 2560, 640], BF16 in VRAM."""
    gate_up, down = [], []
    for expert in range(NUM_EXPERTS):
        base = f"{PREFIX}layers.{index}.mlp.experts.{expert}."
        gate_up.append(torch.cat([source.weight(base + "gate_proj.weight", DEVICE), source.weight(base + "up_proj.weight", DEVICE)]).to(BF16))
        down.append(source.weight(base + "down_proj.weight", DEVICE).to(BF16))
    return torch.stack(gate_up), torch.stack(down)


def qsa_forward(attention, hidden, cos, sin, stats=None):
    """Same maths as Qwen4ExpTextAttention with its indexer, vectorised over query chunks."""
    indexer = attention.indexer
    T = hidden.shape[0]
    head_dim, ratio = indexer.index_head_dim, indexer.compress_ratio
    qk = indexer.index_qk_proj(hidden)
    q_index, keys = torch.split(qk, [indexer.index_n_heads * head_dim, indexer.index_kv_heads * head_dim], dim=-1)
    q_index = indexer.q_layernorm(q_index.reshape(T, -1, head_dim))
    q_index = _rotate(q_index, cos[0, :T], sin[0, :T])

    blocks = T // ratio
    if blocks:
        pooled = keys[: blocks * ratio].float().reshape(blocks, ratio, head_dim).mean(1).to(hidden.dtype)
        pooled = indexer.k_layernorm(pooled)
        starts = torch.arange(blocks, device=hidden.device) * ratio
        block_keys = _rotate(pooled.unsqueeze(1), cos[0, starts], sin[0, starts]).squeeze(1).float()

    shape = (T, -1, attention.head_dim)
    query, gate = torch.chunk(attention.q_proj(hidden).view(T, -1, attention.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(T, -1)
    query = _rotate(attention.q_norm(query), cos[0, :T], sin[0, :T]).transpose(0, 1)
    key = _rotate(attention.k_norm(attention.k_proj(hidden).view(shape)), cos[0, :T], sin[0, :T]).transpose(0, 1)
    value = attention.v_proj(hidden).view(shape).transpose(0, 1)
    if stats is not None:
        stats["k"] = max(stats["k"], float(key.float().abs().max()))
        stats["v"] = max(stats["v"], float(value.float().abs().max()))
    groups = attention.num_key_value_groups
    key = key.repeat_interleave(groups, 0)
    value = value.repeat_interleave(groups, 0)

    outputs = []
    positions = torch.arange(T, device=hidden.device)
    for start in range(0, T, QUERY_CHUNK):
        queries = positions[start:start + QUERY_CHUNK]
        complete = (queries + 1) // ratio
        visible = positions.unsqueeze(0) <= queries.unsqueeze(1)
        tail = positions.unsqueeze(0) >= (complete * ratio).unsqueeze(1)
        selected = visible & tail
        if blocks:
            scores = (q_index[queries].float() @ block_keys.T).relu().sum(1) / math.sqrt(head_dim)
            usable = torch.arange(blocks, device=hidden.device).unsqueeze(0) < complete.unsqueeze(1)
            scores = scores.masked_fill(~usable, -math.inf)
            top = scores.topk(min(indexer.block_topk, blocks), dim=1).indices
            block_mask = torch.zeros_like(scores, dtype=torch.bool).scatter(1, top, True) & usable
            selected[:, : blocks * ratio] |= block_mask.repeat_interleave(ratio, 1)
        scored = (query[:, queries] @ key.transpose(1, 2)) * attention.scaling
        scored = scored.masked_fill(~selected.unsqueeze(0), -math.inf)
        weights = F.softmax(scored, dim=-1, dtype=torch.float32).to(hidden.dtype)
        outputs.append((weights @ value).transpose(0, 1).reshape(len(queries), -1))
        del scored, weights
    output = torch.cat(outputs) * torch.sigmoid(gate)
    return attention.o_proj(output)


def _rotate(x, cos, sin):
    """RoPE on [N, heads, dim] with [N, rope_dim] tables, covering rope_dim of dim."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    rope = cos.shape[-1]
    head, rest = x[..., :rope], x[..., rope:]
    half = rope // 2
    rotated = torch.cat([-head[..., half:], head[..., :half]], dim=-1)
    return torch.cat([head * cos + rotated * sin, rest], dim=-1)


def expert_output(gate_up, down, x):
    gate, up = (x @ gate_up.T).chunk(2, dim=-1)
    h = F.silu(gate) * up
    return h, h @ down.T


def run_moe(mlp, gate_up, down, mixed, calib_mask, want_hessians=True):
    """Teacher MoE over all tokens, with routing statistics and gate-weighted Hessians."""
    N = mixed.shape[0]
    router = mlp.gate.weight
    top_values, top_indices = [], []
    output = torch.zeros(N, mixed.shape[1], dtype=torch.float32, device=mixed.device)
    for start in range(0, N, TOKEN_CHUNK):
        x = mixed[start:start + TOKEN_CHUNK]
        probabilities = F.softmax(x @ router.T, dim=-1, dtype=torch.float32)
        values, indices = probabilities.topk(mlp.gate.top_k, dim=-1)
        if mlp.gate.norm_topk_prob:
            values = values / values.sum(dim=-1, keepdim=True)
        top_values.append(values)
        top_indices.append(indices)
        output[start:start + TOKEN_CHUNK] = (torch.sigmoid(mlp.shared_expert_gate(x)) * mlp.shared_expert(x)).float()
    top_values, top_indices = torch.cat(top_values), torch.cat(top_indices)
    tokens = torch.arange(N, device=mixed.device).repeat_interleave(mlp.gate.top_k)
    routes = top_indices.reshape(-1)
    route_weights = top_values.reshape(-1)
    order = routes.argsort()
    tokens, routes, route_weights = tokens[order], routes[order], route_weights[order]
    bounds = torch.searchsorted(routes, torch.arange(NUM_EXPERTS + 1, device=routes.device)).tolist()

    stats = {key: torch.zeros(NUM_EXPERTS, dtype=torch.float64) for key in ("routed", "routed_all", "reap", "man", "gate2")}
    h13 = torch.zeros(NUM_EXPERTS, 2560, 2560, dtype=torch.float32, device=mixed.device) if want_hessians else None
    h2 = torch.zeros(NUM_EXPERTS, 640, 640, dtype=torch.float32, device=mixed.device) if want_hessians else None
    for expert in range(NUM_EXPERTS):
        where = tokens[bounds[expert]:bounds[expert + 1]]
        if not len(where):
            continue
        weights = route_weights[bounds[expert]:bounds[expert + 1]]
        x = mixed[where]
        h, out = expert_output(gate_up[expert], down[expert], x)
        out = out.float()
        output.index_add_(0, where, out * weights.unsqueeze(1))
        stats["routed_all"][expert] = len(where)
        is_calib = calib_mask[where]
        if not is_calib.any():
            continue
        norm = out[is_calib].norm(dim=1)
        g = weights[is_calib]
        stats["routed"][expert] = int(is_calib.sum())
        stats["reap"][expert] = float((g * norm).sum())
        stats["man"][expert] = float(norm.sum())
        stats["gate2"][expert] = float((g * g).sum())
        if want_hessians:
            weighted = g.unsqueeze(1)
            xw = x[is_calib].float() * weighted
            hw = h[is_calib].float() * weighted
            h13[expert] = xw.T @ xw
            h2[expert] = hw.T @ hw
    return output, stats, h13, h2


def condition(hessians, gate2, routed):
    """Normalise; blend cold experts toward the layer pool (rank-deficient otherwise); damp."""
    K = hessians.shape[-1]
    normal = gate2.to(hessians.device).float().clamp_min(1e-12).view(-1, 1, 1)
    hessians /= normal
    used = routed > 0
    pool = hessians[used.to(hessians.device)].mean(0)
    n_min = 2 * K
    blended = routed < n_min
    alpha = (n_min / routed.clamp_min(1)).clamp_max(1e6).float().to(hessians.device)
    for expert in torch.nonzero(blended).flatten().tolist():
        hessians[expert] += alpha[expert] * pool
    mean_diag = hessians.diagonal(dim1=1, dim2=2).mean(1).clamp_min(1e-8)
    hessians.diagonal(dim1=1, dim2=2).add_(0.01 * mean_diag.unsqueeze(1))
    return hessians, blended


def inverse_factor(H):
    """Upper factor U with H^-1 = U^T U, damping harder any expert that fails to factor."""
    factor, info = torch.linalg.cholesky_ex(H)
    damp = 0.1
    while info.any():
        for expert in torch.nonzero(info).flatten().tolist():
            H[expert].diagonal().add_(damp * H[expert].diagonal().mean())
        factor, info = torch.linalg.cholesky_ex(H)
        damp *= 10
    return torch.linalg.cholesky(torch.cholesky_inverse(factor), upper=True)


def group_scales(w, diag, bits, group):
    """Static per-row group scales from a clip search weighted by diag(H). Rounded to F16."""
    E, R, K = w.shape
    G = K // group
    grouped = w.reshape(E, R, G, group)
    weights = diag.reshape(E, 1, G, group)
    amax = grouped.abs().amax(-1)
    best, best_error = None, None
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    for ratio in CLIP_RATIOS:
        scale = (ratio * 2 * amax / (2**bits - 1)).clamp_min(1e-6).to(torch.float16).float()
        q = (grouped / scale.unsqueeze(-1)).round().clamp(qmin, qmax)
        error = (((q * scale.unsqueeze(-1) - grouped) ** 2) * weights).sum(-1)
        if best is None:
            best, best_error = scale, error
        else:
            better = error < best_error
            best = torch.where(better, scale, best)
            best_error = torch.where(better, error, best_error)
    return best


def gptq(w, hessians, bits, group, chunk=64):
    """Blocked error-feedback GPTQ with fixed scales, in batches of experts. w: [E, R, K]."""
    E, R, K = w.shape
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    all_q = torch.empty(E, R, K, dtype=torch.int8, device="cpu")
    all_scales = torch.empty(E, R, K // group, dtype=torch.float16, device="cpu")
    errors = torch.empty(E)
    for begin in range(0, E, chunk):
        batch = range(begin, min(begin + chunk, E))
        original = w[batch.start:batch.stop].float()
        W = original.clone()
        H = hessians[batch.start:batch.stop].clone()
        scales = group_scales(W, H.diagonal(dim1=1, dim2=2), bits, group)
        all_scales[batch.start:batch.stop] = scales.to(torch.float16).cpu()
        expanded = scales.repeat_interleave(group, 2)
        upper = inverse_factor(H)
        Q = torch.empty_like(W)
        for i1 in range(0, K, 128):
            i2 = min(i1 + 128, K)
            R1 = W[:, :, i1:i2].clone()
            U11 = upper[:, i1:i2, i1:i2]
            E1 = torch.zeros_like(R1)
            for i in range(i2 - i1):
                s = expanded[:, :, i1 + i]
                q = (R1[:, :, i] / s).round().clamp(qmin, qmax)
                Q[:, :, i1 + i] = q
                error = (R1[:, :, i] - q * s) / U11[:, i, i].unsqueeze(1)
                E1[:, :, i] = error
                R1[:, :, i:] -= error.unsqueeze(2) * U11[:, i, i:].unsqueeze(1)
            if i2 < K:
                W[:, :, i2:] -= E1 @ upper[:, i1:i2, i2:]
        all_q[batch.start:batch.stop] = Q.to(torch.int8).cpu()
        delta = Q * expanded - original
        errors[batch.start:batch.stop] = (((delta @ H) * delta).sum((1, 2)) / ((original @ H) * original).sum((1, 2)).clamp_min(1e-12)).sqrt().cpu()
        del W, H, Q, upper, original, delta, expanded
    return all_q, all_scales, errors


def packed_tensors(prefix, q, scales, bits):
    """One expert projection in the live checkpoint's auto_gptq naming and shapes."""
    offset = 2 ** (bits - 1)
    biased = (q.int().numpy() + offset).astype(np.uint32).T  # [K, R]
    words = pack(biased, bits)
    rows = q.shape[0]
    groups = scales.shape[1]
    zeros = np.full((rows, groups), offset - 1, dtype=np.uint32)  # The stored zero is true zero minus 1.
    zero_words = pack(zeros, bits).T  # Stored as [groups, words].
    return {
        prefix + ".qweight": torch.from_numpy(words.copy()),
        prefix + ".qzeros": torch.from_numpy(zero_words.copy()),
        prefix + ".scales": scales.T.clone().contiguous(),
    }


def library(job, out, index, gate_up, down, h13, h2, stats, bit_list):
    gate_h13, blended = condition(h13, stats["gate2"], stats["routed"])
    gate_h2, _ = condition(h2, stats["gate2"], stats["routed"])
    report = {"blended": int(blended.sum())}
    (out / f"lib/L{index}").mkdir(parents=True, exist_ok=True)
    for bits in bit_list:
        path = out / f"lib/L{index}/w{bits}.safetensors"
        group = 64 if bits == 2 else 128
        started = time.time()
        q13, s13, err13 = gptq(gate_up, gate_h13, bits, group)
        q2, s2, err2 = gptq(down, gate_h2, bits, group)
        tensors = {}
        for expert in range(NUM_EXPERTS):
            base = f"{PREFIX}layers.{index}.mlp.experts.{expert}."
            tensors |= packed_tensors(base + "gate_proj", q13[expert, :640], s13[expert, :640], bits)
            tensors |= packed_tensors(base + "up_proj", q13[expert, 640:], s13[expert, 640:], bits)
            tensors |= packed_tensors(base + "down_proj", q2[expert], s2[expert], bits)
        temporary = path.with_suffix(".tmp")
        save_file(tensors, temporary)
        temporary.replace(path)
        report[f"w{bits}"] = {"err13_mean": float(err13.mean()), "err2_mean": float(err2.mean()), "seconds": round(time.time() - started, 1)}
        job.log(f"L{index} W{bits}: err13 {err13.mean():.4f} err2 {err2.mean():.4f} in {time.time() - started:.0f}s")
    return report


def dequant_live(live, index, device):
    gate_up, down = [], []
    from requant_common import unpack

    def tensor(base, bits, group):
        words = live.read(base + ".qweight").numpy()
        scales = live.read(base + ".scales").float()
        k = words.shape[0] * 32 // bits
        q = torch.from_numpy(unpack(words, bits, k)).float() - 2 ** (bits - 1)
        return (q.T * scales.repeat_interleave(group, 0).T).to(device)

    bits, group = (3, 128) if index in EDGE_LAYERS else (2, 64)
    for expert in range(NUM_EXPERTS):
        base = f"{PREFIX}layers.{index}.mlp.experts.{expert}."
        gate_up.append(torch.cat([tensor(base + "gate_proj", bits, group), tensor(base + "up_proj", bits, group)]))
        down.append(tensor(base + "down_proj", bits, group))
    return torch.stack(gate_up), torch.stack(down)


def smoke_compare(job, out, index, router, gate_up, down, mixed, weights_mask, live):
    """GPTQ W3 vs the live AutoRound weights, on the same held-out own-traffic tokens."""
    held = mixed[weights_mask]
    if not len(held):
        return {}
    library_file = load_file(out / f"lib/L{index}/w3.safetensors", device="cpu")
    from requant_common import unpack

    def own(base):
        words = library_file[base + ".qweight"].numpy()
        scales = library_file[base + ".scales"].float()
        q = torch.from_numpy(unpack(words, 3, words.shape[0] * 32 // 3)).float() - 4
        return (q.T * scales.repeat_interleave(128, 0).T).to(DEVICE)

    ours_up = torch.stack([torch.cat([own(f"{PREFIX}layers.{index}.mlp.experts.{e}.gate_proj"), own(f"{PREFIX}layers.{index}.mlp.experts.{e}.up_proj")]) for e in range(NUM_EXPERTS)])
    ours_down = torch.stack([own(f"{PREFIX}layers.{index}.mlp.experts.{e}.down_proj") for e in range(NUM_EXPERTS)])
    live_up, live_down = dequant_live(live, index, DEVICE)
    probe = held.float()
    routed = (probe @ router.float().T).topk(10, dim=-1).indices
    ours_err, live_err, reference = [], [], []
    for expert in range(NUM_EXPERTS):
        tokens = (routed == expert).any(dim=1)
        if not tokens.any():
            continue
        probe_e = probe[tokens]
        _, teacher = expert_output(gate_up[expert].float(), down[expert].float(), probe_e)
        _, ours = expert_output(ours_up[expert], ours_down[expert], probe_e)
        _, theirs = expert_output(live_up[expert], live_down[expert], probe_e)
        reference.append(float(teacher.norm()))
        ours_err.append(float((ours - teacher).norm()))
        live_err.append(float((theirs - teacher).norm()))
    reference, ours_err, live_err = map(np.array, (reference, ours_err, live_err))
    result = {
        "layer": index, "probe_tokens": len(probe),
        "gptq_rel_error": float(ours_err.sum() / reference.sum()),
        "autoround_rel_error": float(live_err.sum() / reference.sum()),
        "gptq_wins_experts": int((ours_err < live_err).sum()),
    }
    job.log(f"smoke compare {result}")
    return result


def module_check(job, config, index, source, live, sequence, hidden, cos, sin):
    """Our pipeline against the module's own forward, on a short sequence: BF16 noise only."""
    from transformers.masking_utils import create_causal_mask

    layer = build_layer(config, index, source, live)
    gate_up, down = load_experts(index, source)
    layer.mlp.experts.gate_up_proj = torch.nn.Parameter(gate_up, requires_grad=False)
    layer.mlp.experts.down_proj = torch.nn.Parameter(down, requires_grad=False)
    T = hidden.shape[0]
    rows = hidden.unsqueeze(0)
    positions = torch.arange(T, device=DEVICE).unsqueeze(0)
    mask = create_causal_mask(config=config, inputs_embeds=rows, attention_mask=None, past_key_values=None, position_ids=positions, allow_is_causal_skip=False)
    reference = layer(rows, position_embeddings=(cos[:, :T], sin[:, :T]), attention_mask=mask, conv_mask=None, past_key_values=None, ple_input_ids=sequence.unsqueeze(0))[0]

    h = rows
    if layer.ple is not None:
        h = h + layer.ple(h, sequence.unsqueeze(0), None, conv_mask=None)
    mixed, hyper, inject = layer.attn_hyper_connection(h)
    if layer.layer_type != "linear_attention":
        attended = qsa_forward(layer.self_attn, mixed[0], cos, sin).unsqueeze(0)
    else:
        attended = layer.linear_attn(mixed, cache_params=None, attention_mask=None)
    h = hyper + (attended.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2)
    mixed, hyper, inject = layer.mlp_hyper_connection(h)
    output, *_ = run_moe(layer.mlp, gate_up, down, mixed[0], torch.zeros(T, dtype=torch.bool, device=DEVICE), want_hessians=False)
    ours = (hyper + (output.to(BF16).unsqueeze(0).unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2))[0]
    error = float((ours.float() - reference.float()).norm() / reference.float().norm())
    job.log(f"module check L{index} ({layer.layer_type}): relative difference {error:.2e}")
    del layer, gate_up, down
    torch.cuda.empty_cache()
    if error > 5e-3:
        raise SystemExit(f"layer {index} does not match the module; refusing to build a library")
    return error


def save_hidden(hidden, path):
    """17 GB does not fit in host RAM: stream raw BF16 in chunks."""
    temporary = path.with_suffix(".tmp")
    with open(temporary, "wb") as f:
        for start in range(0, hidden.shape[0], HOST_CHUNK):
            f.write(hidden[start:start + HOST_CHUNK].view(torch.int16).cpu().numpy().tobytes())
    temporary.replace(path)


def load_hidden(path, total):
    hidden = torch.empty(total, 10240, dtype=BF16, device=DEVICE)
    with open(path, "rb") as f:
        for start in range(0, total, HOST_CHUNK):
            count = min(HOST_CHUNK, total - start)
            array = np.frombuffer(f.read(count * 20480), dtype=np.int16).reshape(count, 10240)
            hidden[start:start + count] = torch.from_numpy(array.copy()).to(DEVICE).view(BF16)
    return hidden


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--edge-bits", default="3")
    parser.add_argument("--middle-bits", default="3,4")
    args = parser.parse_args()

    step = "s2-smoke" if args.smoke else "s2"
    job = Job(step)
    out = job.dir
    edge_bits = [int(b) for b in args.edge_bits.split(",") if b]
    middle_bits = [int(b) for b in args.middle_bits.split(",") if b]
    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        raise SystemExit("no GPU")

    config = AutoConfig.from_pretrained(FP8_SRC).text_config
    config._attn_implementation = "eager"
    source, live = Checkpoint(FP8_SRC), Checkpoint(LIVE)
    sequences = Sequences(args.smoke)
    job.log(f"{len(sequences.ids)} sequences, {sequences.total} tokens, {int(sequences.calib_mask.sum())} calibration")
    ids = [s.to(DEVICE) for s in sequences.ids]

    rotary = model_code.Qwen4ExpTextRotaryEmbedding(config=config).to(DEVICE)
    longest = max(sequences.lengths)
    positions = torch.arange(longest, device=DEVICE).view(1, 1, -1).expand(3, 1, -1)
    cos, sin = rotary(torch.zeros(1, longest, 2560, dtype=BF16, device=DEVICE), positions)

    progress = job.progress()
    first = progress.get("next_layer", 0)
    hidden = None
    if first:
        hidden = load_hidden(out / "hidden.bin", sequences.total)
        job.log(f"resumed before layer {first}")
    else:
        embed = source.weight(PREFIX + "embed_tokens.weight", DEVICE) if PREFIX + "embed_tokens.weight" in source and (FP8_SRC / source.weight_map[PREFIX + "embed_tokens.weight"]).exists() else live.read(PREFIX + "embed_tokens.weight", DEVICE)
        embed = embed.to(BF16)
        hidden = torch.empty(sequences.total, 10240, dtype=BF16, device=DEVICE)
        for i, sequence in enumerate(ids):
            start, end = sequences.offsets[i], sequences.offsets[i + 1]
            hidden[start:end] = embed[sequence].repeat(1, 4)
        del embed

    if args.smoke and not first:
        short = ids[0][:300]
        for index in (0, 1, 3):
            module_check(job, config, index, source, live, short, hidden[:300], cos, sin)

    def check_stop():
        if job.stop:
            raise Stop

    reports = progress.get("reports", {})
    try:
        for index in range(first, args.layers):
            started = time.time()
            layer = build_layer(config, index, source, live)
            is_qsa = layer.layer_type != "linear_attention"
            kv = {"k": 0.0, "v": 0.0}
            for i, sequence in enumerate(ids):
                check_stop()
                start, end = int(sequences.offsets[i]), int(sequences.offsets[i + 1])
                h = hidden[start:end].unsqueeze(0)
                if layer.ple is not None:
                    h = h + layer.ple(h, sequence.unsqueeze(0), None, conv_mask=None)
                mixed, hyper, inject = layer.attn_hyper_connection(h)
                if is_qsa:
                    attended = qsa_forward(layer.self_attn, mixed[0], cos, sin, kv if sequences.calib[i] else None).unsqueeze(0)
                else:
                    attended = layer.linear_attn(mixed, cache_params=None, attention_mask=None)
                hidden[start:end] = (hyper + (attended.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2))[0]
            attention_seconds = time.time() - started

            # Everything from here is free of sequence order: run it over all tokens.
            mixed = torch.empty(sequences.total, 2560, dtype=BF16, device=DEVICE)
            inject = torch.empty(sequences.total, 4, dtype=BF16, device=DEVICE)
            for start in range(0, sequences.total, TOKEN_CHUNK):
                part = hidden[start:start + TOKEN_CHUNK]
                m, _, i = layer.mlp_hyper_connection(part)
                mixed[start:start + TOKEN_CHUNK], inject[start:start + TOKEN_CHUNK] = m, i
            check_stop()
            moe_started = time.time()
            gate_up, down = load_experts(index, source)
            output, stats, h13, h2 = run_moe(layer.mlp, gate_up, down, mixed, sequences.calib_mask)
            moe_seconds = time.time() - moe_started
            for start in range(0, sequences.total, TOKEN_CHUNK):
                part_output = output[start:start + TOKEN_CHUNK].to(BF16)
                hidden[start:start + TOKEN_CHUNK] += (part_output.unsqueeze(-2) * inject[start:start + TOKEN_CHUNK].unsqueeze(-1)).flatten(-2)
            check_stop()

            bit_list = edge_bits if index in EDGE_LAYERS else middle_bits
            del output, inject
            if not args.smoke:
                del mixed
            report = library(job, out, index, gate_up, down, h13, h2, stats, bit_list)
            if args.smoke:
                report["compare"] = smoke_compare(job, out, index, layer.mlp.gate.weight, gate_up, down, mixed, sequences.eval_mask, live)
            stats_dir = out / "stats"
            stats_dir.mkdir(exist_ok=True)
            save_file({k: v.to(torch.float64) for k, v in stats.items()}, stats_dir / f"L{index}.safetensors", metadata={"qsa_k": str(kv["k"]), "qsa_v": str(kv["v"])})
            report |= {"attention_s": round(attention_seconds, 1), "moe_s": round(moe_seconds, 1), "total_s": round(time.time() - started, 1), "kv": kv}
            reports[str(index)] = report
            job.log(f"L{index} ({'qsa' if is_qsa else 'gdn'}) attn {attention_seconds:.0f}s moe {moe_seconds:.0f}s total {time.time() - started:.0f}s; gpu {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
            del layer, gate_up, down, stats, h13, h2
            mixed = None
            torch.cuda.empty_cache()
            if (index + 1) % HIDDEN_CHECKPOINT_EVERY == 0 and index + 1 < args.layers:
                save_hidden(hidden, out / "hidden.bin")
                job.save_progress({"next_layer": index + 1, "reports": reports})
                job.log(f"checkpoint before layer {index + 1}")
            check_stop()

        if args.layers == 48:
            finish(job, out, config, source, live, hidden, sequences)
        job.save_progress({"next_layer": args.layers, "reports": reports})
        job.done({"tokens": sequences.total, "reports": reports})
    except Stop:
        job.log("stopped cleanly; resume restarts from the last hidden-state checkpoint")


def finish(job, out, config, source, live, hidden, sequences):
    """Final mixer, head Hessian from calibration tokens, teacher top-64 for eval tokens."""
    mixer = model_code.Qwen4ExpTextGatedResidual(config, use_combine=False)
    for name, parameter in mixer.named_parameters():
        parameter.data = tensor_for(f"{PREFIX}hyper_connection_mixer.{name}", source, live, DEVICE)
    mixer = mixer.to(DEVICE).requires_grad_(False)
    head = tensor_for("lm_head.weight", source, live, DEVICE)
    teacher = out / "teacher"
    teacher.mkdir(exist_ok=True)
    h_head = torch.zeros(2560, 2560, dtype=torch.float64, device=DEVICE)
    records = {}
    for i, name in enumerate(sequences.names):
        start, end = int(sequences.offsets[i]), int(sequences.offsets[i + 1])
        final = mixer(hidden[start:end])
        if sequences.calib[i]:
            f = final.float()
            h_head += (f.T @ f).double()
            continue
        logits_positions = []
        for begin, stop in sequences.spans[i]:
            # logits at position p predict token p+1; a scored token at b needs position b-1.
            logits_positions.extend(range(max(begin - 1, 0), min(stop - 1, end - start - 1)))
        where = torch.tensor(logits_positions, device=DEVICE, dtype=torch.long)
        ids, logprobs, rest = [], [], []
        for chunk in where.split(4096):
            logits = final[chunk].float() @ head.float().T
            log_all = F.log_softmax(logits, dim=-1)
            values, indices = log_all.topk(TOP_LOGPROBS, dim=-1)
            ids.append(indices.int().cpu())
            logprobs.append(values.to(torch.float16).cpu())
            rest.append(((1 - values.exp().sum(-1)).clamp_min(1e-9)).float().cpu())
        record = records.setdefault(name, {"sequence": [], "position": [], "ids": [], "logprobs": [], "tail": []})
        record["sequence"].append(torch.full((len(where),), i, dtype=torch.int32))
        record["position"].append(where.int().cpu())
        record["ids"].append(torch.cat(ids))
        record["logprobs"].append(torch.cat(logprobs))
        record["tail"].append(torch.cat(rest))
        if job.stop:
            raise Stop
    for name, record in records.items():
        save_file({k: torch.cat(v) for k, v in record.items()}, teacher / f"{name}.safetensors")
    save_file({"h_head": h_head.float().cpu()}, out / "head_hessian.safetensors")
    job.log(f"teacher sets: {sorted(records)}")


if __name__ == "__main__":
    main()
