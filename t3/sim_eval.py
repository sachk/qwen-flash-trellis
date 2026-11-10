#!/usr/bin/env python3
"""Teacher-forced student pass: KL of any weight format against the stored teacher, with no server or kernels.

Each component can come from the source or from a candidate, so KL can be attributed:
  --experts src | ckpt:PATH | lib:DIR
  --dense   src | ckpt:PATH     (W8 projections, routers, shared experts, hyperconnections)
  --head    src | int8:PATH     (the startup INT8 packer applied to PATH's BF16 head)
  --kv      none | fp8          (QSA K/V rounded through FP8 E4M3)
The estimator matches s4_eval.kl_position: union of top-64, imputed tails. Its results are comparable with
server runs; check that with --experts ckpt:R --dense ckpt:R --head int8:R --kv fp8.
"""
import argparse, json, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen4_exp import modeling_qwen4_exp as model_code
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import s2_pass as s
from requant_common import FP8_SRC, LIVE, ROOT, Checkpoint, Job, unpack

DEVICE = "cuda"
BF16 = torch.bfloat16
TEACHER = Path(os.environ.get("T3_TEACHER", ROOT / "s2/teacher"))
POOL = ThreadPoolExecutor(12)
WORK = torch.bfloat16


def to_work(layer):
    if WORK == torch.float32:
        for parameter in layer.parameters():
            if parameter.is_floating_point():
                parameter.data = parameter.data.float()
        if layer.ple is not None:
            table = layer.ple.ple_embedding.ngram_embedding
            original = table.forward
            table.forward = lambda ids: original(ids).float()
    return layer


def dequant_np(ck, base, K):
    words = ck.read(base + ".qweight").numpy()
    scales = ck.read(base + ".scales").float().numpy()
    bits = words.shape[0] * 32 // K
    group = K // scales.shape[0]
    q = unpack(words, bits, K).astype(np.float32) - 2 ** (bits - 1)
    return (q.T * np.repeat(scales, group, axis=0).T)


def read_any(ck, name, device):
    if name in ck:
        return ck.read(name, device)
    stem = name[: -len(".weight")] if name.endswith(".weight") else name
    if stem + ".qweight" in ck:
        words = ck.read(stem + ".qweight").numpy()
        scales = ck.read(stem + ".scales").float().numpy()
        return _dequant_w8(words, scales, device)
    return None


def _dequant_w8(words, scales, device):
    K = words.shape[0] * 32 // 8
    group = K // scales.shape[0]
    q = unpack(words, 8, K).astype(np.float32) - 128
    return torch.from_numpy(q.T * np.repeat(scales, group, axis=0).T).to(device)


class Student:
    def __init__(self, a, source, live):
        self.source, self.live = source, live
        self.expert_mode = a.experts
        self.expert_ck = Checkpoint(a.experts.split(":", 1)[1]) if a.experts.startswith("ckpt:") else None
        self.expert_dir = Path(a.experts.split(":", 1)[1]) if a.experts.startswith("lib:") else None
        self.dense_ck = Checkpoint(a.dense.split(":", 1)[1]) if a.dense.startswith("ckpt:") else None
        self.alt_dir = a.alt_lib
        first, _, last = (a.alt_layers or "0-0").partition("-")
        self.alt_layers = set(range(int(first), int(last or first) + 1))
        self.dense_mode = a.dense_mode
        self.hc_w8, self.shared_w8 = a.hc_w8, a.shared_w8
        self.dense_bits, self.dense_group = a.dense_bits, a.dense_group
        self.dense_match = __import__("re").compile(a.dense_match)
        self.head_mode = a.head
        self.kv = a.kv

    def layer(self, config, index):
        layer = s.build_layer(config, index, self.source, self.live)
        if self.dense_ck is not None:
            prefix = f"{s.PREFIX}layers.{index}."
            for name, parameter in layer.named_parameters():
                if ".experts." in name:
                    continue
                stem = (prefix + name)[: -len(".weight")] if name.endswith(".weight") else prefix + name
                if self.dense_mode == "bf16only" and stem + ".qweight" in self.dense_ck:
                    continue                    # W8 projections stay on the source; only the BF16 tensors change
                if self.dense_mode == "w8only" and stem + ".qweight" not in self.dense_ck:
                    continue
                value = read_any(self.dense_ck, prefix + name, DEVICE)
                if value is None:
                    raise KeyError(prefix + name)
                parameter.data = value.to(BF16).reshape(parameter.shape)
        if self.hc_w8 or self.shared_w8:
            for name, parameter in layer.named_parameters():
                if parameter.dim() != 2 or ".experts." in name:
                    continue
                hit = self.hc_w8 and "input_mix_weight" in name
                hit = hit or (self.shared_w8 and "shared_expert." in name)
                if hit:
                    parameter.data = int8_groups(parameter.data).to(BF16)
        if self.dense_bits:
            # Format idea A: dense attention/GDN/indexer projections at fewer bits,
            # symmetric round-to-nearest per row and K group (GPTQ would do better).
            hits = []
            for name, parameter in layer.named_parameters():
                if parameter.dim() == 2 and ".experts." not in name and self.dense_match.search(name):
                    parameter.data = int_groups(parameter.data, self.dense_bits, self.dense_group).to(BF16)
                    hits.append(name)
            if index < 4:
                print(f"layer {index}: dense W{self.dense_bits} g{self.dense_group} on {hits}", flush=True)
            self.dense_hits = getattr(self, "dense_hits", 0) + len(hits)
            if index == 47 and not self.dense_hits:
                raise SystemExit("--dense-bits matched nothing in any layer")
        return to_work(layer)

    def experts(self, index):
        if self.expert_ck is not None:
            def one(e):
                base = f"{s.PREFIX}layers.{index}.mlp.experts.{e}."
                ck = self.expert_ck
                gu = np.concatenate((dequant_np(ck, base + "gate_proj", 2560), dequant_np(ck, base + "up_proj", 2560)))
                return torch.from_numpy(gu).to(DEVICE).to(BF16), torch.from_numpy(dequant_np(ck, base + "down_proj", 640)).to(DEVICE).to(BF16)
            parts = list(POOL.map(one, range(s.NUM_EXPERTS)))
            return torch.stack([p[0] for p in parts]), torch.stack([p[1] for p in parts])
        if self.alt_dir is not None and index in self.alt_layers:
            import t3pack
            return t3pack.load_layer(self.alt_dir, index, DEVICE)
        if self.expert_dir is not None:
            import t3pack
            return t3pack.load_layer(self.expert_dir, index, DEVICE)
        return s.load_experts(index, self.source)

    def head(self):
        if self.head_mode == "16bit:src":
            return s.tensor_for("lm_head.weight", self.source, self.live, DEVICE).float()
        if self.head_mode.startswith("int8:"):
            ck = Checkpoint(self.head_mode.split(":", 1)[1])
            weight = ck.read("lm_head.weight", DEVICE)
            out = []
            for start in range(0, weight.shape[0], 1024):
                rows = weight[start:start + 1024].float()
                scale = (rows.abs().amax(-1) / 127).clamp_min(1e-12).to(BF16).float()
                out.append((rows / scale[:, None]).round().clamp(-127, 127) * scale[:, None])
            return torch.cat(out)
        return s.tensor_for("lm_head.weight", self.source, self.live, DEVICE).float()

    def mixer_tensor(self, name):
        if self.dense_ck is not None:
            value = read_any(self.dense_ck, name, DEVICE)
            if value is not None:
                return value.to(BF16)
        return s.tensor_for(name, self.source, self.live, DEVICE)


def int_groups(w, bits, group=128):
    """Plain symmetric INT<bits> per row and group of K (uint4b8-style for 4 bits: -8..7, scale on 7)."""
    R, K = w.shape
    top = 2 ** (bits - 1) - 1
    out = w.float().clone()
    for start in range(0, K, group):
        part = out[:, start:start + group]
        scale = (part.abs().amax(-1, keepdim=True) / top).clamp_min(1e-12).to(BF16).float()
        out[:, start:start + group] = (part / scale).round().clamp(-top - 1, top) * scale
    return out


def int8_groups(w, group=128):
    """Plain symmetric INT8 per row and group of K, as a W8 checkpoint would store."""
    R, K = w.shape
    out = w.float().clone()
    for start in range(0, K, group):
        part = out[:, start:start + group]
        scale = (part.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-12).to(BF16).float()
        out[:, start:start + group] = (part / scale).round().clamp(-127, 127) * scale
    return out


def fp8_round(x):
    return x.to(torch.float8_e4m3fn).to(x.dtype)


def small_float_round(x, exp_bits, mant_bits, scale=1.0):
    """Nearest value of a small float format with the exponent bias centred; subnormals included."""
    bias = 2 ** (exp_bits - 1) - 1
    mag = (x.float().abs() / scale)
    emax = 2 ** exp_bits - 2 - bias
    max_value = (2 - 2.0 ** -mant_bits) * 2.0 ** emax
    mag = mag.clamp_max(max_value)
    e = torch.floor(torch.log2(mag.clamp_min(1e-30))).clamp_min(1 - bias)
    step = 2.0 ** (e - mant_bits)
    return ((mag / step).round() * step * x.sign() * scale).to(x.dtype)


def group_float_round(x, exp_bits, mant_bits, group=32):
    """Small float with a BF16 scale per group of the last dim (absmax maps to the format's max)."""
    bias = 2 ** (exp_bits - 1) - 1
    max_value = (2 - 2.0 ** -mant_bits) * 2.0 ** (2 ** exp_bits - 2 - bias)
    shape = x.shape
    g = x.float().reshape(*shape[:-1], shape[-1] // group, group)
    scale = (g.abs().amax(-1, keepdim=True) / max_value).clamp_min(1e-30).to(torch.bfloat16).float()
    return small_float_round(g / scale, exp_bits, mant_bits).float().mul(scale).reshape(shape).to(x.dtype)


def qsa_forward(attention, hidden, cos, sin, kv):
    if kv == "none":
        return s.qsa_forward(attention, hidden, cos, sin)
    # Format idea F: "k6" = K in E3M2 with a scale per 32; "v6g" = V likewise; "idx4" = indexer keys E2M1 per 32.
    k_round = (lambda k: group_float_round(k, 3, 2)) if "k6" in kv else fp8_round
    if "v6g" in kv:
        v_round = lambda v: group_float_round(v, 3, 2)
    elif "v6e3m2" in kv:
        v_round = lambda v: small_float_round(v, 3, 2, 0.25)
    elif "v6e2m3" in kv:
        v_round = lambda v: small_float_round(v, 2, 3, 2.0)
    elif "v7" in kv:
        v_round = lambda v: small_float_round(v, 3, 3, 0.5)
    else:
        v_round = fp8_round
    index_round = "idx" in kv
    # Wrap the K and V norms and projections' results by patching the stats hook point: same maths, FP8 round trip.
    indexer = attention.indexer
    T = hidden.shape[0]
    head_dim, ratio = indexer.index_head_dim, indexer.compress_ratio
    qk = indexer.index_qk_proj(hidden)
    q_index, keys = torch.split(qk, [indexer.index_n_heads * head_dim, indexer.index_kv_heads * head_dim], dim=-1)
    q_index = indexer.q_layernorm(q_index.reshape(T, -1, head_dim))
    q_index = s._rotate(q_index, cos[0, :T], sin[0, :T])
    blocks = T // ratio
    if blocks:
        pooled = keys[: blocks * ratio].float().reshape(blocks, ratio, head_dim).mean(1).to(hidden.dtype)
        pooled = indexer.k_layernorm(pooled)
        starts = torch.arange(blocks, device=hidden.device) * ratio
        block_keys = s._rotate(pooled.unsqueeze(1), cos[0, starts], sin[0, starts]).squeeze(1).float()
        if "idx4" in kv:
            block_keys = group_float_round(block_keys, 2, 1)
        elif index_round:
            block_keys = fp8_round(block_keys)
    shape = (T, -1, attention.head_dim)
    query, gate = torch.chunk(attention.q_proj(hidden).view(T, -1, attention.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(T, -1)
    query = s._rotate(attention.q_norm(query), cos[0, :T], sin[0, :T]).transpose(0, 1)
    key = s._rotate(attention.k_norm(attention.k_proj(hidden).view(shape)), cos[0, :T], sin[0, :T]).transpose(0, 1)
    value = attention.v_proj(hidden).view(shape).transpose(0, 1)
    key, value = k_round(key), v_round(value)
    groups = attention.num_key_value_groups
    key = key.repeat_interleave(groups, 0)
    value = value.repeat_interleave(groups, 0)
    outputs = []
    positions = torch.arange(T, device=hidden.device)
    for start in range(0, T, s.QUERY_CHUNK):
        queries = positions[start:start + s.QUERY_CHUNK]
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


def kl_batch(t_ids, t_logp, t_tail, s_ids, s_logp):
    """Vectorised s4_eval.kl_position. Each [n, 64] except t_tail [n]. Returns KL [n], top-1 agreement [n]."""
    p = t_logp.double().exp()
    q = s_logp.double().exp()
    s_tail = (1 - q.sum(1)).clamp_min(1e-9)
    p_tail = 0.5 * torch.minimum(t_tail.double().clamp_min(1e-9), p.min(1).values)
    q_tail = 0.5 * torch.minimum(s_tail, q.min(1).values)
    match = t_ids.unsqueeze(2) == s_ids.unsqueeze(1)             # [n, 64, 64]
    has_s = match.any(2)
    q_for_t = torch.where(has_s, (match.double() * q.unsqueeze(1)).sum(2), q_tail.unsqueeze(1))
    kl = (p * (t_logp.double() - q_for_t.clamp_min(1e-12).log())).sum(1)
    only_s = ~match.any(1)
    kl += (only_s.double() * p_tail.unsqueeze(1) * (p_tail.unsqueeze(1).log() - q.clamp_min(1e-12).log())).sum(1)
    p_sum = p.sum(1) + only_s.double().sum(1) * p_tail
    q_sum = q_for_t.sum(1) + (only_s.double() * q).sum(1)
    p_rest, q_rest = (1 - p_sum).clamp_min(1e-9), (1 - q_sum).clamp_min(1e-9)
    kl += p_rest * (p_rest / q_rest).log()
    return kl.clamp_min(0), t_ids[:, 0] == s_ids[:, 0]


def row_mask(domain, sequence, teacher, first_global, ids, offsets, tok):
    """Which teacher rows count, as s4_eval.score_domain."""
    excluded = set(tok.all_special_ids) | set(tok.convert_tokens_to_ids(["<think>", "</think>"]))
    im_start, im_end = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")
    assistant = tok.encode("assistant", add_special_tokens=False)[0]
    keep = np.zeros(len(teacher["sequence"]), dtype=bool)
    tseq = teacher["sequence"].numpy()
    tpos = teacher["position"].numpy()
    for g in np.unique(tseq).tolist():
        local = g - first_global
        seq = ids[offsets[local]:offsets[local + 1]]
        in_asst = np.zeros(len(seq), dtype=bool)
        speaking = False
        for p, token in enumerate(seq.tolist()):
            if token == im_end:
                speaking = False
            elif token == assistant and p > 0 and seq[p - 1] == im_start:
                speaking = True
            in_asst[p] = speaking
        rows = np.flatnonzero(tseq == g)
        target = tpos[rows] + 1
        ok = ~np.isin(seq[target], list(excluded))
        if domain == "eval_private":
            ok &= in_asst[target]
        keep[rows[ok]] = True
    return keep


def forward_hidden(student, config, seqs, embed_mode="src", log=print):
    """Stream the target layers once over a batch; return pre-final-mixer HC states.

    The [sum(lengths), hc_count*hidden_size] tensor stays on DEVICE. Offsets
    delimit independent sequences; no attention/cache crosses a sequence.
    The timestamp preserves the CLI's layer-pass timing (embedding excluded).
    """
    lengths = [len(x) for x in seqs]
    offs = np.cumsum([0, *lengths])
    total = int(offs[-1])
    ids_gpu = [x.to(DEVICE) for x in seqs]
    rotary = model_code.Qwen4ExpTextRotaryEmbedding(config=config).to(DEVICE)
    longest = max(lengths)
    positions = torch.arange(longest, device=DEVICE).view(1, 1, -1).expand(3, 1, -1)
    cos, sin = rotary(torch.zeros(1, longest, 2560, dtype=BF16, device=DEVICE), positions)
    embed = s.tensor_for(s.PREFIX + "embed_tokens.weight", student.source, student.live, DEVICE).to(BF16)
    if embed_mode == "int8":
        rows = embed.float()
        scale = (rows.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-12)
        embed = ((rows / scale).round() * scale).to(BF16)
        del rows, scale
    hidden = torch.empty(total, 10240, dtype=WORK, device=DEVICE)
    embed = embed.to(WORK)
    for i, q in enumerate(ids_gpu):
        hidden[offs[i]:offs[i + 1]] = embed[q].repeat(1, 4)
    del embed
    calib_mask = torch.zeros(total, dtype=torch.bool, device=DEVICE)
    started_all = time.time()
    for index in range(48):
        started = time.time()
        layer = student.layer(config, index)
        for i, q in enumerate(ids_gpu):
            st, en = int(offs[i]), int(offs[i + 1])
            h = hidden[st:en].unsqueeze(0)
            if layer.ple is not None:
                h = h + layer.ple(h, q.unsqueeze(0), None, conv_mask=None)
            mixed, hyper, inject = layer.attn_hyper_connection(h)
            if layer.layer_type != "linear_attention":
                att = qsa_forward(layer.self_attn, mixed[0], cos, sin, student.kv).unsqueeze(0)
            else:
                att = layer.linear_attn(mixed, cache_params=None, attention_mask=None)
            hidden[st:en] = (hyper + (att.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2))[0]
        mixed = torch.empty(total, 2560, dtype=WORK, device=DEVICE)
        inject = torch.empty(total, 4, dtype=WORK, device=DEVICE)
        for st in range(0, total, s.TOKEN_CHUNK):
            m, _, i = layer.mlp_hyper_connection(hidden[st:st + s.TOKEN_CHUNK])
            mixed[st:st + s.TOKEN_CHUNK], inject[st:st + s.TOKEN_CHUNK] = m, i
        gate_up, down = student.experts(index)
        gate_up, down = gate_up.to(WORK), down.to(WORK)
        output, _, _, _ = s.run_moe(layer.mlp, gate_up, down, mixed, calib_mask, want_hessians=False)
        for st in range(0, total, s.TOKEN_CHUNK):
            hidden[st:st + s.TOKEN_CHUNK] += (output[st:st + s.TOKEN_CHUNK].to(WORK).unsqueeze(-2) * inject[st:st + s.TOKEN_CHUNK].unsqueeze(-1)).flatten(-2)
        del layer, gate_up, down, output, mixed, inject
        torch.cuda.empty_cache()
        bad = sum(int((~torch.isfinite(hidden[st:st + s.TOKEN_CHUNK])).sum()) for st in range(0, total, s.TOKEN_CHUNK))
        log(f"L{index}: {time.time() - started:.1f}s")
        if bad:
            raise SystemExit(f"L{index}: {bad} non-finite hidden values")
    return hidden, offs, started_all


def final_mixer(student, config):
    mixer = model_code.Qwen4ExpTextGatedResidual(config, use_combine=False)
    for name, parameter in mixer.named_parameters():
        parameter.data = student.mixer_tensor(f"{s.PREFIX}hyper_connection_mixer.{name}").to(WORK)
    return mixer.to(DEVICE).requires_grad_(False)


def sequence_hidden_logits(hidden, mixer, head, positions=None, chunk_size=256):
    """Yield (positions, pre-mixer HC states, logits) for one forwarded sequence.

    Mix the WHOLE sequence exactly as the CLI does, then slice logit rows.
    Never materialize sequence_length*vocab_size logits at once.
    """
    final = mixer(hidden)
    if positions is None:
        positions = torch.arange(len(hidden), device=hidden.device)
    for pos in positions.to(device=hidden.device, dtype=torch.long).split(chunk_size):
        yield pos, hidden[pos], final[pos].float() @ head.T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--experts", default="src")
    p.add_argument("--dense", default="src")
    p.add_argument("--dense-mode", default="all", choices=["all", "bf16only", "w8only"])
    p.add_argument("--head", default="src")
    p.add_argument("--kv", default="none")
    p.add_argument("--embed", default="src", choices=["src", "int8"])
    p.add_argument("--alt-lib", default=None, help="another T3 library for --alt-layers")
    p.add_argument("--alt-layers", default=None)
    p.add_argument("--hc-w8", action="store_true")
    p.add_argument("--shared-w8", action="store_true")
    p.add_argument("--dense-bits", type=int, default=0, help="re-round matching dense projections to this many bits")
    p.add_argument("--dense-group", type=int, default=128)
    p.add_argument("--dense-match", default=r"(linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|self_attn\.(q_proj|k_proj|v_proj|o_proj)|index_qk_proj)\.weight$")
    p.add_argument("--domains", default="eval_public,eval_nemotron,eval_multilingual")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--precise", action="store_true", help="FP32 arithmetic throughout, for a low-noise teacher")
    p.add_argument("--save-teacher", default=None)
    a = p.parse_args()
    torch.set_grad_enabled(False)
    job = Job(f"sim/{a.name}")
    global WORK
    WORK = torch.float32 if a.precise else BF16
    config = AutoConfig.from_pretrained(FP8_SRC).text_config
    config._attn_implementation = "eager"
    source, live = Checkpoint(FP8_SRC), Checkpoint(LIVE)
    student = Student(a, source, live)
    tok = AutoTokenizer.from_pretrained(LIVE)

    seqs, spans_of = [], []
    domains = a.domains.split(",")
    meta = {}
    for domain in domains:
        with safe_open(s.SETS / f"{domain}.safetensors", framework="np") as f:
            ids, offsets = f.get_tensor("input_ids"), f.get_tensor("offsets")
        teacher = load_file(TEACHER / f"{domain}.safetensors")
        first_global = int(teacher["sequence"].min())
        keep = row_mask(domain, None, teacher, first_global, ids, offsets, tok)
        meta[domain] = dict(teacher=teacher, first=first_global, keep=keep, base=len(seqs))
        for i in range(len(offsets) - 1):
            seqs.append(torch.from_numpy(ids[offsets[i]:offsets[i + 1]].astype(np.int64)))
    lengths = [len(x) for x in seqs]
    offs = np.cumsum([0, *lengths])
    total = int(offs[-1])
    job.log(f"{len(seqs)} sequences, {total} tokens; scored {[int(meta[d]['keep'].sum()) for d in domains]}")
    hidden, offs, started_all = forward_hidden(student, config, seqs, a.embed, job.log)
    mixer = final_mixer(student, config)
    head = student.head().to(DEVICE)
    result = {"args": vars(a), "seconds": time.time() - started_all}
    for domain in domains:
        m = meta[domain]
        teacher = m["teacher"]
        tseq = teacher["sequence"].numpy()
        per_seq = {}
        saved = dict(ids=[], logprobs=[], tail=[], sequence=[], position=[])
        for g in np.unique(tseq).tolist():
            if a.save_teacher:
                every = np.flatnonzero(tseq == g)
                i = m["base"] + g - m["first"]
                final_all = mixer(hidden[int(offs[i]):int(offs[i + 1])])
                where = torch.from_numpy(teacher["position"].numpy()[every]).to(DEVICE).long()
                for chunk in where.split(4096):
                    la = F.log_softmax(final_all[chunk].float() @ head.T, dim=-1)
                    values, indices = la.topk(64, dim=-1)
                    saved["ids"].append(indices.int().cpu())
                    saved["logprobs"].append(values.float().cpu())
                    saved["tail"].append(((1 - values.double().exp().sum(-1)).clamp_min(1e-12)).float().cpu())
                saved["sequence"].append(teacher["sequence"][every])
                saved["position"].append(teacher["position"][every])
            rows = np.flatnonzero((tseq == g) & m["keep"])
            if not len(rows):
                per_seq[g] = dict(tokens=0, kl=0.0, top1=0, direct=0.0)
                continue
            i = m["base"] + g - m["first"]
            st, en = int(offs[i]), int(offs[i + 1])
            final = mixer(hidden[st:en])
            pos = torch.from_numpy(teacher["position"].numpy()[rows]).to(DEVICE).long()
            kls, agree, direct = [], [], []
            for chunk in torch.from_numpy(np.arange(len(rows))).split(4096):
                chunk_rows = rows[chunk.numpy()]
                logits = final[pos[chunk.to(DEVICE)]].float() @ head.T
                log_all = F.log_softmax(logits, dim=-1)
                s_logp, s_ids = log_all.topk(64, dim=-1)
                t_ids = teacher["ids"][chunk_rows].to(DEVICE).long()
                t_logp = teacher["logprobs"][chunk_rows].to(DEVICE).double()
                t_tail = teacher["tail"][chunk_rows].to(DEVICE).double()
                kl, ag = kl_batch(t_ids, t_logp, t_tail, s_ids, s_logp)
                kls.append(kl)
                agree.append(ag)
                # Direct KL on the teacher's top 64 plus one tail bin, with the student's real values: no imputation.
                q_top = log_all.gather(1, t_ids).double()
                p_top = t_logp.exp()
                q_tail = (1 - q_top.exp().sum(1)).clamp_min(1e-12)
                p_tail = t_tail.clamp_min(1e-12)
                direct.append((p_top * (t_logp - q_top)).sum(1) + p_tail * (p_tail.log() - q_tail.log()))
            kl = torch.cat(kls)
            per_seq[g] = dict(tokens=len(rows), kl=float(kl.sum()), top1=int(torch.cat(agree).sum()), direct=float(torch.cat(direct).sum()))
        if a.save_teacher:
            from safetensors.torch import save_file
            Path(a.save_teacher).mkdir(parents=True, exist_ok=True)
            save_file({k: torch.cat(v) for k, v in saved.items()}, Path(a.save_teacher) / f"{domain}.safetensors")
        tokens = sum(v["tokens"] for v in per_seq.values())
        mean = sum(v["kl"] for v in per_seq.values()) / max(tokens, 1)
        top1 = sum(v["top1"] for v in per_seq.values()) / max(tokens, 1)
        direct = sum(v["direct"] for v in per_seq.values()) / max(tokens, 1)
        result[domain] = dict(mean_kl=mean, direct_kl=direct, top1=top1, tokens=tokens, sequences=per_seq)
        job.log(f"{domain}: KL {mean:.4f} direct {direct:.4f} top1 {top1:.4f} tokens {tokens}")
    job.done(result)


if __name__ == "__main__":
    main()
