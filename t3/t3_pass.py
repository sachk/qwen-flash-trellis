#!/usr/bin/env python3
"""The calibration pass for the trellis format: one teacher forward over a wide token set, with rounding per layer.

Writes, per layer, every format the allocator may choose, and per-expert sensitivities:
  lib/L{n}/{t3_13,t2_13,t3_2,t4_2}.bin   packets as raw uint32, experts in the order in meta.json
  lib/L{n}/meta.json                     per-expert routed tokens, output energy, relative error per format
The forward stays on teacher weights, as in s2_pass. Teacher top-64 for all eval sets at the end.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen4_exp import modeling_qwen4_exp as model_code
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import s2_pass as s
from requant_common import FP8_SRC, LIVE, Checkpoint, Job
import t3lib as t
import t3pack as p

DEVICE = "cuda"
# Scale law: max(alpha * sigma, beta * amax / top level). Levels are L = (v*v + B) * v, so alpha carries 1/B.
SCALE = {2: (0.22 / 250, 0.55), 3: (0.135 / 1000, 0.8), 4: (0.07 / 4000, 0.9)}
NMIN, DAMP = 16.0, 0.08
CHUNK = 64


class Sequences(s.Sequences):
    """s2_pass.Sequences with the extra calibration and eval sets."""

    def __init__(self, smoke):
        self.ids, self.names, self.calib, self.spans = [], [], [], []
        for name in ("calib", "calib2", "eval_medical", "eval_legal"):   # the first attempt's teacher covers the other eval sets
            with safe_open(s.SETS / f"{name}.safetensors", framework="np") as f:
                ids, offsets, meta = f.get_tensor("input_ids"), f.get_tensor("offsets"), f.metadata()
            spans = json.loads(meta["score_spans"]) if "score_spans" in meta else None
            limit = 12_000 if smoke else None
            used = 0
            is_calib = name.startswith("calib")
            for i in range(len(offsets) - 1):
                seq = ids[offsets[i]:offsets[i + 1]]
                if limit is not None and used + len(seq) > limit:
                    break
                used += len(seq)
                self.ids.append(torch.from_numpy(seq.astype(np.int64)))
                self.names.append(name)
                self.calib.append(is_calib)
                self.spans.append(spans[i] if spans else [[1, len(seq)]])
        self.lengths = [len(x) for x in self.ids]
        self.offsets = np.cumsum([0, *self.lengths])
        self.total = int(self.offsets[-1])
        self.calib_mask = torch.cat([torch.full((n,), c, dtype=torch.bool) for n, c in zip(self.lengths, self.calib)]).to(DEVICE)
        self.eval_mask = torch.cat([torch.full((n,), name == "eval_public", dtype=torch.bool) for n, name in zip(self.lengths, self.names)]).to(DEVICE)


def round_set(job, weights, H_raw, gate2, routed, pool, members, matrix, bits, out_path):
    """Rounds `members` and appends their packets. Returns relative objective per member."""
    alpha, beta = SCALE[bits]
    errors = np.zeros(len(members))
    with open(out_path, "wb") as f:
        for b in range(0, len(members), CHUNK):
            part = torch.tensor(members[b:b + CHUNK], device=DEVICE)
            w = weights[part].float()
            H = t.condition(H_raw[part].clone(), gate2[part.cpu()], routed[part.cpu()], pool, NMIN, DAMP)
            hat, scales, q = t.babai_round(w, H, f"T{bits}", alpha, "c2", beta=beta)
            errors[b:b + CHUNK] = t.objective(w, hat, H).cpu().numpy()
            f.write((p.pack(q, scales, matrix, bits).cpu().numpy().astype(np.uint32)).tobytes())
            del w, H, hat, scales, q
    return errors.tolist()


def layer_library(job, out, index, gate_up, down, h13, h2, stats):
    d = out / f"lib/L{index}"
    d.mkdir(parents=True, exist_ok=True)
    gate2, routed = stats["gate2"].float(), stats["routed"].float()
    used = (routed > 0).to(DEVICE)
    norm = gate2.to(DEVICE).clamp_min(1e-12).view(-1, 1, 1)
    pool13 = (h13[used] / norm[used]).mean(0)
    pool2 = (h2[used] / norm[used]).mean(0)
    energy = ((down.float() @ h2) * down.float()).sum((1, 2)).cpu().numpy()   # sum g^2 ||y||^2 on calibration tokens
    # Early layers tolerate 2-bit everywhere; late layers are where 4-bit pays. See layer_weights.py.
    order = np.argsort(gate2.numpy())                                         # coldest first
    every = list(range(512))
    t2_13 = every if index < 36 else sorted(order[:int(512 * 0.8)].tolist())
    t2_2 = every if index < 20 else (sorted(order[:256].tolist()) if index < 36 else [])
    t4_2 = sorted(order[-int(512 * (0.25 if index < 16 else 0.6)):].tolist())
    meta = dict(routed=routed.tolist(), gate2=gate2.tolist(), energy=energy.tolist(), t2_13_set=t2_13, t2_2_set=t2_2, t4_2_set=t4_2)
    started = time.time()
    for name, weights, H, pool, members, matrix, bits in (
        ("t3_13", gate_up, h13, pool13, every, "w13", 3), ("t3_2", down, h2, pool2, every, "w2", 3),
        ("t2_13", gate_up, h13, pool13, t2_13, "w13", 2), ("t2_2", down, h2, pool2, t2_2, "w2", 2), ("t4_2", down, h2, pool2, t4_2, "w2", 4)):
        if members:
            meta[name] = round_set(job, weights, H, gate2, routed, pool, members, matrix, bits, d / f"{name}.bin")
    job.log(f"L{index} rounding {time.time() - started:.0f}s; " + " ".join(f"{k} {np.mean(meta[k]):.4f}" for k in ("t3_13", "t3_2", "t2_13", "t2_2", "t4_2") if k in meta))
    tmp = d / "meta.json.tmp"
    tmp.write_text(json.dumps(meta))
    tmp.replace(d / "meta.json")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--resume", action="store_true")
    a.add_argument("--smoke", action="store_true")
    a.add_argument("--name", default="t3pass")
    a.add_argument("--layers", type=int, default=48)
    args = a.parse_args()
    job = Job(args.name + ("-smoke" if args.smoke else ""))
    out = job.dir
    torch.set_grad_enabled(False)
    config = AutoConfig.from_pretrained(FP8_SRC).text_config
    config._attn_implementation = "eager"
    source, live = Checkpoint(FP8_SRC), Checkpoint(LIVE)
    sq = Sequences(args.smoke)
    job.log(f"{len(sq.ids)} sequences, {sq.total} tokens, {int(sq.calib_mask.sum())} calibration")
    ids = [x.to(DEVICE) for x in sq.ids]
    rotary = model_code.Qwen4ExpTextRotaryEmbedding(config=config).to(DEVICE)
    longest = max(sq.lengths)
    positions = torch.arange(longest, device=DEVICE).view(1, 1, -1).expand(3, 1, -1)
    cos, sin = rotary(torch.zeros(1, longest, 2560, dtype=s.BF16, device=DEVICE), positions)
    progress = job.progress()
    first = progress.get("next_layer", 0)
    if first:
        hidden = s.load_hidden(out / "hidden.bin", sq.total)
        job.log(f"resumed before layer {first}")
    else:
        embed = s.tensor_for(s.PREFIX + "embed_tokens.weight", source, live, DEVICE).to(s.BF16)
        hidden = torch.empty(sq.total, 10240, dtype=s.BF16, device=DEVICE)
        for i, q in enumerate(ids):
            hidden[sq.offsets[i]:sq.offsets[i + 1]] = embed[q].repeat(1, 4)
        del embed
    reports = progress.get("reports", {})
    try:
        for index in range(first, args.layers):
            started = time.time()
            layer = s.build_layer(config, index, source, live)
            is_qsa = layer.layer_type != "linear_attention"
            kv = {"k": 0.0, "v": 0.0}
            for i, q in enumerate(ids):
                if job.stop:
                    raise s.Stop
                st, en = int(sq.offsets[i]), int(sq.offsets[i + 1])
                h = hidden[st:en].unsqueeze(0)
                if layer.ple is not None:
                    h = h + layer.ple(h, q.unsqueeze(0), None, conv_mask=None)
                mixed, hyper, inject = layer.attn_hyper_connection(h)
                if is_qsa:
                    att = s.qsa_forward(layer.self_attn, mixed[0], cos, sin, kv if sq.calib[i] else None).unsqueeze(0)
                else:
                    att = layer.linear_attn(mixed, cache_params=None, attention_mask=None)
                hidden[st:en] = (hyper + (att.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2))[0]
            attention_s = time.time() - started
            mixed = torch.empty(sq.total, 2560, dtype=s.BF16, device=DEVICE)
            inject = torch.empty(sq.total, 4, dtype=s.BF16, device=DEVICE)
            for st in range(0, sq.total, s.TOKEN_CHUNK):
                m, _, i = layer.mlp_hyper_connection(hidden[st:st + s.TOKEN_CHUNK])
                mixed[st:st + s.TOKEN_CHUNK], inject[st:st + s.TOKEN_CHUNK] = m, i
            gate_up, down = s.load_experts(index, source)
            moe_started = time.time()
            output, stats, h13, h2 = s.run_moe(layer.mlp, gate_up, down, mixed, sq.calib_mask)
            moe_s = time.time() - moe_started
            for st in range(0, sq.total, s.TOKEN_CHUNK):
                hidden[st:st + s.TOKEN_CHUNK] += (output[st:st + s.TOKEN_CHUNK].to(s.BF16).unsqueeze(-2) * inject[st:st + s.TOKEN_CHUNK].unsqueeze(-1)).flatten(-2)
            del output, inject, mixed
            torch.cuda.empty_cache()
            rstart = time.time()
            layer_library(job, out, index, gate_up, down, h13, h2, stats)
            reports[str(index)] = dict(kv=kv, attention_s=round(attention_s, 1), moe_s=round(moe_s, 1), round_s=round(time.time() - rstart, 1), total_s=round(time.time() - started, 1))
            job.log(f"L{index} attn {attention_s:.0f}s moe {moe_s:.0f}s round {time.time() - rstart:.0f}s total {time.time() - started:.0f}s; gpu {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
            del layer, gate_up, down, stats, h13, h2
            torch.cuda.empty_cache()
            if (index + 1) % 8 == 0 and index + 1 < args.layers:
                s.save_hidden(hidden, out / "hidden.bin")
                job.save_progress({"next_layer": index + 1, "reports": reports})
                job.log(f"checkpoint before layer {index + 1}")
            if job.stop:
                raise s.Stop
        if args.layers == 48:
            s.finish(job, out, config, source, live, hidden, sq)
        job.save_progress({"next_layer": args.layers, "reports": reports})
        job.done({"tokens": sq.total, "reports": reports})
    except s.Stop:
        job.log("stopped; resume restarts from the last hidden-state checkpoint")


if __name__ == "__main__":
    main()
