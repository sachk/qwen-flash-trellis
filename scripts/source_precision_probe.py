#!/usr/bin/env python3
"""Compare BF16 and FP8 source noise without downloading full checkpoint shards.

Fetch only safetensors headers and selected tensor ranges. Run on ordinary Torch
(including RTX 5070); no serving kernels or full model are loaded. T4 uses this
repo's real C2/Babai solver with an identity Hessian, NOT fresh calibration.
These weight-space diagnostics are not task accuracy or end-to-end KL.
"""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from urllib.parse import quote

import httpx
import torch


class Source:
    def __init__(self, client, repo, revision, cache):
        self.client, self.repo, self.revision = client, repo, revision
        self.cache = cache / repo.replace("/", "--") / revision
        self.cache.mkdir(parents=True, exist_ok=True)
        self.bytes_downloaded = 0
        self.headers = {}
        self.files = set()
        index = self.cache / "model.safetensors.index.json"
        if not index.exists():
            response = client.get(self.url(index.name))
            response.raise_for_status()
            index.write_bytes(response.content)
            self.bytes_downloaded += len(response.content)
        self.weight_map = json.loads(index.read_text())["weight_map"]

    def url(self, name):
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{quote(name)}"

    def read_range(self, shard, start, end):
        length = end - start + 1
        path = self.cache / f"{shard}.{start}-{end}.bin"
        self.files.add(shard)
        if path.exists():
            data = path.read_bytes()
        else:
            url = self.url(shard) + f"?download=true&source_precision_range={start}-{end}"
            with self.client.stream("GET", url, headers={"Range": f"bytes={start}-{end}",
                                                         "Accept-Encoding": "identity"}) as response:
                response.raise_for_status()
                expected = f"bytes {start}-{end}/"
                if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                    raise RuntimeError(f"server did not honor bounded range {expected}: {response.status_code}")
                data = response.read()
            if len(data) != length:
                raise RuntimeError(f"short tensor range: {len(data)} != {length}")
            partial = path.with_suffix(".partial")
            partial.write_bytes(data)
            partial.replace(path)
            self.bytes_downloaded += len(data)
        if len(data) != length:
            raise RuntimeError(f"invalid cached range {path}")
        return data

    def entry(self, name):
        shard = self.weight_map[name]
        if shard not in self.headers:
            size = struct.unpack("<Q", self.read_range(shard, 0, 7))[0]
            if size > 64 * 1024 * 1024:
                raise RuntimeError("invalid safetensors header size")
            header = json.loads(self.read_range(shard, 8, 7 + size))
            self.headers[shard] = (header, size + 8)
        header, base = self.headers[shard]
        return shard, base, header[name]

    def tensor(self, name, expert=None):
        shard, base, entry = self.entry(name)
        first, last = entry["data_offsets"]
        shape = entry["shape"]
        if expert is not None:
            if len(shape) != 3 or not 0 <= expert < shape[0]:
                raise ValueError(f"invalid fused expert slice for {name}: {expert}")
            stride = (last - first) // shape[0]
            first += expert * stride
            last = first + stride
            shape = shape[1:]
        raw = self.read_range(shard, base + first, base + last - 1)
        dtypes = {"BF16": torch.bfloat16, "F32": torch.float32,
                  "F16": torch.float16, "F8_E4M3": torch.float8_e4m3fn,
                  "F8_E4M3FN": torch.float8_e4m3fn}
        value = torch.frombuffer(bytearray(raw), dtype=dtypes[entry["dtype"]]).reshape(shape)
        return value, {"shard": shard, "dtype": entry["dtype"], "shape": shape,
                       "tensor": name, "expert": expert, "raw_sha256": hashlib.sha256(raw).hexdigest()}

    def weight(self, name, device, expert=None):
        value, info = self.tensor(name, expert)
        value = value.to(device).float()
        scale_name = name + "_scale_inv"
        if scale_name in self.weight_map:
            scales, _ = self.tensor(scale_name)
            scales = scales.to(device).float()
            value *= scales.repeat_interleave(128, 0).repeat_interleave(128, 1)[:value.shape[0], :value.shape[1]]
        return value.to(torch.bfloat16).float(), info


def errors(reference, candidate):
    delta = candidate - reference
    relative_mse = (delta.square().sum() / reference.square().sum().clamp_min(1e-30)).item()
    return {"relative_mse": relative_mse, "relative_l2": relative_mse ** 0.5,
            "max_abs": delta.abs().max().item(), "identical": torch.equal(reference, candidate)}


def int8_round(value):
    grouped = value.reshape(value.shape[0], -1, 128)
    scale = (grouped.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-12).to(torch.bfloat16).float()
    return ((grouped / scale).round().clamp(-127, 127) * scale).reshape_as(value).to(torch.bfloat16).float()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bf16-revision", default="812d96e10a0dd82b69d749e68693698cf6c36daa")
    p.add_argument("--fp8-revision", default="236dfdf285828023ca3bcd3f37366c58a3469b13")
    p.add_argument("--layer", type=int, default=44)
    p.add_argument("--expert", type=int, default=5)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    dense_prefix = "model.language_model.layers.0."
    targets = [
        ("BF16 control", dense_prefix + "attn_hyper_connection.input_mix_weight_down.weight", None, None),
        ("W8 dense source proxy", dense_prefix + "linear_attn.out_proj.weight", None, None),
        ("T4 routed expert", f"model.language_model.layers.{args.layer}.mlp.experts.{args.expert}.down_proj.weight",
         f"model.language_model.layers.{args.layer}.mlp.experts.down_proj", args.expert),
    ]
    report = {"device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
              "torch": torch.__version__, "bf16_revision": args.bf16_revision,
              "fp8_revision": args.fp8_revision, "results": [],
              "limits": ["Weight-space pilot; no task accuracy or full-model KL measured.",
                         "T4 uses an identity Hessian shared by both source branches, not recalibrated activations.",
                         "W8 proxy is symmetric RTN, not the inherited AutoRound dense checkpoint.",
                         "Layer/expert sample is not representative of every layer or domain.",
                         "Pinned BF16 and FP8 releases may differ beyond storage precision; inspect BF16 controls."]}
    with httpx.Client(follow_redirects=True, timeout=120) as client, torch.inference_mode():
        bf16 = Source(client, "Qwen/Qwen3.8-Flash-Next", args.bf16_revision, args.out / "ranges")
        fp8 = Source(client, "Qwen/Qwen3.8-Flash-Next-FP8", args.fp8_revision, args.out / "ranges")
        for kind, name, bf16_name, expert in targets:
            reference, original = bf16.weight(bf16_name or name, device, expert)
            rounded, source = fp8.weight(name, device)
            if reference.shape != rounded.shape:
                raise RuntimeError(f"source shape mismatch for {name}")
            row = {"kind": kind, "tensor": name, "bf16": original, "fp8": source,
                   "fp8_source_error": errors(reference, rounded)}
            if kind.startswith("W8"):
                row["int8_from_bf16"] = errors(reference, int8_round(reference))
                row["int8_from_fp8"] = errors(reference, int8_round(rounded))
            elif kind.startswith("T4"):
                sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "t3"))
                import t3lib
                h = torch.eye(reference.shape[-1], device=device).unsqueeze(0)
                for label, value in (("t4_from_bf16", reference), ("t4_from_fp8", rounded)):
                    hat, _, _ = t3lib.babai_round(value.unsqueeze(0), h, "T4", 0.07 / 4000,
                                                 "c2", beta=0.9)
                    row[label] = errors(reference, hat[0])
            report["results"].append(row)
            (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
            del reference, rounded
        report["downloads"] = {"bf16_bytes": bf16.bytes_downloaded, "fp8_bytes": fp8.bytes_downloaded,
                               "bf16_shards_touched": sorted(bf16.files), "fp8_shards_touched": sorted(fp8.files)}
    if device.type == "cuda":
        report["peak_gpu_bytes"] = torch.cuda.max_memory_allocated(device)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(args.out / "report.json"), "downloads": report["downloads"],
                      "peak_gpu_bytes": report.get("peak_gpu_bytes")}), flush=True)


if __name__ == "__main__":
    main()
