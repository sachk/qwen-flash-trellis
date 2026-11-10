"""Shared by GPU jobs: job contract (§9.3), checkpoint readers, the auto_gptq format."""
import json
import os
import signal
import struct
import time
from pathlib import Path

import numpy as np
import torch

from paths import ROOT, FP8_SRC, LIVE


class Job:
    """Heartbeat log, SIGTERM flag, progress file and DONE marker for one step."""

    def __init__(self, step_id):
        self.dir = ROOT / step_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.dir / "log.txt", "a", buffering=1)
        self.stop = False
        self.started = time.time()
        signal.signal(signal.SIGTERM, self._on_term)

    def _on_term(self, *_):
        self.stop = True
        self.log("SIGTERM: stopping at the next safe point")

    def log(self, message):
        line = f"{time.strftime('%H:%M:%S')} {message}"
        print(line, flush=True)
        self.log_file.write(line + "\n")

    def progress(self):
        path = self.dir / "progress.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def save_progress(self, value):
        tmp = self.dir / "progress.json.tmp"
        tmp.write_text(json.dumps(value))
        tmp.replace(self.dir / "progress.json")

    def done(self, result):
        (self.dir / "result.json").write_text(json.dumps(result, indent=1))
        (self.dir / "DONE").touch()
        self.log(f"DONE in {(time.time() - self.started) / 60:.1f} min")


def fp8_table():
    """float32 value of every E4M3 byte: 4 exponent bits, 3 mantissa bits, bias 7."""
    table = np.zeros(256, dtype=np.float32)
    for byte in range(256):
        sign = -1.0 if byte & 0x80 else 1.0
        exponent = (byte >> 3) & 0xF
        mantissa = byte & 0x7
        if exponent == 0:
            value = mantissa / 8 * 2.0**-6
        elif exponent == 15 and mantissa == 7:
            value = float("nan")
        else:
            value = (1 + mantissa / 8) * 2.0 ** (exponent - 7)
        table[byte] = sign * value
    return torch.from_numpy(table)


DTYPES = {
    "BF16": (np.uint16, torch.bfloat16), "F16": (np.uint16, torch.float16), "F32": (np.float32, torch.float32),
    "I32": (np.int32, torch.int32), "I64": (np.int64, torch.int64), "U8": (np.uint8, torch.uint8), "F8_E4M3": (np.uint8, None),
}


class Checkpoint:
    """Read single tensors from a sharded safetensors checkpoint without loading shards."""

    def __init__(self, root):
        self.root = Path(root)
        self.weight_map = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]
        self.headers = {}
        self.table = fp8_table()

    def header(self, shard):
        if shard not in self.headers:
            with open(self.root / shard, "rb") as f:
                size = struct.unpack("<Q", f.read(8))[0]
                self.headers[shard] = (json.loads(f.read(size)), 8 + size)
        return self.headers[shard]

    def __contains__(self, name):
        return name in self.weight_map

    def location(self, name):
        shard = self.weight_map[name]
        header, base = self.header(shard)
        entry = header[name]
        return self.root / shard, base + entry["data_offsets"][0], entry["data_offsets"][1] - entry["data_offsets"][0], entry

    def read(self, name, device="cpu"):
        path, offset, length, entry = self.location(name)
        fd = os.open(path, os.O_RDONLY)
        try:
            raw = os.pread(fd, length, offset)
        finally:
            os.close(fd)
        np_type, torch_type = DTYPES[entry["dtype"]]
        array = np.frombuffer(raw, dtype=np_type).reshape(entry["shape"]).copy()
        if torch_type is None:
            return self.table.to(device)[torch.from_numpy(array).to(device).long()]
        if torch_type in (torch.bfloat16, torch.float16):
            return torch.from_numpy(array.astype(np.int16)).to(device).view(torch_type)
        return torch.from_numpy(array).to(device)

    def weight(self, name, device="cpu"):
        """float32 weight. FP8 source tensors scale by 128x128 blocks of weight_scale_inv."""
        value = self.read(name, device)
        scale_name = name + "_scale_inv"
        if scale_name in self:
            scale = self.read(scale_name, device).float()
            block_r = value.shape[0] // scale.shape[0]
            block_c = value.shape[1] // scale.shape[1]
            value = value * scale.repeat_interleave(block_r, 0).repeat_interleave(block_c, 1)
        return value.float()


def word_layout(bits, count):
    """auto_gptq packing: fields fill int32 words low to high and may straddle words."""
    positions = np.arange(count, dtype=np.int64) * bits
    return positions // 32, positions % 32


def pack(q, bits):
    """q: uint array [K, R] in 0..2^bits-1. Returns int32 words [K*bits/32, R]."""
    k, r = q.shape
    words, shifts = word_layout(bits, k)
    out = np.zeros((k * bits // 32, r), dtype=np.uint64)
    low = q.astype(np.uint64) << shifts[:, None].astype(np.uint64)
    np.add.at(out, words, low & 0xFFFFFFFF)
    spill = (shifts + bits) > 32
    if spill.any():
        np.add.at(out, words[spill] + 1, low[spill] >> np.uint64(32))
    return out.astype(np.uint32).astype(np.int32)


def unpack(words, bits, k):
    """Inverse of pack: int32 words [W, R] to int32 q [K, R]."""
    data = words.astype(np.uint32).astype(np.uint64)
    positions, shifts = word_layout(bits, k)
    value = data[positions] >> shifts[:, None].astype(np.uint64)
    spill = (shifts + bits) > 32
    if spill.any():
        value[spill] |= data[positions[spill] + 1] << (32 - shifts[spill])[:, None].astype(np.uint64)
    return (value & ((1 << bits) - 1)).astype(np.int32)
