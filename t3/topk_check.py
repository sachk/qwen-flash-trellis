#!/usr/bin/env python3
"""Run AFTER t3 setup has installed topk_mask + generated reference.

  t3 python t3/topk_check.py --no-bench

No eager PyTorch sort surrogate: every output is compared bitwise to the actual
apply_top_k_top_p_triton, including NaN payloads and in-place aliasing. k=0 is
tested both raw (not a disabled sentinel in Triton) and normalized to V as the
request path does. Default timing: M=4 and 16, V=248320, k20/p=.95 and 1.
In-place timing MUST restore full logits before every mask; otherwise it times
an already-masked sparse row. Reports copy+mask, reset-only, and their difference.
The latter is an estimate, not an isolated kernel-event measurement.
"""
import argparse
import json
import math
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib", type=Path, default=Path(os.environ["KUDA_TOPK_MASK_LIB"]))
    parser.add_argument("--vocab", type=int, default=248320)
    parser.add_argument("--rows", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--no-bench", action="store_true")
    args = parser.parse_args()
    if not 32 <= args.vocab <= 262144 or any(not 1 <= m <= 64 for m in args.rows):
        parser.error("need 32<=vocab<=262144 and rows in 1..64")
    if args.iters < 200:
        parser.error("need >=200 timed graph replays")
    os.environ["KUDA_TOPK_MASK"] = "1"
    os.environ["KUDA_TOPK_MASK_LIB"] = str(args.lib.resolve())

    import torch
    from vllm.v1.sample.ops import topk_mask as candidate
    from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton as baseline
    if not candidate._AVAILABLE:
        raise RuntimeError("native library unavailable; refusing to benchmark fallback-only wiring")
    if torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError("SM80 checker; do not silently measure another architecture")
    torch.set_grad_enabled(False)
    torch.manual_seed(1520)
    candidate.prepare_topk_mask(torch.device("cuda", torch.cuda.current_device()))

    def settings(rows, vocab, kvals, pvals):
        k = None if kvals is None else torch.tensor(
            [kvals[i % len(kvals)] for i in range(rows)], dtype=torch.int32, device="cuda")
        p = None if pvals is None else torch.tensor(
            [pvals[i % len(pvals)] for i in range(rows)], dtype=torch.float32, device="cuda")
        return k, p

    def equal_bits(actual, expected, label):
        mismatch = actual.view(torch.int32) != expected.view(torch.int32)
        count = mismatch.sum().item()
        if count:
            indices = mismatch.nonzero()[:12]
            details = [(int(r), int(c), int(actual.view(torch.int32)[r, c]),
                        int(expected.view(torch.int32)[r, c])) for r, c in indices.cpu().tolist()]
            raise AssertionError(f"{label}: {count} bit mismatches; (row,col,actual_bits,reference_bits)={details}")

    def compare(src, k, p, label, stride=False):
        rows, vocab = src.shape
        if stride:
            a_base = torch.full((rows, vocab + 17), 12345., device="cuda")
            b_base = a_base.clone()
            actual, expected = a_base[:, :vocab], b_base[:, :vocab]
            actual.copy_(src)
            expected.copy_(src)
        else:
            actual, expected = src.clone(), src.clone()
        returned_ref = baseline(expected, k, p)
        returned = candidate.apply_topk_mask(actual, k, p)
        equal_bits(returned, returned_ref, label)
        if returned.data_ptr() != actual.data_ptr() or returned_ref.data_ptr() != expected.data_ptr():
            raise AssertionError(f"{label}: supported contiguous-vocab input lost alias contract")
        if stride and not torch.equal(a_base[:, vocab:], b_base[:, vocab:]):
            raise AssertionError(f"{label}: row padding overwritten")
        # Every call starts from the original input: repeat without accidentally
        # accepting idempotence on already-truncated logits as determinism.
        actual.copy_(src)
        candidate.apply_topk_mask(actual, k, p)
        equal_bits(actual, returned_ref, label + " repeat")
        print(json.dumps(dict(case=label, rows=rows, vocab=vocab, bit_mismatches=0)), flush=True)
        return returned_ref

    def distributions(rows, vocab):
        random = torch.randn(rows, vocab, device="cuda")
        yield "gaussian", random
        spikes = random.clone()
        spikes[:, ::max(1, vocab // 23)] += 8
        yield "spikes", spikes
        ties = torch.randint(-4, 5, (rows, vocab), dtype=torch.int32, device="cuda").float()
        yield "heavy-ties", ties
        threshold = torch.full((rows, vocab), -10., device="cuda")
        threshold[:, :19] = torch.arange(19, device="cuda") + 2
        # Boundary duplicates span blocks and slices, exposing prefix-order bugs.
        threshold[:, 31::257] = 1
        yield "kth-duplicates", threshold
        masked = random.clone()
        masked[:, ::3] = -float("inf")
        masked[:, :min(vocab, 8192)] = -float("inf")
        yield "grammar-first-tile-empty", masked
        for finite in (0, 1, 20):
            sparse = torch.full_like(random, -float("inf"))
            if finite:
                sparse[:, -finite:] = random[:, -finite:]
            yield f"finite-{finite}", sparse
        yield "uniform", torch.ones_like(random)
        zero = torch.zeros_like(random)
        zero[:, ::2] = -0.0
        yield "signed-zero", zero
        nan = random.clone()
        nan.view(torch.int32)[:, ::997] = 0x7fc12345
        yield "nan-payload", nan
        inf = random.clone()
        inf[:, ::4001] = float("inf")
        yield "positive-infinity", inf
        # Wide dynamic range exercises capped searches and log/exp cancellation.
        yield "wide-range", random * 1000

    for rows in sorted(set(args.rows + [1, 64])):
        v = args.vocab
        schedules = [([20], [.95]), ([1], [1.]), ([20], [.5]),
                     ([1, 20, v, v + 1, 0], [.5, .95, 1.]),
                     ([v], [1.]),  # request k=0 normalized to V
                     ([0], [1.]),  # raw invalid sentinel preserved, not normalized
                     (None, [.5, .95, 1.]), ([20], None), (None, None)]
        for name, source in distributions(rows, v):
            for index, (ks, ps) in enumerate(schedules):
                k, p = settings(rows, v, ks, ps)
                compare(source, k, p, f"{name}/schedule{index}", stride=index == 0)

    # Real probability boundaries (not arbitrary decimal choices): compare p
    # immediately below/at/above mass computed from large, distinct logits.
    for vocab in sorted(set([32, 8191, 8192, 8193, args.vocab])):
        source = torch.randn(4, vocab, device="cuda")
        source[:, :20] = torch.arange(20, device="cuda") * .125 + 12
        sorted_probs = source[:, :20].softmax(-1).sort(descending=True).values
        p0 = sorted_probs[:, :7].sum(-1)
        k, _ = settings(4, vocab, [20], None)
        for direction in (-1, 0, 1):
            p = p0 if direction == 0 else torch.nextafter(p0, torch.full_like(p0, float(direction) * float("inf")))
            compare(source, k, p, f"mass-boundary/{direction}")

    # Independent streams AND independently captured graphs: scratch ownership
    # must not cross-contaminate mixed requests. The reference is run serially
    # since its global scratch cache is not a concurrency oracle.
    rows, vocab = 4, args.vocab
    sources = [torch.randn(rows, vocab, device="cuda"), torch.randn(rows, vocab, device="cuda") * 2]
    k, p = settings(rows, vocab, [1, 20, vocab, 0], [.95, 1., .5])
    expected = [baseline(src.clone(), k, p) for src in sources]
    targets = [src.clone() for src in sources]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs = []
    for stream, src, dst in zip(streams, sources, targets):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            candidate.apply_topk_mask(dst, k, p)
            dst.copy_(src)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            dst.copy_(src)
            candidate.apply_topk_mask(dst, k, p)
        graphs.append(graph)
    for _ in range(8):
        for stream, graph in zip(streams, graphs):
            with torch.cuda.stream(stream):
                graph.replay()
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    for i, (actual, ref) in enumerate(zip(targets, expected)):
        equal_bits(actual, ref, f"concurrent-graph-{i}")
    print("PASS: independent graph/stream scratch", flush=True)
    del graphs

    # Unsupported layout uses the same contiguous-copy fallback as baseline.
    storage = torch.randn(4, args.vocab * 2, device="cuda")
    k, p = settings(4, args.vocab, [20], [.95])
    source = storage[:, ::2]
    ref = baseline(source.clone()[:, :], k, p)
    out = candidate.apply_topk_mask(source, k, p)
    equal_bits(out, ref, "vocab-stride-fallback")
    if out.data_ptr() == source.data_ptr():
        raise AssertionError("unsupported vocab-stride path must return a contiguous copy")

    def capture(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        return graph

    def time_us(graph, calls):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            graph.replay()
        stop.record()
        stop.synchronize()
        return start.elapsed_time(stop) * 1000 / (args.iters * calls)

    if not args.no_bench:
        for rows in args.rows:
            copies = max(8, math.ceil(64 * 1024**2 / (rows * args.vocab * 4)))
            sources = [torch.randn(rows, args.vocab, device="cuda") for _ in range(copies)]
            for source in sources:
                source[:, ::9973] += 8
            targets = [torch.empty_like(source) for source in sources]
            for probability in (.95, 1.):
                k, p = settings(rows, args.vocab, [20], [probability])

                def run(fn=None):
                    for src, dst in zip(sources, targets):
                        dst.copy_(src)
                        if fn is not None:
                            fn(dst, k, p)

                reset = capture(lambda: run())
                original = capture(lambda: run(baseline))
                modified = capture(lambda: run(candidate.apply_topk_mask))
                times = {name: [] for name in ("reset", "reference", "candidate")}
                for trial in range(3):
                    order = [("reset", reset), ("reference", original), ("candidate", modified)]
                    for name, graph in (order if trial % 2 == 0 else order[::-1]):
                        times[name].append(time_us(graph, copies))
                med = {name: sorted(values)[1] for name, values in times.items()}
                record = dict(rows=rows, vocab=args.vocab, k=20, p=probability,
                              copies=copies, graph_replays=args.iters,
                              reset_us=med["reset"], reference_copy_mask_us=med["reference"],
                              candidate_copy_mask_us=med["candidate"],
                              reference_estimated_mask_us=med["reference"] - med["reset"],
                              candidate_estimated_mask_us=med["candidate"] - med["reset"])
                # Verify the outputs produced by the timed graph, not only
                # correctness cases before benchmarking.
                modified.replay()
                torch.cuda.synchronize()
                for i, (src, dst) in enumerate(zip(sources, targets)):
                    equal_bits(dst, baseline(src.clone(), k, p), f"timed-graph-{i}")
                print(json.dumps(record), flush=True)
                del reset, original, modified
    torch.cuda.synchronize()
    print("PASS: bitwise reference masks, alias/layout contract, graph replay and stream isolation", flush=True)


if __name__ == "__main__":
    main()
