# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A/B the output correction plus NCCL input staging, with CUDA graphs."""

import json
from pathlib import Path

import torch

from vllm.v1.attention.ops.dcp import CPTritonContext, correct_attn_out


def measure(fn):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(64):
            fn()
    samples = []
    for _ in range(7):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(20):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / (20 * 64))
    return sorted(samples)[len(samples) // 2]


def main():
    torch.manual_seed(1)
    rows = []
    for b, h, d, n in [
        (1, 32, 512, 4),
        (4, 32, 512, 4),
        (16, 32, 512, 4),
        (64, 32, 512, 4),
        (128, 32, 512, 4),
        (4, 64, 512, 8),
        (64, 64, 512, 8),
        (128, 64, 512, 8),
    ]:
        source = torch.randn(b, h, d, device="cuda", dtype=torch.bfloat16)
        work = source.clone()
        lses = torch.randn(n, b, h, device="cuda")
        contexts = [CPTritonContext(), CPTritonContext()]

        # Baseline modifies its input; restore via a common copy in both paths
        # so graph replays use finite, identical data.
        def baseline(work=work, source=source, lses=lses, contexts=contexts):
            work.copy_(source)
            out, _ = correct_attn_out(work, lses, 0, contexts[0])
            return out.movedim(0, 1).contiguous()

        def fused(work=work, source=source, lses=lses, contexts=contexts):
            work.copy_(source)
            out, _ = correct_attn_out(
                work, lses, 0, contexts[1], output_head_major=True
            )
            staged = out.movedim(0, 1).contiguous()
            assert staged.data_ptr() == out.data_ptr()
            return staged

        expected = baseline().clone()
        actual = fused()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        a, btime = measure(baseline), measure(fused)
        # Reverse order to expose ordering/clock effects.
        b2, a2 = measure(fused), measure(baseline)
        row = dict(
            tokens=b,
            heads=h,
            head_dim=d,
            dcp=n,
            baseline_us=(a + a2) / 2,
            fused_us=(btime + b2) / 2,
        )
        row["speedup"] = row["baseline_us"] / row["fused_us"]
        rows.append(row)
        print(json.dumps(row), flush=True)
    Path("layout_benchmark.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
