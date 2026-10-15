"""Benchmark reduce_rows_128 implementations (AS, SSA) vs torch.sum."""

from benchmark import (
    benchmark,
    BenchmarkConfig,
    NpuEventMeasurer,
    Reference,
    print_results_table,
    print_result,
)

import torch
import torch_npu

from sgl_glm53_ascend_ops.ops import reduce_rows_128

device = "npu"

D = 128

config = BenchmarkConfig(
    warmup=10,
    rounds=400,
    batch_size=20,
)
measurer = NpuEventMeasurer()


def bench_reduce_rows_128(num_rows: int):
    src = torch.randn(num_rows, D, device=device, dtype=torch.float32)

    out_as = None
    out_ssa = None
    out_torch = None

    def call_as():
        nonlocal out_as
        out_as = reduce_rows_128(src, impl=0)

    def call_ssa():
        nonlocal out_ssa
        out_ssa = reduce_rows_128(src, impl=1)

    def call_torch():
        nonlocal out_torch
        out_torch = src.sum(dim=-1)

    return benchmark(
        call_as,
        references=[
            # Reference("reduce_rows_128(SSA)", call_ssa, lambda: out_ssa.clone()),
            Reference("torch.sum", call_torch, lambda: out_torch.clone()),
        ],
        measurer=measurer,
        config=config,
        output_fn=lambda: out_as.clone(),
    )


def main():
    results = []
    for num_rows in [16, 64, 256, 1024, 4096]:
        results.append((f"rows={num_rows}", bench_reduce_rows_128(num_rows)))
    print()
    print_results_table(results)


if __name__ == "__main__":
    main()
