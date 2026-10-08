from benchmark import (
    BenchmarkConfig,
    benchmark,
    NpuEventMeasurer,
    Reference,
    print_results_table,
)

import original, opt

import triton
import torch
import torch_npu

device = "npu"

T = 1
TOTAL = 16384
HC_MULT = 4
HIDDEN = TOTAL // HC_MULT  # 4096
M = HC_MULT
STRIDE = 2 * M + M * M  # 24
RMS_EPS = 1e-5
HC_EPS = 1e-6
SINKHORN_ITERS = 3
POST_MULT_VALUE = 2.0
OUT_NORM_EPS = 1e-5

config = BenchmarkConfig(
    warmup=5,
    rounds=500,
    batch_size=10,
    interleave=True,
    alternate_order=True,
)
measurer = NpuEventMeasurer()


# ── shared input factories ──────────────────────────────────────────────

def make_hc_pre_inputs(with_norm=False):
    x = torch.randn(T, TOTAL, device=device, dtype=torch.bfloat16)
    hc_fn = torch.randn(STRIDE, TOTAL, device=device, dtype=torch.float32)
    hc_scale = torch.randn(3, device=device, dtype=torch.float32)
    hc_base = torch.randn(STRIDE, device=device, dtype=torch.float32)
    out_norm_weight = None
    out_norm_eps = None
    if with_norm:
        out_norm_weight = torch.randn(HIDDEN, device=device, dtype=torch.bfloat16)
        out_norm_eps = OUT_NORM_EPS
    return x, hc_fn, hc_scale, hc_base, out_norm_weight, out_norm_eps


def make_hc_post_inputs():
    x = torch.randn(T, HIDDEN, device=device, dtype=torch.bfloat16)
    residual = torch.randn(T, TOTAL, device=device, dtype=torch.bfloat16)
    h_post = torch.randn(T, M, device=device, dtype=torch.float32)
    h_res = torch.randn(T, M * M, device=device, dtype=torch.float32)
    return x, residual, h_post, h_res


# ── high-level function benchmarks ──────────────────────────────────────

def bench_hc_pre():
    x, hc_fn, hc_scale, hc_base, _, _ = make_hc_pre_inputs(with_norm=False)

    def call_opt():
        return opt.hc_pre(
            x, hc_fn, hc_scale, hc_base, HC_MULT,
            RMS_EPS, HC_EPS, SINKHORN_ITERS, POST_MULT_VALUE,
        )

    def call_original():
        return original.hc_pre(
            x, hc_fn, hc_scale, hc_base, HC_MULT,
            RMS_EPS, HC_EPS, SINKHORN_ITERS, POST_MULT_VALUE,
        )

    return benchmark(
        call_opt,
        references=[Reference("original", call_original, lambda: call_original()[0].clone())],
        measurer=measurer,
        config=config,
        output_fn=lambda: call_opt()[0].clone(),
    )


def bench_hc_pre_fused_norm():
    x, hc_fn, hc_scale, hc_base, out_norm_weight, out_norm_eps = make_hc_pre_inputs(
        with_norm=True
    )

    def call_opt():
        return opt.hc_pre(
            x, hc_fn, hc_scale, hc_base, HC_MULT,
            RMS_EPS, HC_EPS, SINKHORN_ITERS, POST_MULT_VALUE,
            out_norm_weight=out_norm_weight, out_norm_eps=out_norm_eps,
        )

    def call_original():
        return original.hc_pre(
            x, hc_fn, hc_scale, hc_base, HC_MULT,
            RMS_EPS, HC_EPS, SINKHORN_ITERS, POST_MULT_VALUE,
            out_norm_weight=out_norm_weight, out_norm_eps=out_norm_eps,
        )

    return benchmark(
        call_opt,
        references=[Reference("original", call_original, lambda: call_original()[0].clone())],
        measurer=measurer,
        config=config,
        output_fn=lambda: call_opt()[0].clone(),
    )


def bench_hc_post():
    x, residual, h_post, h_res = make_hc_post_inputs()

    def call_opt():
        return opt.hc_post(x, residual, h_post, h_res, HC_MULT)

    def call_original():
        return original.hc_post(x, residual, h_post, h_res, HC_MULT)

    return benchmark(
        call_opt,
        references=[Reference("original", call_original, lambda: call_original().clone())],
        measurer=measurer,
        config=config,
        output_fn=lambda: call_opt().clone(),
    )


# ── kernel-level benchmarks ─────────────────────────────────────────────

def bench_project_mhc():
    x = torch.randn(T, TOTAL, device=device, dtype=torch.bfloat16)
    weight = torch.randn(STRIDE, TOTAL, device=device, dtype=torch.float32)

    out_opt = torch.empty((T, STRIDE), device=device, dtype=torch.float32)
    out_ref = torch.empty((T, STRIDE), device=device, dtype=torch.float32)

    def call_opt():
        opt._hc_projection_fp32[(T, STRIDE)](
            x, weight, out_opt, TOTAL,
            enable_fp_fusion=False, enable_auto_bind_sub_block=False,
        )

    def call_original():
        original._hc_projection_fp32[(T, STRIDE)](
            x, weight, out_ref, TOTAL,
            enable_fp_fusion=False, enable_auto_bind_sub_block=False,
        )

    return benchmark(
        call_opt,
        references=[Reference("original", call_original, lambda: out_ref.clone())],
        measurer=measurer,
        config=config,
        output_fn=lambda: out_opt.clone(),
    )


def bench_hc_coefficients():
    x = torch.randn(T, TOTAL, device=device, dtype=torch.bfloat16)
    mixes = torch.randn(T, STRIDE, device=device, dtype=torch.float32)
    hc_scale = torch.randn(3, device=device, dtype=torch.float32)
    hc_base = torch.randn(STRIDE, device=device, dtype=torch.float32)

    pre_opt = torch.empty(T, M, device=device, dtype=torch.float32)
    post_opt = torch.empty(T, M, device=device, dtype=torch.float32)
    comb_opt = torch.empty(T, M * M, device=device, dtype=torch.float32)
    pre_ref = torch.empty_like(pre_opt)
    post_ref = torch.empty_like(post_opt)
    comb_ref = torch.empty_like(comb_opt)

    kw = dict(
        TOTAL=TOTAL, BT=triton.next_power_of_2(TOTAL), M=M,
        RMS_EPS=RMS_EPS, EPS=HC_EPS, ITERS=SINKHORN_ITERS, POST_MULT=POST_MULT_VALUE,
        num_warps=1, enable_fp_fusion=False, enable_auto_bind_sub_block=False,
    )

    def call_opt():
        opt._hc_coefficients[(T,)](
            x, mixes, hc_scale, hc_base, pre_opt, post_opt, comb_opt, **kw,
        )

    def call_original():
        original._hc_coefficients[(T,)](
            x, mixes, hc_scale, hc_base, pre_ref, post_ref, comb_ref, **kw,
        )

    return benchmark(
        call_opt,
        references=[Reference("original", call_original,
                              lambda: (pre_ref.clone(), post_ref.clone(), comb_ref.clone()))],
        measurer=measurer,
        config=config,
        output_fn=lambda: (pre_opt.clone(), post_opt.clone(), comb_opt.clone()),
    )


def bench_hc_pre_apply():
    x = torch.randn(T, TOTAL, device=device, dtype=torch.bfloat16)
    pre = torch.randn(T, M, device=device, dtype=torch.float32)

    out_opt = torch.empty(T, HIDDEN, device=device, dtype=torch.bfloat16)
    out_ref = torch.empty_like(out_opt)

    grid = (T, triton.cdiv(HIDDEN, 1024))
    kw = dict(H=HIDDEN, M=M, B=1024, num_warps=1,
              enable_fp_fusion=False, enable_auto_bind_sub_block=False)

    def call_opt():
        opt._hc_pre_apply[grid](x, pre, out_opt, **kw)

    def call_original():
        original._hc_pre_apply[grid](x, pre, out_ref, **kw)

    return benchmark(
        call_opt,
        references=[Reference("original", call_original, lambda: out_ref.clone())],
        measurer=measurer,
        config=config,
        output_fn=lambda: out_opt.clone(),
    )


def bench_hc_pre_apply_norm():
    x = torch.randn(T, TOTAL, device=device, dtype=torch.bfloat16)
    pre = torch.randn(T, M, device=device, dtype=torch.float32)
    weight = torch.randn(HIDDEN, device=device, dtype=torch.bfloat16)

    out_opt = torch.empty(T, HIDDEN, device=device, dtype=torch.bfloat16)
    out_ref = torch.empty_like(out_opt)

    kw = dict(enable_fp_fusion=False, enable_auto_bind_sub_block=False)

    def call_opt():
        opt._hc_pre_apply_norm[(T,)](x, pre, weight, out_opt, HIDDEN, OUT_NORM_EPS, **kw)

    def call_original():
        original._hc_pre_apply_norm[(T,)](x, pre, weight, out_ref, HIDDEN, OUT_NORM_EPS, **kw)

    return benchmark(
        call_opt,
        references=[Reference("original", call_original, lambda: out_ref.clone())],
        measurer=measurer,
        config=config,
        output_fn=lambda: out_opt.clone(),
    )


def bench_hc_post_apply():
    x = torch.randn(T, HIDDEN, device=device, dtype=torch.bfloat16)
    residual = torch.randn(T, TOTAL, device=device, dtype=torch.bfloat16)
    h_post = torch.randn(T, M, device=device, dtype=torch.float32)
    h_res = torch.randn(T, M * M, device=device, dtype=torch.float32)

    out_opt = torch.empty(T, TOTAL, device=device, dtype=torch.bfloat16)
    out_ref = torch.empty_like(out_opt)

    grid = (T, triton.cdiv(HIDDEN, 4096))
    kw = dict(H=HIDDEN, M=M, B=4096, num_warps=1,
              enable_fp_fusion=False, enable_auto_bind_sub_block=False)

    def call_opt():
        opt._hc_post_apply[grid](x, residual, h_post, h_res, out_opt, **kw)

    def call_original():
        original._hc_post_apply[grid](x, residual, h_post, h_res, out_ref, **kw)

    return benchmark(
        call_opt,
        references=[Reference("original", call_original, lambda: out_ref.clone())],
        measurer=measurer,
        config=config,
        output_fn=lambda: out_opt.clone(),
    )


# ── main ────────────────────────────────────────────────────────────────

def main():
    results = [
        ("hc_pre", bench_hc_pre()),
        ("hc_pre (fused norm)", bench_hc_pre_fused_norm()),
        ("hc_post", bench_hc_post()),
        ("project_mhc", bench_project_mhc()),
        ("_hc_coefficients", bench_hc_coefficients()),
        ("_hc_pre_apply", bench_hc_pre_apply()),
        ("_hc_pre_apply_norm", bench_hc_pre_apply_norm()),
        ("_hc_post_apply", bench_hc_post_apply()),
    ]
    print()
    print_results_table(results)


if __name__ == "__main__":
    main()
