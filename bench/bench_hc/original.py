"""Ascend graph-safe fused mHC; FP32 projection and mixing arithmetic."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _hc_coefficients(
    X,
    MIX,
    SCALE,
    BASE,
    PRE,
    POST,
    COMB,
    TOTAL: tl.constexpr,
    BT: tl.constexpr,
    M: tl.constexpr,
    RMS_EPS: tl.constexpr,
    EPS: tl.constexpr,
    ITERS: tl.constexpr,
    POST_MULT: tl.constexpr,
):
    t = tl.program_id(0)
    k = tl.arange(0, BT)
    x = tl.load(X + t * TOTAL + k, k < TOTAL, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / TOTAL + RMS_EPS)
    i = tl.arange(0, M)
    stride: tl.constexpr = 2 * M + M * M
    a = tl.load(MIX + t * stride + i) * inv
    b = tl.load(MIX + t * stride + M + i) * inv
    a = a * tl.load(SCALE) + tl.load(BASE + i)
    b = b * tl.load(SCALE + 1) + tl.load(BASE + M + i)
    pre = 1.0 / (1.0 + tl.exp(-a)) + EPS
    post = POST_MULT / (1.0 + tl.exp(-b))
    ij = i[:, None] * M + i[None, :]
    c = tl.load(MIX + t * stride + 2 * M + ij) * inv
    c = c * tl.load(SCALE + 2) + tl.load(BASE + 2 * M + ij)
    c = tl.exp(c - tl.max(c, 1)[:, None])
    c = c / tl.sum(c, 1)[:, None] + EPS
    c = c / (tl.sum(c, 0)[None, :] + EPS)
    for _ in range(ITERS - 1):
        c = c / (tl.sum(c, 1)[:, None] + EPS)
        c = c / (tl.sum(c, 0)[None, :] + EPS)
    tl.store(PRE + t * M + i, pre)
    tl.store(POST + t * M + i, post)
    tl.store(COMB + t * M * M + ij, c)


@triton.jit
def _hc_pre_apply(X, PRE, OUT, H: tl.constexpr, M: tl.constexpr, B: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1) * B + tl.arange(0, B)
    i = tl.arange(0, M)
    x = tl.load(X + t * M * H + i[:, None] * H + h[None, :], h[None, :] < H, 0).to(
        tl.float32
    )
    pre = tl.load(PRE + t * M + i)
    out = tl.sum(x * pre[:, None], 0)
    tl.store(OUT + t * H + h, out, h < H)


@triton.jit
def _hc_post_apply(
    X, RES, POST, COMB, OUT, H: tl.constexpr, M: tl.constexpr, B: tl.constexpr
):
    # Four contiguous residual vectors avoid the expensive strided broadcast
    # and reduction generated for [hc_mult, hidden_tile] on Ascend.
    tl.static_assert(M == 4)
    t = tl.program_id(0)
    h = tl.program_id(1) * B + tl.arange(0, B)
    x = tl.load(X + t * H + h, h < H, 0).to(tl.float32)
    r0 = tl.load(RES + t * 4 * H + h, h < H, 0).to(tl.float32)
    r1 = tl.load(RES + t * 4 * H + H + h, h < H, 0).to(tl.float32)
    r2 = tl.load(RES + t * 4 * H + 2 * H + h, h < H, 0).to(tl.float32)
    r3 = tl.load(RES + t * 4 * H + 3 * H + h, h < H, 0).to(tl.float32)
    for j in tl.static_range(4):
        c0 = tl.load(COMB + t * 16 + j)
        c1 = tl.load(COMB + t * 16 + 4 + j)
        c2 = tl.load(COMB + t * 16 + 8 + j)
        c3 = tl.load(COMB + t * 16 + 12 + j)
        post = tl.load(POST + t * 4 + j)
        # Preserve the existing Ascend reduction tree and FP32 rounding.
        mixed = (r0 * c0 + r2 * c2) + (r1 * c1 + r3 * c3)
        out = post * x + mixed
        tl.store(OUT + t * 4 * H + j * H + h, out, h < H)


@triton.jit
def _round_bf16_as_fp32(x):
    bits = x.to(tl.uint32, bitcast=True)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    # Preserve Inf; keep NaN a quiet NaN even if its payload is only low bits.
    nonfinite = (bits & 0x7F800000) == 0x7F800000
    payload = tl.where((bits & 0x007FFFFF) != 0, bits | 0x00400000, bits) & 0xFFFF0000
    result = tl.where(nonfinite, payload, rounded)
    return result.to(tl.float32, bitcast=True)


@triton.jit
def _hc_pre_apply_norm(X, PRE, WEIGHT, OUT, H: tl.constexpr, EPS: tl.constexpr):
    t = tl.program_id(0)
    h = tl.arange(0, H)
    r0 = tl.load(X + t * 4 * H + h).to(tl.float32)
    r1 = tl.load(X + t * 4 * H + H + h).to(tl.float32)
    r2 = tl.load(X + t * 4 * H + 2 * H + h).to(tl.float32)
    r3 = tl.load(X + t * 4 * H + 3 * H + h).to(tl.float32)
    p0 = tl.load(PRE + t * 4)
    p1 = tl.load(PRE + t * 4 + 1)
    p2 = tl.load(PRE + t * 4 + 2)
    p3 = tl.load(PRE + t * 4 + 3)
    mixed = (r0 * p0 + r2 * p2) + (r1 * p1 + r3 * p3)
    narrow = _round_bf16_as_fp32(mixed)
    inv = tl.rsqrt(tl.sum(narrow * narrow, 0) / H + EPS)
    w = tl.load(WEIGHT + h).to(tl.float32)
    tl.store(OUT + t * H + h, (narrow * inv) * w)


@triton.jit
def _hc_projection_fp32(X, WEIGHT, OUT, K: tl.constexpr):
    token = tl.program_id(0)
    row = tl.program_id(1)
    h = tl.arange(0, K)
    x = tl.load(X + token * K + h).to(tl.float32)
    weight = tl.load(WEIGHT + row * K + h)
    value = tl.sum(x * weight, 0)
    tl.store(OUT + token * 24 + row, value)


def project_mhc(x, weight):
    # The checkpoint stores this small mHC projection in FP32. Retain its
    # full precision and avoid launching a general matrix multiplication
    # for the small attention batch used by decode graphs.
    if (
        0 < x.shape[0] <= 16
        and x.shape[1] == 16384
        and x.dtype == torch.bfloat16
        and x.device.type == "npu"
        and x.is_contiguous()
        and weight.shape == (24, 16384)
        and weight.dtype == torch.float32
        and weight.device == x.device
        and weight.is_contiguous()
    ):
        output = torch.empty((x.shape[0], 24), device=x.device, dtype=torch.float32)
        _hc_projection_fp32[(x.shape[0], 24)](
            x,
            weight,
            output,
            16384,
            enable_fp_fusion=False,
            enable_auto_bind_sub_block=False,
        )
        return output
    return F.linear(x.float(), weight)


def hc_pre(
    x,
    hc_fn,
    hc_scale,
    hc_base,
    hc_mult,
    rms_eps,
    hc_eps,
    sinkhorn_iters,
    post_mult_value=2.0,
    hc_norm_weight=None,
    out_norm_weight=None,
    out_norm_eps=None,
):
    tokens, total = x.shape
    hidden = total // hc_mult
    layer_input = x.new_empty((tokens, hidden))
    comb = torch.empty(
        (tokens, hc_mult * hc_mult), device=x.device, dtype=torch.float32
    )
    post = torch.empty((tokens, hc_mult), device=x.device, dtype=torch.float32)
    fuse_norm = (
        tokens > 0
        and out_norm_weight is not None
        and out_norm_eps is not None
        and out_norm_weight.shape == (hidden,)
        and out_norm_weight.device == x.device
        and out_norm_weight.dtype in (torch.bfloat16, torch.float32)
        and out_norm_weight.is_contiguous()
    )
    if tokens:
        pre = torch.empty_like(post)
        mixes = project_mhc(x, hc_fn)
        _hc_coefficients[(tokens,)](
            x,
            mixes,
            hc_scale,
            hc_base,
            pre,
            post,
            comb,
            TOTAL=total,
            BT=triton.next_power_of_2(total),
            M=hc_mult,
            RMS_EPS=rms_eps,
            EPS=hc_eps,
            ITERS=sinkhorn_iters,
            POST_MULT=post_mult_value,
            num_warps=1,
            enable_fp_fusion=False,
            enable_auto_bind_sub_block=False,
        )
        if fuse_norm:
            _hc_pre_apply_norm[(tokens,)](
                x,
                pre,
                out_norm_weight,
                layer_input,
                hidden,
                out_norm_eps,
                enable_fp_fusion=False,
                enable_auto_bind_sub_block=False,
            )
        else:
            _hc_pre_apply[(tokens, triton.cdiv(hidden, 1024))](
                x,
                pre,
                layer_input,
                H=hidden,
                M=hc_mult,
                B=1024,
                num_warps=1,
                enable_fp_fusion=False,
                enable_auto_bind_sub_block=False,
            )
    return layer_input, comb, post, fuse_norm


def hc_post(x, residual, h_post, h_res, hc_mult):
    tokens, hidden = x.shape
    out = x.new_empty((tokens, hc_mult * hidden))
    if tokens:
        _hc_post_apply[(tokens, triton.cdiv(hidden, 4096))](
            x,
            residual,
            h_post,
            h_res,
            out,
            H=hidden,
            M=hc_mult,
            B=4096,
            num_warps=1,
            enable_fp_fusion=False,
            enable_auto_bind_sub_block=False,
        )
    return out
