"""A Triton implementation of the chunked KDA forward, for TND varlen input.

Split across several kernels, which is what the data dependencies and this
target's on-chip budget allow: the chunk-local work is embarrassingly parallel,
but the state recurrence across chunks is serial per sequence, so it gets its
own pass and is kept as small as possible.

  1. ``kda_gates``        per (chunk, head, K-tile): activate the gate,
                          cumulative-sum it within the chunk, and emit the
                          gate-scaled q and k.
  2. ``kda_attn``         per (chunk, head, row-block): the two chunk-local
                          pairwise matrices, ``Aqk`` and the strictly-lower
                          ``L``.
  3. ``kda_akk`` and
     ``kda_akk_merge``    invert ``I + L``, hierarchically: ``BB``-wide
                          diagonal blocks, then one merge pass per level.
  4. ``kda_wu``           apply that inverse to the w and u seeds -- the WY
                          representation. One kernel serves both.
  5. ``kda_scan``         per (sequence, head, value-tile): walk the chunks in
                          order, carrying the state; emits ``v_new`` and each
                          chunk's input state.
  6. ``kda_output``       per (chunk, head, value-tile):
                          ``qg @ h + Aqk @ v_new``.

The one real numerical constraint is in kernel 2. Reaching a matmul requires
the separable form ``(q 2^gk) (k 2^-gk)^T``, but the chunk-local cumulative
gate falls by roughly ``C * |gamma| / ln2`` across a chunk -- about 175 in
log2 for a 64-token chunk under the deployed bounded gate -- and ``2^175``
overflows fp32, so the naive factoring produces inf * 0 = NaN. The fix is to
split the chunk into ``BC``-row sub-tiles and rebase the exponent of each
sub-tile pair to the first row of the *column* tile. Both factors then stay
within ``2^(BC * |gamma| / ln2)`` -- at worst 115 for BC=16 under the bounded
gate, and about 44 on the gate values actually seen -- while their product is
unchanged, since the rebasing terms cancel exactly. That margin is only as good
as the bound on ``gamma``, so ``chunk_kda_triton`` checks it against ``BC``.

Correctness is anchored on ``reference.py``, not on this file.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl

LN2 = math.log(2.0)
# constexpr so the kernels can read it; a plain float global is rejected.
INV_LN2 = tl.constexpr(1.0 / LN2)


@triton.jit
def _activate(g, A_log, dt_bias, SAFE_GATE: tl.constexpr,
              LOWER_BOUND, USE_GATE: tl.constexpr):
    """The activated natural-log gate, matching the op's three branches."""
    if USE_GATE:
        x = g + dt_bias
        a = tl.exp(A_log)
        if SAFE_GATE:
            return LOWER_BOUND * tl.sigmoid(a * x)
        # softplus, evaluated so large x does not overflow the exp.
        sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(tl.minimum(x, 20.0))))
        return -a * sp
    return g


@triton.jit
def kda_gates(
    q_ptr, k_ptr, g_ptr, A_log_ptr, dt_bias_ptr, cu_ptr, chunk_ptr,
    gk_ptr, gk_last_ptr, qg_ptr, kg_ptr,
    scale, LOWER_BOUND,
    T, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, G: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, SAFE_GATE: tl.constexpr,
    USE_GATE: tl.constexpr, HAS_DT_BIAS: tl.constexpr,
):
    # Tiled over K as well as (chunk, head): every column of the gate is
    # independent, since the cumsum runs along the token axis, and a full
    # [BT, K] fp32 working set with this many live temporaries overflows UB.
    i_c, i_hv, i_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n = tl.load(chunk_ptr + i_c * 2).to(tl.int32)
    i_t = tl.load(chunk_ptr + i_c * 2 + 1).to(tl.int32)

    bos = tl.load(cu_ptr + i_n).to(tl.int32)
    eos = tl.load(cu_ptr + i_n + 1).to(tl.int32)
    start = bos + i_t * BT
    length = tl.minimum(BT, eos - start)

    rows = tl.arange(0, BT)
    dims = i_k * BK + tl.arange(0, BK)
    valid = rows < length
    m = valid[:, None]

    # Gate: activate, then cumulative-sum within the chunk in log2 units.
    # Zeroing the invalid rows *before* the cumsum makes every row at or past
    # `length` carry exactly gk_last, so the last row of the tile is gk_last
    # whatever the chunk's true length is -- no separate indexed read.
    g = tl.load(
        g_ptr + (start + rows)[:, None] * (HV * K) + i_hv * K + dims[None, :],
        mask=m, other=0.0,
    ).to(tl.float32)
    a_log = tl.load(A_log_ptr + i_hv).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + i_hv * K + dims).to(tl.float32)[None, :]
    else:
        dt_bias = 0.0
    gamma = _activate(g, a_log, dt_bias, SAFE_GATE, LOWER_BOUND, USE_GATE)
    gk = tl.cumsum(tl.where(m, gamma, 0.0) * INV_LN2, axis=0)
    gk_last = tl.sum(tl.where(rows[:, None] == BT - 1, gk, 0.0), 0)

    # GQA: this value head reads the query/key head it is grouped under.
    i_h = i_hv // G
    qk_off = (start + rows)[:, None] * (H * K) + i_h * K + dims[None, :]
    q = tl.load(q_ptr + qk_off, mask=m, other=0.0).to(tl.float32)
    k = tl.load(k_ptr + qk_off, mask=m, other=0.0).to(tl.float32)

    # `scale` is folded into qg here so the scan kernel is a plain matmul.
    # Both exponents below are <= 0, so neither can overflow.
    qg = q * scale * tl.exp2(gk)
    kg = k * tl.exp2(gk_last[None, :] - gk)

    out_off = i_hv * T * K + (start + rows)[:, None] * K + dims[None, :]
    tl.store(gk_ptr + out_off, gk, mask=m)
    tl.store(qg_ptr + out_off, qg, mask=m)
    tl.store(kg_ptr + out_off, kg, mask=m)
    tl.store(gk_last_ptr + i_c * (HV * K) + i_hv * K + dims, gk_last)


@triton.jit
def kda_attn(
    q_ptr, k_ptr, beta_ptr, gk_ptr, cu_ptr, chunk_ptr,
    aqk_ptr, l_ptr,
    scale,
    T, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, G: tl.constexpr,
    BT: tl.constexpr, BC: tl.constexpr,
):
    """One ``BC``-row strip of ``Aqk`` and ``L`` for one chunk and head.

    Column sub-tiles at or past the row strip are entirely masked out, so the
    loop stops at the diagonal.
    """
    i_c, i_hv, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n = tl.load(chunk_ptr + i_c * 2).to(tl.int32)
    i_t = tl.load(chunk_ptr + i_c * 2 + 1).to(tl.int32)

    bos = tl.load(cu_ptr + i_n).to(tl.int32)
    eos = tl.load(cu_ptr + i_n + 1).to(tl.int32)
    start = bos + i_t * BT
    length = tl.minimum(BT, eos - start)

    # A row strip that starts at or past the chunk's end has nothing to store,
    # and its `ref` load below -- the one load here that cannot be masked, since
    # it is a single row broadcast over the tile -- would read past the end of
    # `gk`. On a short tail chunk of the last sequence that is off the end of the
    # tensor. Skipping the strip outright is also less work than masking it: the
    # rows it would write are never read, because every consumer of `aqk`/`L`
    # masks to `rows < length` or (in `kda_akk`) relies on them being zero, and
    # the caller zero-initializes both buffers.
    if i_b * BC < length:
        dims = tl.arange(0, K)
        rows_i = i_b * BC + tl.arange(0, BC)
        mi = (rows_i < length)[:, None]
        i_h = i_hv // G

        gk_base = gk_ptr + i_hv * T * K
        qk_base_i = (start + rows_i)[:, None] * (H * K) + i_h * K + dims[None, :]
        q_i = tl.load(q_ptr + qk_base_i, mask=mi, other=0.0).to(tl.float32)
        k_i = tl.load(k_ptr + qk_base_i, mask=mi, other=0.0).to(tl.float32)
        gk_i = tl.load(
            gk_base + (start + rows_i)[:, None] * K + dims[None, :],
            mask=mi, other=0.0,
        ).to(tl.float32)
        beta_i = tl.load(
            beta_ptr + (start + rows_i) * HV + i_hv, mask=rows_i < length, other=0.0,
        ).to(tl.float32)

        for i_j in range(0, i_b + 1):
            rows_j = i_j * BC + tl.arange(0, BC)
            mj = (rows_j < length)[:, None]
            qk_base_j = (start + rows_j)[:, None] * (H * K) + i_h * K + dims[None, :]
            k_j = tl.load(k_ptr + qk_base_j, mask=mj, other=0.0).to(tl.float32)
            gk_j = tl.load(
                gk_base + (start + rows_j)[:, None] * K + dims[None, :],
                mask=mj, other=0.0,
            ).to(tl.float32)

            # Rebase both sides to the first row of the column tile. The two
            # exponents sum to gk_i - gk_j exactly, but each stays bounded by one
            # sub-tile's worth of decay instead of a whole chunk's.
            #
            # In bounds without a mask: i_j <= i_b and the strip guard above
            # gives i_j * BC <= i_b * BC < length, so this row is inside the
            # chunk.
            ref = tl.load(
                gk_base + (start + i_j * BC) * K + dims
            ).to(tl.float32)[None, :]
            dec_i = tl.exp2(gk_i - ref)
            dec_j = tl.exp2(ref - gk_j)

            a_blk = tl.dot(q_i * dec_i, tl.trans(k_j * dec_j)) * scale
            l_blk = tl.dot(k_i * dec_i, tl.trans(k_j * dec_j)) * beta_i[:, None]

            causal = rows_i[:, None] >= rows_j[None, :]
            strict = rows_i[:, None] > rows_j[None, :]
            keep = mi & (rows_j < length)[None, :]
            a_blk = tl.where(causal & keep, a_blk, 0.0)
            l_blk = tl.where(strict & keep, l_blk, 0.0)

            off = i_hv * T * BT + (start + rows_i)[:, None] * BT + rows_j[None, :]
            tl.store(aqk_ptr + off, a_blk, mask=keep)
            tl.store(l_ptr + off, l_blk, mask=keep)


@triton.jit
def kda_akk(
    l_ptr, cu_ptr, chunk_ptr, akk_ptr,
    T, HV: tl.constexpr, BT: tl.constexpr,
    BB: tl.constexpr, LOG2_BB: tl.constexpr,
):
    """Inverts one ``BB``-wide diagonal block of ``I + L``, by doubling.

    ``L`` is strictly lower triangular, hence nilpotent with ``L^BB = 0``, so
    the Neumann series terminates and the inverse is exactly

        (I + L)^-1 = sum_{p=0}^{BB-1} (-L)^p.

    Accumulating that sum by doubling -- square the power, then fold it into the
    running sum, which doubles the number of terms each step -- reaches all
    ``BB`` terms in ``log2(BB)`` steps. For BB=64 that is 6 iterations over
    three [BB, BB] tiles, against the 63 serial steps of the textbook row
    substitution, whose unrolled live temporaries overflow UB on this target by
    about 2.5x.

    Blocked at ``BB`` rather than done over the whole chunk because a [128, 128]
    fp32 working set does not fit either; ``kda_akk_merge`` then stitches the
    diagonal inverses together. Separate from the w/u projections for the same
    budget reason.

    ``BB`` below the largest that fits is also a work reduction, not just a
    budget concession: doubling costs ``2 * log2(BB)`` matmuls of ``BB^3``, which
    is super-cubic in ``BB``, while the merges that put the halves back together
    are two matmuls per level. Halving ``BB`` therefore cuts the matmul count --
    see the note on ``BB`` in ``chunk_kda_triton``.
    """
    i_c, i_hv, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n = tl.load(chunk_ptr + i_c * 2).to(tl.int32)
    i_t = tl.load(chunk_ptr + i_c * 2 + 1).to(tl.int32)

    bos = tl.load(cu_ptr + i_n).to(tl.int32)
    start = bos + i_t * BT + i_b * BB

    eos = tl.load(cu_ptr + i_n + 1).to(tl.int32)
    length = tl.minimum(BB, eos - start)

    rows = tl.arange(0, BB)
    cols = tl.arange(0, BB)

    # The load runs unpredicated -- this kernel's live tiles sit near the UB
    # ceiling -- so the caller pads `l_ptr`/`akk_ptr` by one [BT, BT] tile to
    # keep a short final chunk's overshoot in bounds. Reading past the sequence
    # end is harmless: the inverse of a lower-triangular block is lower
    # triangular, so row r depends only on rows <= r, and the rows past the end
    # cannot reach a valid one.
    #
    # The *store* must be masked, though. A tail chunk shorter than BB would
    # otherwise write all BB rows and clobber the next sequence's rows, which
    # its own program is computing concurrently.
    l_off = i_hv * T * BT + (start + rows)[:, None] * BT + i_b * BB + cols[None, :]
    b_A = tl.load(l_ptr + l_off)

    # Series in -L, so the accumulated sum is (I + L)^-1 and not (I - L)^-1.
    # The identity diagonal is folded in here rather than kept as a separate
    # live tile: with it included, `acc` *is* the running partial sum and the
    # step below is a single in-place update.
    b_A = tl.where(rows[:, None] > cols[None, :], -b_A, 0.0)
    acc = b_A + (rows[:, None] == cols[None, :]).to(tl.float32)

    # After step p, `acc` holds the first 2^(p+1) terms of the series.
    # The fold is `acc @ b_A + acc`, expressed as the matmul's accumulator so
    # the product does not need a separate live tile and a vector add.
    for _ in range(1, LOG2_BB):
        b_A = tl.dot(b_A, b_A)
        acc = tl.dot(acc, b_A, acc=acc)

    tl.store(akk_ptr + l_off, acc, mask=(rows < length)[:, None])


@triton.jit
def kda_akk_merge(
    l_ptr, akk_ptr, cu_ptr, chunk_ptr,
    T, HV: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr,
):
    """Joins two adjacent ``BS``-wide inverses into one ``2*BS``-wide inverse.

    For a 2x2 block lower-triangular matrix, the inverse is

        [A 0]^-1   [    A^-1          0   ]
        [C D]    = [ -D^-1 C A^-1   D^-1  ]

    so with both diagonal inverses already in place, the remaining block is one
    product of three tiles. ``C`` is read from ``L`` rather than ``I + L``
    because an off-diagonal block never touches the identity.

    Applied level by level, smallest ``BS`` first: ``kda_akk`` inverts the
    ``BB``-wide blocks on the diagonal, then this runs once per level for
    ``BS = BB, 2*BB, ... BT/2``, each level halving the number of independent
    blocks. ``A^-1`` and ``D^-1`` at level ``BS`` are whole sub-inverses the
    previous level assembled, including their own off-diagonals, which is what
    makes the recursion work. At ``BT == BB`` there are no levels and this never
    runs.

    One program per (chunk, head, block pair); ``i_p`` selects the pair, so the
    ``2*BS`` region it owns starts at row ``i_p * 2 * BS`` of the chunk.
    """
    i_c, i_hv, i_p = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n = tl.load(chunk_ptr + i_c * 2).to(tl.int32)
    i_t = tl.load(chunk_ptr + i_c * 2 + 1).to(tl.int32)

    bos = tl.load(cu_ptr + i_n).to(tl.int32)
    eos = tl.load(cu_ptr + i_n + 1).to(tl.int32)
    start = bos + i_t * BT
    r0 = i_p * 2 * BS

    rows = tl.arange(0, BS)
    cols = tl.arange(0, BS)
    base = (
        i_hv * T * BT + (start + r0 + rows)[:, None] * BT + r0 + cols[None, :]
    )
    # Masked for the same reason as in `kda_akk`: the rows this writes belong to
    # the lower block, which a short chunk may not reach at all. When it does
    # not, `d_inv` is still whatever `akk` was initialized to -- zero -- so the
    # unmasked loads below stay in bounds off the caller's pad and the product
    # they feed is discarded here.
    keep = (r0 + BS + rows < tl.minimum(BT, eos - start))[:, None]

    a_inv = tl.load(akk_ptr + base)
    d_inv = tl.load(akk_ptr + base + BS * BT + BS)
    c = tl.load(l_ptr + base + BS * BT)

    tl.store(
        akk_ptr + base + BS * BT, -tl.dot(d_inv, tl.dot(c, a_inv)), mask=keep,
    )


@triton.jit
def kda_wu(
    src_ptr, beta_ptr, gk_ptr, akk_ptr, cu_ptr, chunk_ptr, dst_ptr,
    T, HV: tl.constexpr, HS: tl.constexpr, DS: tl.constexpr,
    D: tl.constexpr, BD: tl.constexpr, G: tl.constexpr, BT: tl.constexpr,
    APPLY_GATE: tl.constexpr,
):
    """``Akk @ (beta * src)``, optionally gate-scaling ``src`` first.

    One kernel for both w and u: ``w`` is the ``k``-sourced case with
    ``APPLY_GATE=True`` (seed ``beta * k * 2^gk``), ``u`` the ``v``-sourced case
    without it (seed ``beta * v``). ``HS``/``DS`` are the source tensor's head
    count and head dim, which differ between the two because ``k`` is indexed by
    query head and ``v`` by value head.
    """
    i_c, i_hv, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n = tl.load(chunk_ptr + i_c * 2).to(tl.int32)
    i_t = tl.load(chunk_ptr + i_c * 2 + 1).to(tl.int32)

    bos = tl.load(cu_ptr + i_n).to(tl.int32)
    eos = tl.load(cu_ptr + i_n + 1).to(tl.int32)
    start = bos + i_t * BT
    length = tl.minimum(BT, eos - start)

    rows = tl.arange(0, BT)
    cols = tl.arange(0, BT)
    dims = i_d * BD + tl.arange(0, BD)
    valid = rows < length
    m = valid[:, None]

    akk = tl.load(
        akk_ptr + i_hv * T * BT + (start + rows)[:, None] * BT + cols[None, :],
        mask=m & (cols < length)[None, :], other=0.0,
    )
    i_src_head = i_hv // G if HS != HV else i_hv
    src = tl.load(
        src_ptr + (start + rows)[:, None] * (HS * DS) + i_src_head * DS
        + dims[None, :],
        mask=m, other=0.0,
    ).to(tl.float32)
    beta = tl.load(
        beta_ptr + (start + rows) * HV + i_hv, mask=valid, other=0.0,
    ).to(tl.float32)[:, None]

    seed = beta * src
    if APPLY_GATE:
        gk = tl.load(
            gk_ptr + i_hv * T * D + (start + rows)[:, None] * D + dims[None, :],
            mask=m, other=0.0,
        ).to(tl.float32)
        seed = seed * tl.exp2(gk)

    tl.store(
        dst_ptr + i_hv * T * D + (start + rows)[:, None] * D + dims[None, :],
        tl.dot(akk, seed), mask=m,
    )


@triton.jit
def kda_scan(
    kg_ptr, w_ptr, u_ptr, gk_last_ptr,
    init_ptr, cu_ptr, chunk_off_ptr,
    v_new_ptr, h_ptr, final_ptr,
    T, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr, HAS_INIT: tl.constexpr,
):
    """The serial part: carry the state across the chunks of one sequence.

    Parallel over value tiles -- each program owns a ``BV``-wide slice of the
    state's V axis and needs the full K axis, because every matmul here
    contracts over K.

    Deliberately narrow: this kernel computes ``v_new`` and the state
    recurrence and nothing else, writing the per-chunk input state ``h`` out
    for ``kda_output`` to consume. Folding the output matmuls in here as well
    costs an extra [BT, BT] operand tile and overflows cbuf at C=128 -- and the
    output is chunk-parallel anyway, so it does not belong in the one kernel
    that has to run serially.
    """
    i_n, i_hv, i_v = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    bos = tl.load(cu_ptr + i_n).to(tl.int32)
    eos = tl.load(cu_ptr + i_n + 1).to(tl.int32)
    # Flat index of this sequence's first chunk, so chunk metadata lines up.
    c0 = tl.load(chunk_off_ptr + i_n).to(tl.int32)
    n_chunks = tl.cdiv(eos - bos, BT)

    dims_k = tl.arange(0, K)
    dims_v = i_v * BV + tl.arange(0, BV)
    rows = tl.arange(0, BT)

    if HAS_INIT:
        h = tl.load(
            init_ptr + i_n * (HV * K * V) + i_hv * (K * V)
            + dims_k[:, None] * V + dims_v[None, :]
        ).to(tl.float32)
    else:
        h = tl.zeros([K, BV], dtype=tl.float32)

    for i_t in range(0, n_chunks):
        start = bos + i_t * BT
        length = tl.minimum(BT, eos - start)
        m = (rows < length)[:, None]

        k_off = i_hv * T * K + (start + rows)[:, None] * K + dims_k[None, :]
        v_off = i_hv * T * V + (start + rows)[:, None] * V + dims_v[None, :]

        # The state this chunk is entered with, which is what the output pass
        # needs and what the op exposes as `h`.
        tl.store(
            h_ptr + (c0 + i_t) * (HV * K * V) + i_hv * (K * V)
            + dims_k[:, None] * V + dims_v[None, :],
            h,
        )

        w = tl.load(w_ptr + k_off, mask=m, other=0.0)
        u = tl.load(u_ptr + v_off, mask=m, other=0.0)
        kg = tl.load(kg_ptr + k_off, mask=m, other=0.0)

        v_new = u - tl.dot(w, h)
        tl.store(v_new_ptr + v_off, v_new, mask=m)

        gk_last = tl.load(
            gk_last_ptr + (c0 + i_t) * (HV * K) + i_hv * K + dims_k
        ).to(tl.float32)
        # The decayed state seeds the matmul's accumulator, so the rank-BT
        # update lands in place instead of into a second [K, BV] tile.
        h = tl.dot(tl.trans(kg), v_new, acc=tl.exp2(gk_last)[:, None] * h)

    tl.store(
        final_ptr + i_n * (HV * K * V) + i_hv * (K * V)
        + dims_k[:, None] * V + dims_v[None, :],
        h,
    )


@triton.jit
def kda_output(
    qg_ptr, aqk_ptr, v_new_ptr, h_ptr, cu_ptr, chunk_ptr, chunk_off_ptr,
    out_ptr,
    T, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr,
):
    """``attn_out = qg @ h + Aqk @ v_new``, one chunk per program.

    Split out of ``kda_scan`` so it runs across all chunks at once instead of
    inside the serial walk; `scale` is already folded into ``qg``.
    """
    i_c, i_hv, i_v = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n = tl.load(chunk_ptr + i_c * 2).to(tl.int32)
    i_t = tl.load(chunk_ptr + i_c * 2 + 1).to(tl.int32)

    bos = tl.load(cu_ptr + i_n).to(tl.int32)
    eos = tl.load(cu_ptr + i_n + 1).to(tl.int32)
    start = bos + i_t * BT
    length = tl.minimum(BT, eos - start)

    dims_k = tl.arange(0, K)
    dims_v = i_v * BV + tl.arange(0, BV)
    rows = tl.arange(0, BT)
    cols = tl.arange(0, BT)
    m = (rows < length)[:, None]

    qg = tl.load(
        qg_ptr + i_hv * T * K + (start + rows)[:, None] * K + dims_k[None, :],
        mask=m, other=0.0,
    )
    aqk = tl.load(
        aqk_ptr + i_hv * T * BT + (start + rows)[:, None] * BT + cols[None, :],
        mask=m & (cols < length)[None, :], other=0.0,
    )
    v_new = tl.load(
        v_new_ptr + i_hv * T * V + (start + rows)[:, None] * V + dims_v[None, :],
        mask=m, other=0.0,
    )
    h = tl.load(
        h_ptr + i_c * (HV * K * V) + i_hv * (K * V)
        + dims_k[:, None] * V + dims_v[None, :]
    )

    # Both terms accumulate in fp32 -- the second matmul adds into the first's
    # accumulator rather than producing a tile to be added -- and only the
    # store rounds, to `out`'s dtype.
    acc = tl.dot(qg, h)
    acc = tl.dot(aqk, v_new, acc=acc)

    tl.store(
        out_ptr + (start + rows)[:, None] * (HV * V) + i_hv * V + dims_v[None, :],
        acc.to(out_ptr.dtype.element_ty), mask=m,
    )


def chunk_meta(cu_seqlens: torch.Tensor, chunk_size: int, device=None):
    """``(chunk_indices, chunk_offsets)`` in sequence-major canonical order.

    ``chunk_indices`` is an int32 ``[NC, 2]`` of ``(sequence, chunk-in-sequence)``
    for every chunk, and ``chunk_offsets`` an int32 ``[N]`` giving each
    sequence's first flat chunk index.

    Hoisted out of ``chunk_kda_triton`` because it depends only on the sequence
    lengths and the chunk size, not on any layer's tensors: a device-to-host
    sync, a Python loop and two device allocations that a decoder would
    otherwise repeat for every layer of every step. Compute it once per batch
    and pass it in.
    """
    lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    pairs = [(n, c) for n, ln in enumerate(lens)
             for c in range((ln + chunk_size - 1) // chunk_size)]
    counts = [(ln + chunk_size - 1) // chunk_size for ln in lens]
    offsets = [0]
    for c in counts:
        offsets.append(offsets[-1] + c)
    device = cu_seqlens.device if device is None else device
    return (
        torch.tensor(pairs, dtype=torch.int32, device=device).reshape(-1, 2),
        torch.tensor(offsets[:-1], dtype=torch.int32, device=device),
    )


def chunk_kda_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    chunk_size: int,
    cu_seqlens: torch.Tensor,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    safe_gate: bool = True,
    lower_bound: float = -5.0,
    use_gate_in_kernel: bool = True,
    chunk_indices: Optional[torch.Tensor] = None,
    chunk_offsets: Optional[torch.Tensor] = None,
    BC: int = 16,
    BV_SCAN: int | None = None,
    BV_OUT: int | None = None,
    BK: int = 32,
    BB: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(attn_out, final_state)`` for TND varlen input, matching the op.

    ``q``/``k`` are ``[T, H, K]``, ``v`` is ``[T, HV, V]``, ``g`` is
    ``[T, HV, K]``, ``beta`` is ``[T, HV]``. Returns ``attn_out`` as
    ``[T, HV, V]`` in ``q``'s dtype and ``final_state`` as fp32
    ``[N, HV, K, V]``.

    ``chunk_indices``/``chunk_offsets`` are what ``chunk_meta`` returns. Pass
    them to skip the host-side metadata build, which is identical for every
    layer of a step; omitted, they are computed here.
    """
    t, h, kdim = q.shape
    hv, vdim = v.shape[1], v.shape[2]
    if hv % h:
        raise ValueError(f"HV={hv} must be a multiple of H={h}")
    if chunk_size % BC:
        raise ValueError(f"chunk_size={chunk_size} must be a multiple of BC={BC}")
    # `kda_wu` tiles both the K and the V axis at BD=BK, and neither its loads
    # nor the grid handle a remainder tile.
    for name, dim in (("K", kdim), ("V", vdim)):
        if dim % BK:
            raise ValueError(f"{name}={dim} must be a multiple of BK={BK}")

    # The rebasing in `kda_attn` bounds each factor's exponent by one BC-row
    # sub-tile of decay rather than a whole chunk's, which is what keeps the
    # separable form inside fp32. With the bounded gate that bound is static:
    # |gamma| < |lower_bound| per token, so the exponent cannot exceed
    # BC * |lower_bound| * log2(e) -- 115 at the deployed BC=16, lower_bound=-5,
    # against the 127 where exp2 overflows. Raising BC, or loosening
    # lower_bound, eats that margin directly and is checked here.
    #
    # With safe_gate=False there is no such bound: gamma = -exp(A_log) *
    # softplus(g + dt_bias) grows without limit in g, so no BC is provably safe
    # and a large enough input will produce inf and then NaN. Left unchecked
    # rather than guarded, because a correct guard needs the data -- a reduction
    # over g and a device sync on every call. The deployed path is safe_gate.
    if safe_gate and use_gate_in_kernel:
        exp_bound = BC * abs(lower_bound) / LN2
        if exp_bound >= 127.0:
            raise ValueError(
                f"BC={BC} with lower_bound={lower_bound} gives a worst-case "
                f"exponent of 2^{exp_bound:.0f}, which overflows fp32 in "
                f"kda_attn; lower BC to {int(127.0 * LN2 / abs(lower_bound))} "
                f"or less"
            )

    dev = q.device
    bt = chunk_size
    # The two kernels tile the V axis independently: they only have to agree on
    # the layout of `v_new` and `h_state`, not on how they walk it.
    #
    # `kda_output` wants the widest tile that fits. Its grid is over all chunks,
    # so it has parallelism to spare, and V=128 in one shot measures fastest at
    # every shape -- a chunk's `qg`, `aqk` and `h` are each read once per value
    # tile, so splitting V re-reads them.
    #
    # `kda_scan` is the opposite: one program per (sequence, head, value tile),
    # and with a single packed sequence that is only HV * V/BV programs. BV is
    # what sets occupancy there, not footprint, and the measured optimum tracks
    # the core count rather than the tile size -- 32 at HV=4, 64 at HV=8, both
    # landing near 16 programs. 32 is the compromise default and is within ~1%
    # of the per-shape best everywhere; override it if HV is unusual.
    #
    # Footprint still bounds it. `kda_scan` holds [BT, BT], [BT, K] and [K, BV]
    # operands at once, which used to cap BV at 16 for C=128; feeding the
    # decayed state in as the matmul's accumulator (see `kda_scan`) removed the
    # separate result tile, and 64 now fits at both chunk sizes.
    if BV_SCAN is None:
        BV_SCAN = 32
    if BV_OUT is None:
        BV_OUT = vdim
    for name, bv in (("BV_SCAN", BV_SCAN), ("BV_OUT", BV_OUT)):
        if vdim % bv:
            raise ValueError(f"V={vdim} must be a multiple of {name}={bv}")
    group = hv // h
    n = cu_seqlens.numel() - 1
    if chunk_indices is None or chunk_offsets is None:
        chunk_indices, chunk_offsets = chunk_meta(cu_seqlens.cpu(), bt, dev)
    nc = chunk_indices.shape[0]
    cu = cu_seqlens.to(device=dev, dtype=torch.int32)

    f32 = dict(dtype=torch.float32, device=dev)
    # `empty` wherever the producing kernel writes every element that is ever
    # read. The exceptions are `aqk`, `lmat` and `akk`: those are [BT, BT] and
    # the kernels that fill them write only the causal half and only the valid
    # rows, while `kda_wu`/`kda_output`/`kda_akk_merge` read regions wider than
    # that, relying on the rest being zero.
    gk = torch.empty((hv, t, kdim), **f32)
    qg = torch.empty((hv, t, kdim), **f32)
    kg = torch.empty((hv, t, kdim), **f32)
    gk_last = torch.empty((nc, hv, kdim), **f32)
    aqk = torch.zeros((hv, t, bt), **f32)
    # One extra [BT, BT] tile of slack: `kda_akk` runs its loads and stores
    # unmasked (see the note there), so a short final chunk overshoots the last
    # head's rows by up to BT.
    lmat = torch.zeros((hv * t * bt + bt * bt,), **f32)
    w = torch.empty((hv, t, kdim), **f32)
    u = torch.empty((hv, t, vdim), **f32)
    # Written in the output dtype directly: `kda_output` accumulates in fp32 and
    # rounds at the store, which saves a full [T, HV, V] cast on the way out.
    out = torch.empty((t, hv, vdim), dtype=q.dtype, device=dev)
    final_state = torch.empty((n, hv, kdim, vdim), **f32)

    if A_log is None:
        A_log = torch.zeros((hv,), **f32)
    has_dt_bias = dt_bias is not None
    if not has_dt_bias:
        dt_bias = torch.zeros((hv * kdim,), **f32)

    common = dict(T=t, H=h, HV=hv, K=kdim, G=group, BT=bt)

    kda_gates[(nc, hv, kdim // BK)](
        q, k, g, A_log, dt_bias, cu, chunk_indices,
        gk, gk_last, qg, kg, scale, lower_bound,
        BK=BK, SAFE_GATE=safe_gate, USE_GATE=use_gate_in_kernel,
        HAS_DT_BIAS=has_dt_bias, **common,
    )
    kda_attn[(nc, hv, bt // BC)](
        q, k, beta, gk, cu, chunk_indices, aqk, lmat, scale, BC=BC, **common,
    )
    akk = torch.zeros((hv * t * bt + bt * bt,), **f32)
    # The triangular inverse is hierarchical: `kda_akk` inverts BB-wide blocks
    # on the diagonal by doubling, then `kda_akk_merge` joins adjacent pairs one
    # level at a time until a whole BT chunk is covered.
    #
    # BB <= 64 is forced -- a [128, 128] fp32 working set with kda_akk's live
    # tiles does not fit in UB. Going *below* 64 cuts the arithmetic sharply but
    # measures slower, so 64 is the default and the smaller tiles are left as a
    # knob.
    #
    # The arithmetic says otherwise, which is why this is worth recording.
    # Doubling an n-wide block costs 2*log2(n) matmuls of n^3, against two per
    # merge, so per BT=128 rows BB=64 is 2 * 12 * 64^3 plus one 2 * 64^3 merge
    # while BB=32 is 4 * 10 * 32^3 plus 6 * 32^3 over two levels -- 22% of the
    # multiply-accumulates. Measured on 910B4 over H4/H8 x T1024..16384, BB=32
    # is instead 5-8% *slower* than BB=64 and BB=16 is 13-17% slower. Narrow
    # matmuls do not fill the cube, the merge levels are serial, and this whole
    # stage is a small part of the total. An 80% arithmetic saving being a net
    # loss is the useful part: the inverse is not where the time goes.
    bb = BB if BB is not None else min(bt, 64)
    if bb & (bb - 1) or bt % bb:
        raise ValueError(f"BB={bb} must be a power of two dividing BT={bt}")
    if bb > 64:
        raise ValueError(f"BB={bb} exceeds 64; kda_akk's tiles overflow UB")
    kda_akk[(nc, hv, bt // bb)](
        lmat, cu, chunk_indices, akk,
        T=t, HV=hv, BT=bt, BB=bb, LOG2_BB=bb.bit_length() - 1,
    )
    # One level per doubling of the block width: BS = bb, 2*bb, ... bt/2, with
    # bt/(2*BS) independent pairs at each. Levels are serial -- a merge consumes
    # what the level below assembled -- but the pairs within one are not.
    bs = bb
    while bs < bt:
        kda_akk_merge[(nc, hv, bt // (2 * bs))](
            lmat, akk, cu, chunk_indices, T=t, HV=hv, BT=bt, BS=bs,
        )
        bs *= 2
    # w from k with the gate applied, u from v without: same matmul, so one
    # kernel covers both. BD keeps each program's working set inside UB.
    kda_wu[(nc, hv, kdim // BK)](
        k, beta, gk, akk, cu, chunk_indices, w,
        T=t, HV=hv, HS=h, DS=kdim, D=kdim, BD=BK, G=group, BT=bt,
        APPLY_GATE=True,
    )
    kda_wu[(nc, hv, vdim // BK)](
        v, beta, gk, akk, cu, chunk_indices, u,
        T=t, HV=hv, HS=hv, DS=vdim, D=vdim, BD=BK, G=group, BT=bt,
        APPLY_GATE=False,
    )
    # `v_new` is written masked, so its padding rows stay uninitialized; the
    # only reader is `kda_output`, which masks the same rows.
    v_new = torch.empty((hv, t, vdim), **f32)
    h_state = torch.empty((nc, hv, kdim, vdim), **f32)
    kda_scan[(n, hv, vdim // BV_SCAN)](
        kg, w, u, gk_last,
        initial_state if initial_state is not None else qg,
        cu, chunk_offsets, v_new, h_state, final_state,
        T=t, HV=hv, K=kdim, V=vdim, BT=bt, BV=BV_SCAN,
        HAS_INIT=initial_state is not None,
    )
    kda_output[(nc, hv, vdim // BV_OUT)](
        qg, aqk, v_new, h_state, cu, chunk_indices, chunk_offsets, out,
        T=t, HV=hv, K=kdim, V=vdim, BT=bt, BV=BV_OUT,
    )

    return out, final_state
