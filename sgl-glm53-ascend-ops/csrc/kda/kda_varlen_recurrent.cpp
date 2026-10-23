// GLM-5.3 KDA varlen recurrence — host op: validation, dispatch, registration.
//
// Substitutes for the implementation-selection block inside sglang's
// `glm_kda_varlen_recurrent_npu` — both of its arms, not for the Python
// function around it. Upstream that block picks between two implementations:
// `run_prepared_prefill` (the prepared fast path, for single-sequence prefill
// at H=4/8) and the `_glm_kda_varlen_recurrent_kernel[grid](...)` launch (the
// generic Triton kernel, for everything else). Here that choice is the
// prefill/decode launcher table, so the whole block collapses to one call.
//
// The argument list mirrors the Triton kernel's, since that is the wider of the
// two signatures, minus everything the host can work out for itself: the state
// and intermediate strides come from `.stride()`, the `SAVE_INTERMEDIATE` and
// `TRACK_STATE` flags from whether a tensor was passed, the head and feature
// widths from the shapes. The tiling constants (`BK`/`BV`/`BHV`, the grid,
// `num_warps`) do not survive the move off Triton at all, and neither does the
// prepared path's FP32 scratch: unlike `prepared_recurrent`, which takes
// pre-normalized inputs, this op takes the raw bf16 q/k/v/a/b and folds the
// gate, beta and q/k L2 normalization in, so the caller materializes nothing.
//
// The caller keeps what it already does before the selection: the upstream
// shape/metadata `ValueError`s, `zeros_like(v)`, the `scale` default,
// `.contiguous()`. None of it is repeated here. What this file checks is what
// the kernels index against and cannot detect themselves, and what it decides
// is which kernel runs.
//
// This file holds no device code. The five kernels live in their own
// translation units, reached through the launchers declared in
// `kda_varlen_recurrent.h`, which also documents the split, the division of
// labour with the caller, and the shared contract (state layout, reference
// semantics). Registration stays here and only here: `TORCH_LIBRARY_FRAGMENT`
// may define the schema once.
//
// The kernels themselves are still unimplemented: dispatch picks the right
// launcher, then stops at an explicit check.

#include "kda_varlen_recurrent.h"

#include <utility>

#include <acl/acl.h>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

// Argument names and order follow sglang's `glm_kda_varlen_recurrent_npu`, so
// the mapping from the call site stays one-for-one and a reader does not need a
// glossary to move between the two. Two deliberate departures, both because
// the op replaces the selection block rather than the function:
//   * `lower_bound` and `scale` are plain floats, not optionals: upstream has
//     already rejected a null or non-negative bound and resolved `scale` to
//     K**-0.5, both before the selection;
//   * every argument is keyword-only, matching the `*` in the Python signature.
//
// The output is allocated here with `zeros_like(v)` and returned, as upstream
// does, so the state pool stays the one in-place argument. Zeroed rather than
// empty because a launcher covers only its own blocks.
at::Tensor kda_varlen_recurrent(
    const at::Tensor &A_log,                 // fp32, H elements
    const at::Tensor &a,                     // bf16 [1,T,H,K]
    const at::Tensor &dt_bias,               // fp32 [H*D]
    const at::Tensor &q,                     // bf16 [1,T,H,D]
    const at::Tensor &k,                     // bf16 [1,T,H,D]
    const at::Tensor &v,                     // bf16 [1,T,H,D]
    const at::Tensor &b,                     // bf16 [1,T,H]
    at::Tensor &initial_state_source,        // fp32 [slots,H,K,V], K-contiguous
    const at::Tensor &initial_state_indices, // int32/int64 [numReq]
    const at::Tensor &cu_seqlens,            // int32 [numReq+1]
    double lower_bound,
    double scale,
    const c10::optional<at::Tensor> &intermediate_state, // fp32 5-D, unsupported
    bool prefill,
    const c10::optional<at::Tensor> &track_state_indices, // int32 [numReq]
    const c10::optional<at::Tensor> &track_state_lens)    // int32 [numReq]
{
    // Upstream's shape and metadata contract — the eleven `ValueError` checks,
    // the beta and gate shapes, the HV/H divisibility, the one-index-per-
    // sequence rule, the negative `lower_bound`, the prefix-checkpoint
    // metadata — is validated by `glm_kda_varlen_recurrent_npu` before the
    // selection block this op replaces, so none of it is repeated here. What
    // follows is only what the kernels index against and cannot detect
    // themselves: get one of these wrong and the result is out-of-bounds
    // memory traffic, not a wrong answer.
    const int64_t num_q_heads = k.size(2);
    const int64_t key_dim = k.size(3);
    const int64_t num_value_heads = v.size(2);
    const int64_t value_dim = v.size(3);
    const bool track_state = track_state_indices.has_value();

    // Target verify (upstream's SAVE_INTERMEDIATE) has no per-token checkpoint
    // path here, so refuse rather than silently leave the buffer unwritten.
    TORCH_CHECK(!intermediate_state.has_value(),
                "kda_varlen_recurrent does not support intermediate_state");
    TORCH_CHECK(q.device().is_privateuseone(), "q must be an NPU tensor");
    for (const auto &named : {std::pair<const char *, const at::Tensor &>{"q", q},
                              {"k", k},
                              {"v", v},
                              {"a", a},
                              {"b", b}}) {
        TORCH_CHECK(named.second.scalar_type() == at::kBFloat16,
                    named.first, " must be bf16");
        TORCH_CHECK(named.second.is_contiguous(),
                    named.first, " must be contiguous");
        TORCH_CHECK(named.second.device() == q.device(),
                    named.first, " must be on the same device as q");
    }
    TORCH_CHECK(num_q_heads == num_value_heads,
                "kda_varlen_recurrent requires H == HV, got ", num_q_heads,
                " and ", num_value_heads);
    TORCH_CHECK(key_dim == config_D && value_dim == config_D,
                "kda_varlen_recurrent requires K == V == ", config_D, ", got ",
                key_dim, " and ", value_dim);
    TORCH_CHECK(A_log.scalar_type() == at::kFloat && A_log.numel() == num_value_heads,
                "A_log must be fp32 with HV elements");
    TORCH_CHECK(dt_bias.scalar_type() == at::kFloat &&
                    dt_bias.numel() == num_value_heads * key_dim,
                "dt_bias must be fp32 with HV*K elements");
    TORCH_CHECK(initial_state_source.dim() == 4 &&
                    initial_state_source.scalar_type() == at::kFloat,
                "state must be a 4-D fp32 [slots,HV,K,V] pool");
    TORCH_CHECK(initial_state_source.size(1) == num_value_heads &&
                    initial_state_source.size(2) == key_dim &&
                    initial_state_source.size(3) == value_dim,
                "state must be [slots,HV,K,V] matching the head and feature dims");
    // The only layout that reaches the kernel in production: the backend
    // normalizes both state pools to it, and the generic Triton kernel is
    // numerically wrong in the transposed one. See the header note.
    TORCH_CHECK(initial_state_source.stride(2) == 1 &&
                    initial_state_source.stride(3) == config_D,
                "state must be K-contiguous (stride(2)==1, stride(3)==",
                config_D, "), got strides (", initial_state_source.stride(2),
                ", ", initial_state_source.stride(3), ")");
    for (const auto &named :
         {std::pair<const char *, const at::Tensor &>{"initial_state_indices",
                                                      initial_state_indices},
          {"cu_seqlens", cu_seqlens}}) {
        TORCH_CHECK(named.second.scalar_type() == at::kInt ||
                        named.second.scalar_type() == at::kLong,
                    named.first, " must be int32 or int64");
        TORCH_CHECK(named.second.is_contiguous(),
                    named.first, " must be contiguous");
        TORCH_CHECK(named.second.device() == q.device(),
                    named.first, " must be on the same device as q");
    }

    const int64_t num_tokens = q.size(1);
    const int64_t num_req = initial_state_indices.numel();

    auto output = at::zeros_like(v);

    auto gm = [](const at::Tensor &t) -> uint8_t * {
        return reinterpret_cast<uint8_t *>(const_cast<void *>(t.data_ptr()));
    };
    auto gm_opt = [&gm](const c10::optional<at::Tensor> &t) -> uint8_t * {
        return t.has_value() ? gm(*t) : nullptr;
    };

    // Scratch for the H=4 prefill kernel's producer blocks. Upstream these are
    // the four tensors `_prepare_inputs` fills in a separate Triton launch;
    // here the producers in the same grid fill them. Only that kernel reads
    // them, so they are allocated only for it — the other four launchers get
    // nulls and never dereference them.
    //
    // They are held in locals rather than passed straight into the args so
    // their lifetime covers the launch: the launch is asynchronous on the
    // stream, but these are caching-allocator tensors and are only returned to
    // the pool once the stream has reached the free, so holding the handles
    // here is enough.
    at::Tensor qn, kn, gdecay, beta, sync;
    const bool needs_scratch =
        prefill && (num_value_heads == 4 || num_value_heads == 8);
    if (needs_scratch) {
        const auto f32 = v.options().dtype(at::kFloat);
        qn = at::empty({1, num_tokens, num_value_heads, value_dim}, f32);
        kn = at::empty({1, num_tokens, num_value_heads, value_dim}, f32);
        gdecay = at::empty({1, num_tokens, num_value_heads, value_dim}, f32);
        // Beta is [H, Tpad], not upstream's [T, H]: a producer owns one head
        // and stores a chunk at a time, and DMA writes whole 32 B datablocks,
        // so in the [T, H] layout storing head h's chunk would also clobber
        // the three heads written concurrently by six other cores. Rounding T
        // up to a whole chunk gives each store its own cache line. Must match
        // `betaStride_` in the kernel.
        const int64_t t_pad =
            (num_tokens + kda_prefill::kChunkTokens - 1) / kda_prefill::kChunkTokens *
            kda_prefill::kChunkTokens;
        beta = at::empty({num_value_heads, t_pad}, f32);
        // The IBSet/IBWait banks. Zeroed, not empty: the API requires a
        // zero-initialized workspace, and a stale nonzero flag either blocks a
        // producer forever or lets a consumer read a chunk never written.
        sync = at::zeros({static_cast<int64_t>(kda_prefill::kSyncInt32Count)},
                         v.options().dtype(at::kInt));
    }

    KdaLaunchArgs args{
        /*q=*/gm(q),
        /*k=*/gm(k),
        /*v=*/gm(v),
        /*a=*/gm(a),
        /*b=*/gm(b),
        /*A_log=*/gm(A_log),
        /*dt_bias=*/gm(dt_bias),
        /*state=*/gm(initial_state_source),
        /*indices=*/gm(initial_state_indices),
        /*starts=*/gm(cu_seqlens),
        /*out=*/gm(output),
        /*track_indices=*/gm_opt(track_state_indices),
        /*track_lens=*/gm_opt(track_state_lens),
        /*qn=*/needs_scratch ? gm(qn) : nullptr,
        /*kn=*/needs_scratch ? gm(kn) : nullptr,
        /*gdecay=*/needs_scratch ? gm(gdecay) : nullptr,
        /*beta=*/needs_scratch ? gm(beta) : nullptr,
        /*sync=*/needs_scratch ? gm(sync) : nullptr,
        /*scale=*/static_cast<float>(scale),
        /*lower_bound=*/static_cast<float>(lower_bound),
        /*num_req=*/static_cast<uint32_t>(num_req),
        /*num_tokens=*/static_cast<uint32_t>(num_tokens),
        /*track_state=*/track_state,
    };

    void *stream = c10_npu::getCurrentNPUStream().stream();

    // This is the selection the op exists to absorb: upstream it was the
    // `if prefill and ... and num_q_heads in (4, 8)` condition choosing between
    // `run_prepared_prefill` and the generic Triton launch. Prefill is passed
    // explicitly rather than inferred, because `T == numReq` cannot tell the
    // regimes apart when both are 1.
    //
    // Selection is on H, which is set by the parallel strategy
    // (H = num_heads / attn_tp_size). GLM-5.3-Flash has 64 heads total and the
    // supported strategies are TP16/TP8 for prefill and TP4/TP2/TP1 for decode,
    // so these five widths are the whole space — an H outside it is a
    // misconfiguration, not a shape to accommodate. Raising names the widths
    // that exist; the caller keeps its own fallback for the regimes the
    // launcher set does not cover (ragged extend, multi-sequence prefill,
    // target verify).
    void (*launch)(const KdaLaunchArgs &, void *) = nullptr;
    if (prefill) {
        TORCH_CHECK(num_req == 1,
                    "kda_varlen_recurrent prefill expects one packed sequence, "
                    "got ", num_req);
        switch (num_value_heads) {
            case 4:  launch = &LaunchKdaPrefillH4; break;
            case 8:  launch = &LaunchKdaPrefillH8; break;
            default:
                TORCH_CHECK(false,
                            "kda_varlen_recurrent prefill supports H=4 (TP16) "
                            "or H=8 (TP8), got H=", num_value_heads);
        }
    } else {
        TORCH_CHECK(num_tokens == num_req,
                    "kda_varlen_recurrent decode expects one token per request "
                    "(T == numReq), got T=", num_tokens, " and numReq=", num_req);
        TORCH_CHECK(!track_state,
                    "kda_varlen_recurrent tracking is prefill-only");
        switch (num_value_heads) {
            case 16: launch = &LaunchKdaDecodeH16; break;
            case 32: launch = &LaunchKdaDecodeH32; break;
            case 64: launch = &LaunchKdaDecodeH64; break;
            default:
                TORCH_CHECK(false,
                            "kda_varlen_recurrent decode supports H=16 (TP4), "
                            "H=32 (TP2) or H=64 (TP1), got H=", num_value_heads);
        }
    }

    // TODO(kernel): drop this once the three decode kernels compute. Their
    // Init/Process are still empty, so launching one would enqueue a no-op and
    // return a silently zero `output`. `needs_scratch` is exactly the two
    // prefill paths, which are the implemented ones. Checking after selection
    // means an unsupported H still reports as unsupported rather than as
    // unimplemented.
    TORCH_CHECK(needs_scratch, "kda_varlen_recurrent is not implemented yet");

    launch(args, stream);
    return output;
}

TORCH_LIBRARY_FRAGMENT(sgl_glm53_ascend, m) {
    // Keyword-only (leading `*`), following `glm_kda_varlen_recurrent_npu`.
    // The returned output is freshly allocated, so it carries no alias
    // annotation; the state pool is the one mutated argument. `scale` and
    // `lower_bound` are non-optional because the caller resolves both before
    // this point.
    m.def("kda_varlen_recurrent(*, Tensor A_log, Tensor a, Tensor dt_bias, "
          "Tensor q, Tensor k, Tensor v, Tensor b, "
          "Tensor(a!) initial_state_source, Tensor initial_state_indices, "
          "Tensor cu_seqlens, float lower_bound, float scale, "
          "Tensor(b!)? intermediate_state=None, bool prefill=False, "
          "Tensor? track_state_indices=None, "
          "Tensor? track_state_lens=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(sgl_glm53_ascend, PrivateUse1, m) {
    m.impl("kda_varlen_recurrent", &kda_varlen_recurrent);
}
