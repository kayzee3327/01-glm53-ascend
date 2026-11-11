// GLM-5.3 KDA varlen recurrence v2 — host op: validation, dispatch,
// registration.
//
// Registered as `kda_varlen_recurrent_v2` during development.
// Cutover: change the TORCH_LIBRARY_FRAGMENT string to `kda_varlen_recurrent`
// and remove v1 from CMakeLists. The Python wrapper needs no change.

#include "kda_varlen_recurrent_v2.h"

#include <utility>

#include <acl/acl.h>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

at::Tensor kda_varlen_recurrent_v2(
    const at::Tensor &A_log,                 // fp32, H elements
    const at::Tensor &a,                     // bf16 [1,T,H,K]
    const at::Tensor &dt_bias,               // fp32 [H*D]
    const at::Tensor &q,                     // bf16 [1,T,H,D]
    const at::Tensor &k,                     // bf16 [1,T,H,D]
    const at::Tensor &v,                     // bf16 [1,T,H,D]
    const at::Tensor &b,                     // bf16 [1,T,H]
    at::Tensor &initial_state_source,        // fp32 [slots,H,K,V], K-contiguous
    const at::Tensor &initial_state_indices, // int32/int64 [numReq]
    const at::Tensor &cu_seqlens,            // int32/int64 [numReq+1]
    double lower_bound,
    double scale,
    const c10::optional<at::Tensor> &intermediate_state, // fp32 5-D, unsupported
    bool prefill,
    const c10::optional<at::Tensor> &track_state_indices, // int32/int64 [numReq]
    const c10::optional<at::Tensor> &track_state_lens)    // int32/int64 [numReq]
{
    const int64_t num_q_heads    = k.size(2);
    const int64_t key_dim        = k.size(3);
    const int64_t num_value_heads = v.size(2);
    const int64_t value_dim      = v.size(3);
    const bool    track_state    = track_state_indices.has_value();
    const bool    save_intermediate = intermediate_state.has_value();

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
                "kda_varlen_recurrent_v2 requires H == HV, got ", num_q_heads,
                " and ", num_value_heads);
    TORCH_CHECK(key_dim == config_D_v2 && value_dim == config_D_v2,
                "kda_varlen_recurrent_v2 requires K == V == ", config_D_v2,
                ", got ", key_dim, " and ", value_dim);
    TORCH_CHECK(A_log.scalar_type() == at::kFloat &&
                    A_log.numel() == num_value_heads,
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
                "state must be [slots,HV,K,V] matching the head and feature "
                "dims");
    TORCH_CHECK(initial_state_source.stride(2) == 1 &&
                    initial_state_source.stride(3) == config_D_v2,
                "state must be K-contiguous (stride(2)==1, stride(3)==",
                config_D_v2, "), got strides (",
                initial_state_source.stride(2), ", ",
                initial_state_source.stride(3), ")");
    if (save_intermediate) {
        const auto &mid = *intermediate_state;
        TORCH_CHECK(mid.dim() == 5 && mid.scalar_type() == at::kFloat,
                    "intermediate_state must be a 5-D fp32 "
                    "[slots,steps,HV,K,V] buffer");
        TORCH_CHECK(mid.size(2) == num_value_heads && mid.size(3) == key_dim &&
                        mid.size(4) == value_dim,
                    "intermediate_state must be [slots,steps,HV,K,V] matching "
                    "the head and feature dims");
        TORCH_CHECK(mid.stride(3) == 1 && mid.stride(4) == config_D_v2,
                    "intermediate_state must be K-contiguous (stride(3)==1, "
                    "stride(4)==", config_D_v2, "), got strides (",
                    mid.stride(3), ", ", mid.stride(4), ")");
        TORCH_CHECK(mid.device() == q.device(),
                    "intermediate_state must be on the same device as q");
        TORCH_CHECK(mid.size(0) >= initial_state_indices.numel(),
                    "intermediate_state needs a row per request, got ",
                    mid.size(0), " for ", initial_state_indices.numel(),
                    " requests");
        TORCH_CHECK(mid.size(1) * initial_state_indices.numel() >= q.size(1),
                    "intermediate_state has ", mid.size(1),
                    " step rows, too few for ", q.size(1), " tokens across ",
                    initial_state_indices.numel(), " requests");
    }
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

    auto as_int32 = [](const at::Tensor &t) {
        return t.scalar_type() == at::kInt ? t : t.to(at::kInt);
    };
    auto as_int32_opt = [&as_int32](const c10::optional<at::Tensor> &t) {
        return t.has_value() ? c10::optional<at::Tensor>(as_int32(*t))
                             : c10::nullopt;
    };
    const at::Tensor indices_i32 = as_int32(initial_state_indices);
    const at::Tensor starts_i32  = as_int32(cu_seqlens);
    const c10::optional<at::Tensor> track_indices_i32 =
        as_int32_opt(track_state_indices);
    const c10::optional<at::Tensor> track_lens_i32 =
        as_int32_opt(track_state_lens);

    const int64_t num_tokens = q.size(1);
    const int64_t num_req    = initial_state_indices.numel();

    auto output = at::zeros_like(v);

    auto gm = [](const at::Tensor &t) -> uint8_t * {
        return reinterpret_cast<uint8_t *>(const_cast<void *>(t.data_ptr()));
    };
    auto gm_opt = [&gm](const c10::optional<at::Tensor> &t) -> uint8_t * {
        return t.has_value() ? gm(*t) : nullptr;
    };

    at::Tensor qn, kn, gdecay, beta, sync;
    const bool needs_scratch =
        prefill && (num_value_heads == 4 || num_value_heads == 8);
    if (needs_scratch) {
        const auto f32 = v.options().dtype(at::kFloat);
        qn     = at::empty({1, num_tokens, num_value_heads, value_dim}, f32);
        kn     = at::empty({1, num_tokens, num_value_heads, value_dim}, f32);
        gdecay = at::empty({1, num_tokens, num_value_heads, value_dim}, f32);
        const int64_t t_pad =
            (num_tokens + kda_prefill_v2::kChunkTokens - 1) /
            kda_prefill_v2::kChunkTokens * kda_prefill_v2::kChunkTokens;
        beta = at::empty({num_value_heads, t_pad}, f32);
        sync = at::zeros(
            {static_cast<int64_t>(kda_prefill_v2::kSyncInt32Count)},
            v.options().dtype(at::kInt));
    }

    KdaV2LaunchArgs args{
        /*q=*/gm(q),
        /*k=*/gm(k),
        /*v=*/gm(v),
        /*a=*/gm(a),
        /*b=*/gm(b),
        /*A_log=*/gm(A_log),
        /*dt_bias=*/gm(dt_bias),
        /*state=*/gm(initial_state_source),
        /*indices=*/gm(indices_i32),
        /*starts=*/gm(starts_i32),
        /*out=*/gm(output),
        /*track_indices=*/gm_opt(track_indices_i32),
        /*track_lens=*/gm_opt(track_lens_i32),
        /*mid=*/gm_opt(intermediate_state),
        /*mid_stride_req=*/
        save_intermediate
            ? static_cast<uint32_t>(intermediate_state->stride(0)) : 0u,
        /*mid_stride_step=*/
        save_intermediate
            ? static_cast<uint32_t>(intermediate_state->stride(1)) : 0u,
        /*mid_stride_head=*/
        save_intermediate
            ? static_cast<uint32_t>(intermediate_state->stride(2)) : 0u,
        /*qn=*/needs_scratch ? gm(qn)     : nullptr,
        /*kn=*/needs_scratch ? gm(kn)     : nullptr,
        /*gdecay=*/needs_scratch ? gm(gdecay) : nullptr,
        /*beta=*/needs_scratch ? gm(beta)  : nullptr,
        /*sync=*/needs_scratch ? gm(sync)  : nullptr,
        /*scale=*/static_cast<float>(scale),
        /*lower_bound=*/static_cast<float>(lower_bound),
        /*num_req=*/static_cast<uint32_t>(num_req),
        /*num_tokens=*/static_cast<uint32_t>(num_tokens),
        /*track_state=*/track_state,
    };

    void *stream = c10_npu::getCurrentNPUStream().stream();

    // Single kernel handles all prefill and decode widths.
    if (prefill) {
        TORCH_CHECK(num_req == 1,
                    "kda_varlen_recurrent_v2 prefill expects one packed "
                    "sequence, got ", num_req);
        TORCH_CHECK(!save_intermediate,
                    "kda_varlen_recurrent_v2 prefill does not support "
                    "intermediate_state");
        TORCH_CHECK(num_value_heads == 4 || num_value_heads == 8,
                    "kda_varlen_recurrent_v2 prefill supports H=4 (TP16) "
                    "or H=8 (TP8), got H=", num_value_heads);
    } else {
        TORCH_CHECK(num_tokens >= num_req,
                    "kda_varlen_recurrent_v2 decode expects at least one token "
                    "per request, got T=", num_tokens, " and numReq=", num_req);
        TORCH_CHECK(!track_state,
                    "kda_varlen_recurrent_v2 tracking is prefill-only");
        TORCH_CHECK(num_value_heads == 16 || num_value_heads == 32 ||
                        num_value_heads == 64,
                    "kda_varlen_recurrent_v2 decode supports H=16 (TP4), "
                    "H=32 (TP2) or H=64 (TP1), got H=", num_value_heads);
    }

    LaunchKdaRecurrentV2(args, stream);
    return output;
}

// Development name. Change to "kda_varlen_recurrent" on cutover (drop v1 too).
TORCH_LIBRARY_FRAGMENT(sgl_glm53_ascend, m) {
    m.def("kda_varlen_recurrent_v2(*, Tensor A_log, Tensor a, Tensor dt_bias, "
          "Tensor q, Tensor k, Tensor v, Tensor b, "
          "Tensor(a!) initial_state_source, Tensor initial_state_indices, "
          "Tensor cu_seqlens, float lower_bound, float scale, "
          "Tensor(b!)? intermediate_state=None, bool prefill=False, "
          "Tensor? track_state_indices=None, "
          "Tensor? track_state_lens=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(sgl_glm53_ascend, PrivateUse1, m) {
    m.impl("kda_varlen_recurrent_v2", &kda_varlen_recurrent_v2);
}
