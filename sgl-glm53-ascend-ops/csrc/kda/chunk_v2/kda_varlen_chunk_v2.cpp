// GLM-5.3 chunked KDA v2 — host op: validation, workspace, dispatch,
// registration.
//
// Registered as `kda_varlen_chunk_v2` during development.
// Cutover: change the TORCH_LIBRARY_FRAGMENT string to `kda_varlen_chunk` and
// remove v1 from CMakeLists. The Python wrapper needs no change.

#include "kda_varlen_chunk_v2.h"

#include <utility>

#include <acl/acl.h>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

at::Tensor kda_varlen_chunk_v2(
    const at::Tensor &A_log,                 // fp32 [HV]
    const at::Tensor &a,                     // bf16 [1,T,HV,K]
    const at::Tensor &dt_bias,               // fp32 [HV,K] or [HV*K]
    const at::Tensor &q,                     // bf16 [1,T,H,K]
    const at::Tensor &k,                     // bf16 [1,T,H,K]
    const at::Tensor &v,                     // bf16 [1,T,HV,V]
    const at::Tensor &b,                     // bf16 [1,T,HV]
    at::Tensor &initial_state_source,        // fp32 [slots,HV,K,V], K-contig
    const at::Tensor &initial_state_indices, // int32/int64 [numReq]
    const at::Tensor &cu_seqlens,            // int32/int64 [numReq+1]
    double lower_bound,
    double scale,
    bool prefill,
    const c10::optional<at::Tensor> &track_state_indices, // int32/int64
    const c10::optional<at::Tensor> &track_state_lens)    // int32/int64
{
    const int64_t num_q_heads    = k.size(2);
    const int64_t key_dim        = k.size(3);
    const int64_t num_value_heads = v.size(2);
    const int64_t value_dim      = v.size(3);
    const bool    track_state    = track_state_indices.has_value();

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
                "kda_varlen_chunk_v2 requires H == HV, got ", num_q_heads,
                " and ", num_value_heads);
    TORCH_CHECK(key_dim == chunk_v2_config_D && value_dim == chunk_v2_config_D,
                "kda_varlen_chunk_v2 requires K == V == ", chunk_v2_config_D,
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
                "state must be [slots,HV,K,V] matching the head and feature dims");
    TORCH_CHECK(initial_state_source.stride(2) == 1 &&
                    initial_state_source.stride(3) == chunk_v2_config_D,
                "state must be K-contiguous (stride(2)==1, stride(3)==",
                chunk_v2_config_D, "), got strides (",
                initial_state_source.stride(2), ", ",
                initial_state_source.stride(3), ")");

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
    if (track_state) {
        TORCH_CHECK(track_state_lens.has_value(),
                    "track_state_indices requires track_state_lens");
        for (const auto &named :
             {std::pair<const char *, const at::Tensor &>{
                  "track_state_indices", *track_state_indices},
              {"track_state_lens", *track_state_lens}}) {
            TORCH_CHECK(named.second.scalar_type() == at::kInt ||
                            named.second.scalar_type() == at::kLong,
                        named.first, " must be int32 or int64");
            TORCH_CHECK(named.second.numel() == initial_state_indices.numel(),
                        named.first, " must have one entry per request");
            TORCH_CHECK(named.second.device() == q.device(),
                        named.first, " must be on the same device as q");
        }
    }

    // int64 indices read as int32 are silently all-zero on this device.
    auto as_int32 = [](const at::Tensor &t) {
        return t.scalar_type() == at::kInt ? t.contiguous()
                                           : t.to(at::kInt).contiguous();
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

    const int64_t num_req = initial_state_indices.numel();
    TORCH_CHECK(cu_seqlens.numel() == num_req + 1,
                "cu_seqlens must have numReq+1 entries, got ",
                cu_seqlens.numel(), " for ", num_req, " requests");

    const at::Tensor starts_cpu = starts_i32.to(at::kCPU);
    const int32_t *starts = starts_cpu.data_ptr<int32_t>();
    for (int64_t i = 0; i < num_req; ++i) {
        const int32_t len = starts[i + 1] - starts[i];
        TORCH_CHECK(len >= 0, "cu_seqlens must be non-decreasing, got ",
                    starts[i], " then ", starts[i + 1]);
        TORCH_CHECK(len % static_cast<int32_t>(kda_chunk_v2::kChunkTokens) == 0,
                    "kda_varlen_chunk_v2 requires every sequence length to be a "
                    "multiple of ", kda_chunk_v2::kChunkTokens, ", got ", len,
                    " for request ", i);
    }

    auto output = at::empty_like(v);

    auto gm = [](const at::Tensor &t) -> uint8_t * {
        return reinterpret_cast<uint8_t *>(const_cast<void *>(t.data_ptr()));
    };
    auto gm_opt = [&gm](const c10::optional<at::Tensor> &t) -> uint8_t * {
        return t.has_value() ? gm(*t) : nullptr;
    };

    const int64_t num_blocks = num_req * num_value_heads;
    auto workspace = at::zeros(
        {num_blocks *
         static_cast<int64_t>(kda_chunk_v2::kWorkspaceFloatsPerBlock)},
        v.options().dtype(at::kFloat));

    KdaChunkV2LaunchArgs args{
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
        /*workspace=*/gm(workspace),
        /*scale=*/static_cast<float>(scale),
        /*lower_bound=*/static_cast<float>(lower_bound),
        /*num_req=*/static_cast<uint32_t>(num_req),
        /*num_heads=*/static_cast<uint32_t>(num_value_heads),
        /*track_state=*/track_state,
    };

    void *stream = c10_npu::getCurrentNPUStream().stream();
    LaunchKdaVarlenChunkV2(args, static_cast<uint32_t>(num_value_heads), stream);
    return output;
}

// Development name. Change to "kda_varlen_chunk" on cutover (drop v1 too).
TORCH_LIBRARY_FRAGMENT(sgl_glm53_ascend, m) {
    m.def("kda_varlen_chunk_v2(*, Tensor A_log, Tensor a, Tensor dt_bias, "
          "Tensor q, Tensor k, Tensor v, Tensor b, "
          "Tensor(a!) initial_state_source, Tensor initial_state_indices, "
          "Tensor cu_seqlens, float lower_bound, float scale, "
          "bool prefill=False, Tensor? track_state_indices=None, "
          "Tensor? track_state_lens=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(sgl_glm53_ascend, PrivateUse1, m) {
    m.impl("kda_varlen_chunk_v2", &kda_varlen_chunk_v2);
}
