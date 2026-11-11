// GLM-5.3 chunked KDA v2 — seam between the host op and the kernel.
//
//   kda_varlen_chunk_v2.cpp          host: validation, workspace, registration
//   kda_varlen_chunk_v2_kernel.cpp   the MIX kernel (1 AIC + 2 AIV per block)
//
// Registers as `kda_varlen_chunk_v2` during development. Cutover to
// `kda_varlen_chunk` (replacing v1) is two changes: rename the
// TORCH_LIBRARY_FRAGMENT string here and drop v1 from CMakeLists. The Python
// wrapper in ops/kda.py needs no change — it already calls
// torch.ops.sgl_glm53_ascend.kda_varlen_chunk.
//
// Included by both TUs; must stay free of torch headers.

#pragma once

#include <cstdint>
#include "kernel_operator.h"

constexpr int64_t chunk_v2_config_D = 128;

namespace kda_chunk_v2 {

// Tokens per chunk. Compile-time: the UB map, workspace stride and rebase
// block count all derive from it.
constexpr uint32_t kChunkTokens = 64;

// Rows per gate-rebase block. Correctness bound on fp32 exp overflow (+88.7).
constexpr uint32_t kRebaseRows   = 32;
constexpr uint32_t kRebaseBlocks = kChunkTokens / kRebaseRows;
static_assert(kChunkTokens % kRebaseRows == 0,
              "the chunk must split into whole rebase blocks");

#ifndef KDA_CHUNK_V2_INVERSE_SQUARING
#define KDA_CHUNK_V2_INVERSE_SQUARING 0
#endif

constexpr uint32_t kInvBlock  = 32;
constexpr uint32_t kInvBlocks = kChunkTokens / kInvBlock;
static_assert(kInvBlocks == 2,
              "the off-diagonal merge is written for exactly two blocks");

constexpr uint32_t kInverseSteps = 6;
static_assert((1u << kInverseSteps) == kChunkTokens,
              "the squaring chain must reach the nilpotency degree");

// Mode-2 (AIC <-> both AIV) flag IDs.
constexpr uint16_t kFlagAivToAic    = 0;
constexpr uint16_t kFlagAicToAiv    = 1;
constexpr uint16_t kFlagAivToAicAux = 2;
// Mode-1 (AIV <-> AIV) ID.
constexpr uint16_t kFlagAivToAiv    = 3;

// Per-block GM workspace, fp32 elements.
constexpr uint32_t kD     = static_cast<uint32_t>(chunk_v2_config_D);
constexpr uint32_t kBT    = kChunkTokens;

constexpr uint32_t kTileBT_D  = kBT * kD;
constexpr uint32_t kTileBT_BT = kBT * kBT;
constexpr uint32_t kTileD_D   = kD  * kD;

// Workspace slot offsets — identical layout to v1 so the same pool size is
// reused. Replace this enum when the v2 algorithm changes the layout.
enum : uint32_t {
    kWsRowK      = 0,
    kWsRowQ      = kWsRowK      + kTileBT_D,
    kWsColK      = kWsRowQ      + kTileBT_D,
    kWsA         = kWsColK      + kRebaseBlocks * kTileBT_D,
    kWsAqk       = kWsA         + kTileBT_BT,
    kWsX         = kWsAqk       + kTileBT_BT,
    kWsDpow      = kWsX         + kTileBT_BT,
    kWsFactor    = kWsDpow      + kTileBT_BT,
    kWsScratch   = kWsFactor    + kTileBT_BT,
    kWsInvB10    = kWsDpow,
    kWsInvTmp    = kWsFactor,
    kWsInvB11    = kWsScratch,
    kWsKgBeta    = kWsScratch   + kTileBT_BT,
    kWsVBeta     = kWsKgBeta    + kTileBT_D,
    kWsQg        = kWsVBeta     + kTileBT_D,
    kWsKDecay    = kWsQg        + kTileBT_D,
    kWsW         = kWsKDecay    + kTileBT_D,
    kWsU         = kWsW         + kTileBT_D,
    kWsVNew      = kWsU         + kTileBT_D,
    kWsOut       = kWsVNew      + kTileBT_D,
    kWsTmpBT_D   = kWsOut       + kTileBT_D,
    kWsState     = kWsTmpBT_D   + kTileBT_D,
    kWsStateUpd  = kWsState     + kTileD_D,
    kWsStateSave = kWsStateUpd  + kTileD_D,
    kWsPerBlock  = kWsStateSave + kTileD_D,
};

constexpr uint32_t kWorkspaceFloatsPerBlock = kWsPerBlock;

}  // namespace kda_chunk_v2

struct KdaChunkV2LaunchArgs {
    GM_ADDR q;             // bf16 [1,T,H,K]
    GM_ADDR k;             // bf16 [1,T,H,K]
    GM_ADDR v;             // bf16 [1,T,H,V]
    GM_ADDR a;             // bf16 [1,T,H,K]
    GM_ADDR b;             // bf16 [1,T,H]
    GM_ADDR A_log;         // fp32 [H]
    GM_ADDR dt_bias;       // fp32 [H*K]
    GM_ADDR state;         // fp32 [slots,H,K,V], K-contiguous
    GM_ADDR indices;       // int32 [numReq]
    GM_ADDR starts;        // int32 [numReq+1]
    GM_ADDR out;           // bf16 [1,T,H,V], allocated by the caller
    GM_ADDR track_indices; // int32 [numReq], null unless tracking
    GM_ADDR track_lens;    // int32 [numReq], null unless tracking
    GM_ADDR workspace;     // fp32, kWorkspaceFloatsPerBlock per block

    float    scale;
    float    lower_bound;
    uint32_t num_req;
    uint32_t num_heads;
    bool     track_state;
};

void LaunchKdaVarlenChunkV2(const KdaChunkV2LaunchArgs &args,
                            uint32_t num_heads, void *stream);
