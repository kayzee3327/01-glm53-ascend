// GLM-5.3 chunked KDA — seam between the host op and the kernel.
//
//   kda_varlen_chunk.cpp          host: validation, workspace, registration
//   kda_varlen_chunk_kernel.cpp   the MIX kernel (one AIC + two AIV per block)
//
// Included by both the host TU (which also includes torch) and the device TU,
// so it must stay free of torch headers.

#pragma once

#include <cstdint>
#include "kernel_operator.h"

// The only supported feature width, shared by the device tiling and the host
// validation so there is one definition of it.
constexpr int64_t chunk_config_D = 128;

namespace kda_chunk {

// Tokens per chunk. A compile-time constant rather than an argument: the UB
// map, the workspace stride and the rebase block count are all derived from
// it. 128 is the other value the algebra admits; see the kernel header.
constexpr uint32_t kChunkTokens = 64;

// Rows per gate-rebase block. NOT a tuning knob — a correctness bound. The
// factored A/Aqk a Cube matmul needs splits one bounded exponent
// exp(gc[r]-gc[c]) into exp(gc[r]-m) * exp(m-gc[c]), and the cumulative gate
// spans ~177 over 64 tokens on deployed inputs. Measured max exponent by
// rebase granularity: whole chunk NaN, 32 rows +88.4 (fp32 overflows at
// +88.7), 16 rows +43. Sixteen is the margin.
constexpr uint32_t kRebaseRows = 16;
constexpr uint32_t kRebaseBlocks = kChunkTokens / kRebaseRows;
static_assert(kChunkTokens % kRebaseRows == 0,
              "the chunk must split into whole rebase blocks");

// Squaring steps for the triangular inverse: (I - Araw)^-1 = prod_s
// (I + Araw^(2^s)) because Araw is strictly lower triangular, hence nilpotent
// with Araw^BT = 0. 2^kInverseSteps == kChunkTokens.
constexpr uint32_t kInverseSteps = 6;
static_assert((1u << kInverseSteps) == kChunkTokens,
              "the squaring chain must reach the nilpotency degree");

// Cross-core flag IDs for the mode-2 (AIC <-> both AIV, within one block)
// handshake. Range is 0-15; two are enough because the two directions never
// have more than one message outstanding.
constexpr uint16_t kFlagAivToAic = 0;
constexpr uint16_t kFlagAicToAiv = 1;

// Per-block GM workspace, in fp32 elements. One block owns one slice, so no
// two blocks share a cache line.
constexpr uint32_t kD = static_cast<uint32_t>(chunk_config_D);
constexpr uint32_t kBT = kChunkTokens;

constexpr uint32_t kTileBT_D = kBT * kD;       // [64, 128]
constexpr uint32_t kTileBT_BT = kBT * kBT;     // [64, 64]
constexpr uint32_t kTileD_D = kD * kD;         // [128, 128]

// Offsets into one block's slice.
enum : uint32_t {
    kWsRowK = 0,                                  // [64, 128] k rebased by row
    kWsRowQ = kWsRowK + kTileBT_D,                // [64, 128] q rebased by row
    kWsColK = kWsRowQ + kTileBT_D,                // [4][64, 128] per block
    kWsA = kWsColK + kRebaseBlocks * kTileBT_D,   // [64, 64]  Araw
    kWsAqk = kWsA + kTileBT_BT,                   // [64, 64]
    kWsX = kWsAqk + kTileBT_BT,                   // [64, 64]  running inverse
    // The nilpotent power and the factor built from it are separate buffers:
    // the chain squares Araw^(2^s) but multiplies by (I + Araw^(2^s)), and
    // squaring the factor instead would give I + 2Araw^(2^s) + Araw^(2^(s+1)).
    kWsDpow = kWsX + kTileBT_BT,                  // [64, 64]  Araw^(2^s)
    kWsFactor = kWsDpow + kTileBT_BT,             // [64, 64]  I + Araw^(2^s)
    kWsScratch = kWsFactor + kTileBT_BT,          // [64, 64]  matmul landing
    kWsKgBeta = kWsScratch + kTileBT_BT,          // [64, 128] beta*(e^gc * kn)
    kWsVBeta = kWsKgBeta + kTileBT_D,             // [64, 128] beta * v
    kWsQg = kWsVBeta + kTileBT_D,                 // [64, 128] qn * e^gc
    kWsKDecay = kWsQg + kTileBT_D,                // [64, 128] e^(gl-gc) * kn
    kWsW = kWsKDecay + kTileBT_D,                 // [64, 128]
    kWsU = kWsW + kTileBT_D,                      // [64, 128]
    kWsVNew = kWsU + kTileBT_D,                   // [64, 128]
    kWsOut = kWsVNew + kTileBT_D,                 // [64, 128] o accumulator
    kWsTmpBT_D = kWsOut + kTileBT_D,              // [64, 128] matmul landing
    // S and its increment, both transposed: [V, K] row-major, not [K, V].
    // The state pool is K-contiguous ([slots,H,K,V] with stride(2)==1), so a
    // verbatim copy of a slot is already [V][K] and the alternative is a
    // 128x128 transpose on the AIV. The three matmuls that touch S take it as
    // a B_TRANS operand instead, and the increment is computed as
    // vnew^T @ kdecay.
    kWsState = kWsTmpBT_D + kTileBT_D,            // [128, 128] S^T
    kWsStateUpd = kWsState + kTileD_D,            // [128, 128] vnew^T @ kdecay
    // The chunk's start state, kept only when tracking: a tracked prefix
    // state overwrites S, and the chunk's own update has to run from the
    // value S had at the chunk's start.
    kWsStateSave = kWsStateUpd + kTileD_D,        // [128, 128] S^T
    kWsPerBlock = kWsStateSave + kTileD_D,
};

// ~1.1 MB per block.
constexpr uint32_t kWorkspaceFloatsPerBlock = kWsPerBlock;

}  // namespace kda_chunk

struct KdaChunkLaunchArgs {
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

    float scale;
    float lower_bound;
    uint32_t num_req;
    uint32_t num_heads;
    bool track_state;
};

void LaunchKdaVarlenChunk(const KdaChunkLaunchArgs &args, uint32_t num_heads,
                          void *stream);
