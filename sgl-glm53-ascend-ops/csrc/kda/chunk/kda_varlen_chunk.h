// GLM-5.3 chunked KDA — seam between the host op and the kernel.
//
//   kda_varlen_chunk.cpp          host: validation, workspace, registration
//   kda_varlen_chunk_kernel.cpp   the MIX kernel (1 AIC + 2 AIV per block)
//   kda_varlen_chunk.md           design notes
//
// Included by the host TU as well, so it must stay free of torch headers.

#pragma once

#include <cstdint>
#include "kernel_operator.h"

// The only supported feature width, shared by the device tiling and the host
// validation so there is one definition of it.
constexpr int64_t chunk_config_D = 128;

namespace kda_chunk {

// Tokens per chunk. Compile-time: the UB map, the workspace stride and the
// rebase block count all derive from it.
constexpr uint32_t kChunkTokens = 64;

// Rows per gate-rebase block. NOT a tuning knob — a correctness bound on the
// factored A/Aqk's positive exponent, which fp32 exp overflows at +88.7.
// Measured margins and why the anchor is the midpoint: kda_varlen_chunk.md.
constexpr uint32_t kRebaseRows = 32;
constexpr uint32_t kRebaseBlocks = kChunkTokens / kRebaseRows;
static_assert(kChunkTokens % kRebaseRows == 0,
              "the chunk must split into whole rebase blocks");

// ── Which triangular inverse ─────────────────────────────────────────
//
//   0 (default)  FLA-style blocked forward substitution, 104.7 us/chunk
//   1            repeated squaring, the original, 110.8
//
// Both are validated. Select with CMake -DKDA_CHUNK_INVERSE=squaring; the md
// has the breakdown.
#ifndef KDA_CHUNK_INVERSE_SQUARING
#define KDA_CHUNK_INVERSE_SQUARING 0
#endif

// Side length of a substitution-solved diagonal block. 32 is FLA's
// `SUB_CHUNK_SIZE` and splits the chunk two ways, which is both the fewest a
// substitution can use and what the AIV pair solves concurrently.
constexpr uint32_t kInvBlock = 32;
constexpr uint32_t kInvBlocks = kChunkTokens / kInvBlock;
static_assert(kInvBlocks == 2,
              "the off-diagonal merge is written for exactly two blocks");

// Squaring steps for the triangular inverse: (I - Araw)^-1 = prod_s
// (I + Araw^(2^s)) because Araw is strictly lower triangular, hence nilpotent
// with Araw^BT = 0. 2^kInverseSteps == kChunkTokens.
constexpr uint32_t kInverseSteps = 6;
static_assert((1u << kInverseSteps) == kChunkTokens,
              "the squaring chain must reach the nilpotency degree");

// Mode-2 (AIC <-> both AIV, within one block) flag IDs, range 0-15.
// `kFlagAivToAicAux` is a second forward ID so the inversion's handoff needs
// no reply; see the md.
constexpr uint16_t kFlagAivToAic = 0;
constexpr uint16_t kFlagAicToAiv = 1;
constexpr uint16_t kFlagAivToAicAux = 2;

// Mode-1 (AIV <-> AIV) ID. The pair meets once per chunk, after the worker
// publishes B11; nothing needs ordering on the way back.
constexpr uint16_t kFlagAivToAiv = 3;

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
    // Separate buffers: the chain squares Araw^(2^s) but multiplies by
    // (I + Araw^(2^s)), and squaring the factor is not the same thing.
    kWsDpow = kWsX + kTileBT_BT,                  // [64, 64]  Araw^(2^s)
    kWsFactor = kWsDpow + kTileBT_BT,             // [64, 64]  I + Araw^(2^s)
    kWsScratch = kWsFactor + kTileBT_BT,          // [64, 64]  matmul landing
    // The substitution path's operands, aliased onto the squaring path's
    // scratch: only one path is compiled in, so they are never both live.
    kWsInvB10 = kWsDpow,                          // [32, 32] Araw's (1,0)
    kWsInvTmp = kWsFactor,                        // [32, 32] lands C11 B10
    kWsInvB11 = kWsScratch,                       // [32, 32] for the helper
    kWsKgBeta = kWsScratch + kTileBT_BT,          // [64, 128] beta*(e^gc * kn)
    kWsVBeta = kWsKgBeta + kTileBT_D,             // [64, 128] beta * v
    kWsQg = kWsVBeta + kTileBT_D,                 // [64, 128] qn * e^gc
    kWsKDecay = kWsQg + kTileBT_D,                // [64, 128] e^(gl-gc) * kn
    kWsW = kWsKDecay + kTileBT_D,                 // [64, 128]
    kWsU = kWsW + kTileBT_D,                      // [64, 128]
    kWsVNew = kWsU + kTileBT_D,                   // [64, 128]
    kWsOut = kWsVNew + kTileBT_D,                 // [64, 128] o accumulator
    kWsTmpBT_D = kWsOut + kTileBT_D,              // [64, 128] matmul landing
    // S and its increment, both transposed: [V, K] row-major, which is what
    // a verbatim copy of a K-contiguous pool slot gives. See the md.
    kWsState = kWsTmpBT_D + kTileBT_D,            // [128, 128] S^T
    kWsStateUpd = kWsState + kTileD_D,            // [128, 128] vnew^T @ kdecay
    // The chunk's start state, kept only when tracking: a tracked prefix
    // overwrites S and the chunk's own update must start from this.
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
