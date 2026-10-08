#include "kernel_operator.h"
#include "utils.h"

// Static Tensor event IDs are manually managed.
// Avoid IDs 6/7.
constexpr int32_t EVENT_MTE2_V = 0;
constexpr int32_t EVENT_MTE2_S = 1;
constexpr int32_t EVENT_S_V = 2;
constexpr int32_t EVENT_V_MTE2 = 3;
constexpr int32_t EVENT_MTE3_V = 4;
constexpr int32_t EVENT_V_MTE3 = 5;

extern "C" __global__ __vector__
void prepared_recurrent_kernel(
    GM_ADDR QN, GM_ADDR KN,         // fp32
    GM_ADDR V,                      // bf16
    GM_ADDR GDECAY, GM_ADDR BETA,   // fp32
    GM_ADDR OUT,                    // bf16
    GM_ADDR STATE,                  // fp32
    GM_ADDR INDICES,                // int32/int64
    GM_ADDR STARTS,                 // int32
    GM_ADDR TRACK_INDICES,          // int32/int64
    GM_ADDR TRACK_LENS              // int32/int64
) {
    AscendC::InitSocState();

    constexpr uint32_t H  = 4;
    constexpr uint32_t D  = 128;
    constexpr uint32_t BV = 16;
    constexpr uint32_t NUM_V_TILES = (D + BV - 1) / BV;

    const uint32_t blockIdx = AscendC::GetBlockIdx();
    const uint32_t iv = blockIdx % NUM_V_TILES;
    const uint32_t head = blockIdx / NUM_V_TILES;

    // assume launching block num is (NUM_V_TILES * H)
    // if (head >= headCount) {
    //     return;
    // }

    AscendC::GlobalTensor<float> qnGm;
    AscendC::GlobalTensor<float> knGm;
    AscendC::GlobalTensor<bfloat16_t> vGm;
    AscendC::GlobalTensor<float> decayGm;
    AscendC::GlobalTensor<float> betaGm;
    AscendC::GlobalTensor<bfloat16_t> outGm;
    AscendC::GlobalTensor<float> stateGm;

    AscendC::GlobalTensor<int32_t> indicesGm;
    AscendC::GlobalTensor<int32_t> startsGm;

    qnGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(QN));
    knGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(KN));
    vGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(V));
    decayGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(GDECAY));
    betaGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(BETA));
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(OUT));
    stateGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(STATE));
    indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(INDICES));
    startsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(STARTS));

    // ============================================================
    // Per token UB layout
    //
    // state       BV * D * fp32     = 16 * 128 * 4 = 8192 B
    // temp        BV * D * fp32     = 8192 B
    // q           128 * fp32        = 512 B
    // k           128 * fp32        = 512 B
    // decay       128 * fp32        = 512 B
    // v           16 * bf16         = 32 B
    // out         16 * bf16         = 32 B
    // tmpReduce   64 * fp32         = 256 B
    // partial     16 * fp32          = 64 B
    //
    // Total ~18 KB, far below UB capacity.
    // ============================================================

    constexpr uint32_t HALF_D = D / 2;

    constexpr uint32_t STATE_ADDR = 0;
    constexpr uint32_t STATE_SIZE = BV * D * sizeof(float);
    AscendC::LocalTensor<float> stateLocal(AscendC::TPosition::VECCALC, STATE_ADDR, BV * D);
    constexpr uint32_t TMPBVD_ADDR = Align32(STATE_ADDR + STATE_SIZE);
    constexpr uint32_t TMPBVD_SIZE = STATE_SIZE;
    AscendC::LocalTensor<float> tmpBVDLocal(AscendC::TPosition::VECCALC, TMPBVD_ADDR, BV * D);
    constexpr uint32_t Q_ADDR = Align32(TMPBVD_ADDR + TMPBVD_SIZE);
    constexpr uint32_t Q_SIZE = D * sizeof(float);
    AscendC::LocalTensor<float> qLocal(AscendC::TPosition::VECCALC, Q_ADDR, D);
    constexpr uint32_t K_ADDR = Align32(Q_ADDR + Q_SIZE);
    constexpr uint32_t K_SIZE = D * sizeof(float);
    AscendC::LocalTensor<float> kLocal(AscendC::TPosition::VECCALC, K_ADDR, D);
    constexpr uint32_t DECAY_ADDR = Align32(K_ADDR + K_SIZE);
    constexpr uint32_t DECAY_SIZE = D * sizeof(float);
    AscendC::LocalTensor<float> decayLocal(AscendC::TPosition::VECCALC, DECAY_ADDR, D);
    constexpr uint32_t V_ADDR = Align32(DECAY_ADDR + DECAY_SIZE);
    constexpr uint32_t V_SIZE = BV * sizeof(bfloat16_t);
    AscendC::LocalTensor<bfloat16_t> vLocal(AscendC::TPosition::VECCALC, V_ADDR, BV);
    constexpr uint32_t OUT_ADDR = Align32(V_ADDR + V_SIZE);
    constexpr uint32_t OUT_SIZE = BV * sizeof(bfloat16_t);
    AscendC::LocalTensor<bfloat16_t> outLocal(AscendC::TPosition::VECCALC, OUT_ADDR, BV);
    constexpr uint32_t TMPBV_ADDR = Align32(OUT_ADDR + OUT_SIZE);
    constexpr uint32_t TMPBV_SIZE = BV * sizeof(float);
    AscendC::LocalTensor<float> tmpBVLocal(AscendC::TPosition::VECCALC, TMPBV_ADDR, BV);
    constexpr uint32_t REDUCE_ADDR = Align32(TMPBV_ADDR + TMPBV_SIZE);
    constexpr uint32_t REDUCE_SIZE = BV * sizeof(float);
    AscendC::LocalTensor<float> reduceLocal(AscendC::TPosition::VECCALC, REDUCE_ADDR, BV);
    constexpr uint32_t TMPHD_ADDR = Align32(REDUCE_ADDR + REDUCE_SIZE);
    constexpr uint32_t TMPHD_SIZE = HALF_D * sizeof(float);
    AscendC::LocalTensor<float> tmpHDLocal(AscendC::TPosition::VECCALC, TMPHD_ADDR, HALF_D);
    constexpr uint32_t TMPBRCB_ADDR = Align32(TMPHD_ADDR + TMPHD_SIZE);
    constexpr uint32_t TMPBRCB_SIZE = 128 * sizeof(float);
    AscendC::LocalTensor<float> tmpBrcbLocal(AscendC::TPosition::VECCALC, TMPBRCB_ADDR, 128);



    // bos = tl.load(STARTS + req).to(tl.int64)
    // eos = tl.load(STARTS + req + 1).to(tl.int64)
    // slot = tl.load(INDICES + req)
    const int32_t bos = startsGm.GetValue(0);
    const int32_t eos = startsGm.GetValue(1);
    const int32_t slot = indicesGm.GetValue(0);

    // stateBase points at: state[slot, head, iv*BV, 0]
    const uint64_t stateBase =
        (static_cast<uint64_t>(slot) * H + head) * D * D 
        + static_cast<uint64_t>(iv*BV) * D;

    if (slot > 0) {
        AscendC::DataCopy(stateLocal, stateGm[stateBase], BV * D);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V);
    } else {
        AscendC::Duplicate(stateLocal, 0.0f, BV * D);
        AscendC::PipeBarrier<PIPE_V>();
    }

    for (int32_t token = bos; token < eos; ++token) {

        const uint64_t row = static_cast<uint64_t>(token) * H + head;

        // ----------------------------------------------------
        // MTE2:
        //
        // QN     [128] fp32 512B
        // KN     [128] fp32 512B
        // DECAY  [128] fp32 512B
        // V      [16]  bf16 32B
        //
        // ----------------------------------------------------

        // q = tl.load(QN + row * D + ks)
        // k = tl.load(KN + row * D + ks)
        // v = tl.load(V + row * D + vs).to(tl.float32)
        // decay = tl.load(GDECAY + row * D + ks)
        AscendC::DataCopy(qLocal, qnGm[row * D], D);
        AscendC::DataCopy(kLocal, knGm[row * D], D);
        AscendC::DataCopy(decayLocal, decayGm[row * D], D);
        AscendC::DataCopy(vLocal, vGm[row * D + iv * BV], BV);

        // Vector needs Q/K/decay.
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V);

        const float betaValue = betaGm.GetValue(row);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_S_V);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_S_V);

        // state *= decay[None, :]
        #pragma unroll
        for (uint32_t j = 0; j < BV; ++j) {
            auto stateRow = stateLocal[j * D];
            AscendC::Mul(stateRow, stateRow, decayLocal, D);
        }
        AscendC::PipeBarrier<PIPE_V>();
        // v -= tl.sum(state * k[None, :], axis=1)
        #pragma unroll
        for (uint32_t j = 0; j < BV; ++j) {
            auto stateRow = stateLocal[j * D];
            auto tempRow = tmpBVDLocal[j * D];
            AscendC::Mul(tempRow, stateRow, kLocal, D);
        }
        AscendC::PipeBarrier<PIPE_V>();
        ReduceRows128_AS<BV>(tmpBVDLocal, tmpHDLocal, reduceLocal);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(tmpBVLocal, vLocal, AscendC::RoundMode::CAST_NONE, BV);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(tmpBVLocal, tmpBVLocal, reduceLocal, BV);
        AscendC::PipeBarrier<PIPE_V>();

        // v *= beta
        // state += k[None, :] * v[:, None]
        AscendC::Muls(tmpBVLocal, tmpBVLocal, betaValue, BV);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Brcb(tmpBrcbLocal, tmpBVLocal, 2, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BinaryRepeatParams params = {
            16,              // dstBlkStride = 16
            0,               // src0BlkStride: reuse same k block
            1,               // src1BlkStride: next v broadcast block
            1,               // dstRepStride: next column block
            1,               // src0RepStride: next k block
            0                // src1RepStride: reuse v0..v7
        };
        // rows 0..7
        AscendC::MulAddDst(stateLocal, kLocal, tmpBrcbLocal, 64, 16, params);
        // rows 8..15
        AscendC::MulAddDst(stateLocal[8 * D], kLocal, tmpBrcbLocal[8 * 8], 64, 16, params);
        AscendC::PipeBarrier<PIPE_V>();
        
        // out = tl.sum(state * q[None, :], axis=1)
        #pragma unroll
        for (uint32_t j = 0; j < BV; ++j) {
            auto stateRow = stateLocal[j * D];
            auto tempRow = tmpBVDLocal[j * D];
            AscendC::Mul(tempRow, stateRow, qLocal, D);
        }
        AscendC::PipeBarrier<PIPE_V>();
        ReduceRows128_AS<BV>(tmpBVDLocal, tmpHDLocal, reduceLocal);
        AscendC::PipeBarrier<PIPE_V>();

        // tl.store(OUT + row * D + vs, out.to(OUT.dtype.element_ty))
        AscendC::Cast(outLocal, reduceLocal, AscendC::RoundMode::CAST_NONE, BV);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_MTE3);

        AscendC::DataCopy(outGm[row * D + iv * BV], outLocal, BV);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_MTE3_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_MTE3_V);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_V_MTE2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_V_MTE2);
    }

    // ========================================================
    // Store final recurrent state
    // ========================================================

    if (slot > 0) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_MTE3);
        AscendC::DataCopy(stateGm[stateBase], stateLocal, BV * D);
    }
}

// ── Host-side: torch custom op registration ────────────────────────────

#ifndef __CCE_KT_TEST__

#include <acl/acl.h>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

void prepared_recurrent(
    const at::Tensor &qn,
    const at::Tensor &kn,
    const at::Tensor &v,
    const at::Tensor &gdecay,
    const at::Tensor &beta,
    at::Tensor &out,
    at::Tensor &state,
    const at::Tensor &indices,
    const at::Tensor &starts)
{
    TORCH_CHECK(qn.is_privateuseone(), "qn must be an NPU tensor");
    TORCH_CHECK(kn.is_privateuseone(), "kn must be an NPU tensor");
    TORCH_CHECK(v.is_privateuseone(),  "v must be an NPU tensor");

    constexpr int32_t H  = 4;
    constexpr int32_t D  = 128;
    constexpr int32_t BV = 16;
    constexpr int32_t NUM_V_TILES = (D + BV - 1) / BV;
    constexpr int32_t blockDim = NUM_V_TILES * H;

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    prepared_recurrent_kernel<<<blockDim, nullptr, stream>>>(
        reinterpret_cast<uint8_t *>(const_cast<void *>(qn.data_ptr())),
        reinterpret_cast<uint8_t *>(const_cast<void *>(kn.data_ptr())),
        reinterpret_cast<uint8_t *>(const_cast<void *>(v.data_ptr())),
        reinterpret_cast<uint8_t *>(const_cast<void *>(gdecay.data_ptr())),
        reinterpret_cast<uint8_t *>(const_cast<void *>(beta.data_ptr())),
        reinterpret_cast<uint8_t *>(out.data_ptr()),
        reinterpret_cast<uint8_t *>(state.data_ptr()),
        reinterpret_cast<uint8_t *>(const_cast<void *>(indices.data_ptr())),
        reinterpret_cast<uint8_t *>(const_cast<void *>(starts.data_ptr())),
        nullptr,
        nullptr);
}

TORCH_LIBRARY_FRAGMENT(sgl_glm53_ascend, m) {
    m.def("prepared_recurrent(Tensor qn, Tensor kn, Tensor v, "
          "Tensor gdecay, Tensor beta, Tensor(a!) out, Tensor(b!) state, "
          "Tensor indices, Tensor starts) -> ()");
}

TORCH_LIBRARY_IMPL(sgl_glm53_ascend, PrivateUse1, m) {
    m.impl("prepared_recurrent", &prepared_recurrent);
}

#endif