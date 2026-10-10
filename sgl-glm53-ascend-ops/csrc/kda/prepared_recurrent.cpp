#include "kernel_operator.h"
#include "utils.h"

#ifndef __CCE_KT_TEST__
#include <acl/acl.h>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#endif

// Static Tensor event IDs are manually managed.
// Avoid IDs 6/7.
constexpr int32_t EVENT_MTE2_V_S = 0;
constexpr int32_t EVENT_MTE2_V_QKVD_0 = 1;
constexpr int32_t EVENT_MTE2_V_QKVD_1 = 2;
constexpr int32_t EVENT_MTE2_V_Q = 1;
constexpr int32_t EVENT_MTE2_V_K = 2;
constexpr int32_t EVENT_MTE2_V_V = 3;
constexpr int32_t EVENT_MTE2_V_D = 4;
constexpr int32_t EVENT_MTE2_S = 1;
constexpr int32_t EVENT_V_S_BETA = 0;
constexpr int32_t EVENT_S_V = 2;
constexpr int32_t EVENT_V_MTE2 = 3;
constexpr int32_t EVENT_MTE3_V = 4;
constexpr int32_t EVENT_V_MTE3 = 5;

class PreparedRecurrent {
public: 
    __aicore__ inline void Init(
        GM_ADDR QN, GM_ADDR KN,         // fp32
        GM_ADDR V,                      // bf16
        GM_ADDR GDECAY, GM_ADDR BETA,   // fp32
        GM_ADDR OUT,                    // bf16
        GM_ADDR STATE,                  // fp32
        GM_ADDR INDICES,                // int32/int64
        GM_ADDR STARTS                  // int32
    );
    __aicore__ inline void Process();
private:
    AscendC::GlobalTensor<float> qnGm;
    AscendC::GlobalTensor<float> knGm;
    AscendC::GlobalTensor<bfloat16_t> vGm;
    AscendC::GlobalTensor<float> decayGm;
    AscendC::GlobalTensor<float> betaGm;
    AscendC::GlobalTensor<bfloat16_t> outGm;
    AscendC::GlobalTensor<float> stateGm;

    AscendC::GlobalTensor<int32_t> indicesGm;
    AscendC::GlobalTensor<int32_t> startsGm;

    struct config {
        static constexpr uint32_t H  = 4;
        static constexpr uint32_t D  = 128;
        static constexpr uint32_t BV = 16;
        static constexpr uint32_t NUM_V_TILES = (D + BV - 1) / BV;
        static constexpr uint32_t NUM_REQ = 1;
        static constexpr uint32_t STAGING_STRIDE = 24;
        static constexpr uint32_t STATE_STRIDE = 144;
    };

    struct UbLayout {
        static constexpr std::uint32_t D = config::D;
        static constexpr std::uint32_t BV = config::BV;

        static constexpr uint32_t STATE_ADDR = 0;
        static constexpr uint32_t STATE_SIZE = BV * config::STATE_STRIDE * sizeof(float);

        static constexpr uint32_t Q0_ADDR = Align32(STATE_ADDR + STATE_SIZE);
        static constexpr uint32_t Q_SIZE = D * sizeof(float);
        static constexpr uint32_t Q1_ADDR = Align32(Q0_ADDR + Q_SIZE);

        static constexpr uint32_t K0_ADDR = Align32(Q1_ADDR + Q_SIZE);
        static constexpr uint32_t K_SIZE = D * sizeof(float);
        static constexpr uint32_t K1_ADDR = Align32(K0_ADDR + K_SIZE);

        static constexpr uint32_t DECAY0_ADDR = Align32(K1_ADDR + K_SIZE);
        static constexpr uint32_t DECAY_SIZE = D * sizeof(float);
        static constexpr uint32_t DECAY1_ADDR = Align32(DECAY0_ADDR + DECAY_SIZE);

        static constexpr uint32_t V0_ADDR = Align32(DECAY1_ADDR + DECAY_SIZE);
        static constexpr uint32_t V_SIZE = BV * sizeof(bfloat16_t);
        static constexpr uint32_t V1_ADDR = Align32(V0_ADDR + V_SIZE);

        static constexpr uint32_t OUT_ADDR = Align32(V1_ADDR + V_SIZE);
        static constexpr uint32_t OUT_SIZE = BV * sizeof(bfloat16_t);

        static constexpr uint32_t UB_END = Align32(OUT_ADDR + OUT_SIZE);
    };

    AscendC::LocalTensor<float> stateLocal;
    AscendC::LocalTensor<float> qLocal[2];
    AscendC::LocalTensor<float> kLocal[2];
    AscendC::LocalTensor<float> decayLocal[2];
    AscendC::LocalTensor<bfloat16_t> vLocal[2];
    AscendC::LocalTensor<bfloat16_t> outLocal;

    __aicore__ inline void LoadStateTile(
        const AscendC::LocalTensor<float>& state,
        const AscendC::LocalTensor<float>& staging,
        const AscendC::GlobalTensor<float>& stateGm,
        uint64_t gmOffset);
    __aicore__ inline void StoreStateTile(
        const AscendC::LocalTensor<float>& state,
        const AscendC::LocalTensor<float>& staging,
        const AscendC::GlobalTensor<float>& stateGm,
        uint64_t gmOffset);
};

__aicore__ inline void PreparedRecurrent::Init(
    GM_ADDR QN, GM_ADDR KN, GM_ADDR V, 
    GM_ADDR GDECAY, GM_ADDR BETA, GM_ADDR OUT, 
    GM_ADDR STATE, GM_ADDR INDICES, GM_ADDR STARTS) 
{
    qnGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(QN));
    knGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(KN));
    vGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(V));
    decayGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(GDECAY));
    betaGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(BETA));
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(OUT));
    stateGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(STATE));
    indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(INDICES));
    startsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(STARTS));

    constexpr uint32_t BV = config::BV;
    constexpr uint32_t D = config::D;

    stateLocal = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, UbLayout::STATE_ADDR, BV * config::STATE_STRIDE);
    qLocal[0] = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, UbLayout::Q0_ADDR, D);
    qLocal[1] = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, UbLayout::Q1_ADDR, D);
    kLocal[0] = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, UbLayout::K0_ADDR, D);
    kLocal[1] = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, UbLayout::K1_ADDR, D);
    decayLocal[0] = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, UbLayout::DECAY0_ADDR, D);
    decayLocal[1] = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, UbLayout::DECAY1_ADDR, D);
    vLocal[0] = AscendC::LocalTensor<bfloat16_t>(AscendC::TPosition::VECCALC, UbLayout::V0_ADDR, BV);
    vLocal[1] = AscendC::LocalTensor<bfloat16_t>(AscendC::TPosition::VECCALC, UbLayout::V1_ADDR, BV);
    outLocal = AscendC::LocalTensor<bfloat16_t>(AscendC::TPosition::VECCALC, UbLayout::OUT_ADDR, BV);
}

__aicore__ inline void PreparedRecurrent::LoadStateTile(
    const AscendC::LocalTensor<float>& state,
    const AscendC::LocalTensor<float>& staging,
    const AscendC::GlobalTensor<float>& stateGm,
    uint64_t gmOffset)
{
    constexpr uint32_t STAGING_STRIDE = config::STAGING_STRIDE;
    constexpr uint32_t STATE_STRIDE = config::STATE_STRIDE;
    // Step 1: GM [128,128] -> UB [128,24]
    // 16 valid FP32 per row, followed by 8 padding slots.
    AscendC::DataCopyParams copyParams{
        128,  // blockCount
        2,    // blockLen: 64 bytes
        14,   // srcStride: skip 448 bytes
        1     // dstStride: skip 32 bytes
    };

    AscendC::DataCopy(staging, stateGm[gmOffset], copyParams);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V_S);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V_S);

    // Step 2: UB [128,24] -> UB [16,144]
    // Step 2: UB [128,24] -> UB [16,144]
    AscendC::TransDataTo5HDParams params{};
    params.repeatTimes = 8;
    params.srcRepStride = 48;
    params.dstRepStride = 2;

    // Each call processes 8 values of V across 128 values of K.
    #pragma unroll
    for (uint32_t vh = 0; vh < 2; ++vh) {
        
        uint64_t srcList[16];
        uint64_t dstList[16];

        #pragma unroll
        for (uint32_t i = 0; i < 16; ++i) {
            // 16 K rows, each contributing 8 V values.
            srcList[i] = (uint64_t)(staging[i * STAGING_STRIDE + vh * 8].GetPhyAddr());

            // 8 output V rows, 2 DataBlocks of K per row.
            const uint32_t v = vh * 8 + i / 2;
            const uint32_t k = (i % 2) * 8;

            dstList[i] = (uint64_t)(state[v * STATE_STRIDE + k].GetPhyAddr());
        }

        AscendC::TransDataTo5HD<float>(dstList, srcList, params);
    }
    // AscendC::TransDataTo5HDParams params{};
    // params.repeatTimes = 16;
    // params.srcRepStride = 8 * STAGING_STRIDE / 8; // 24 DataBlocks
    // params.dstRepStride = 1;                     // 8个float

    // uint64_t srcList[16];
    // uint64_t dstList[16];
    
    // #pragma unroll
    // for (uint32_t i = 0; i < 16; ++i) {
    //     // 前8项：K=0..7，各取V=0..7
    //     // 后8项：K=0..7，各取V=8..15
    //     const uint32_t k = i % 8;
    //     const uint32_t vBase = (i / 8) * 8;
    
    //     srcList[i] = (uint64_t)(
    //         staging[k * STAGING_STRIDE + vBase].GetPhyAddr());
        
    //     // 每对目标地址分别接收：
    //     // 第c行的8个K、第c+8行的8个K。
    //     const uint32_t v = i / 2 + (i % 2) * 8;
        
    //     dstList[i] = (uint64_t)(
    //         state[v * STATE_STRIDE].GetPhyAddr());
    // }
    
    // AscendC::TransDataTo5HD<float>(dstList, srcList, params);
}

__aicore__ inline void PreparedRecurrent::StoreStateTile(
    const AscendC::LocalTensor<float>& state,
    const AscendC::LocalTensor<float>& staging,
    const AscendC::GlobalTensor<float>& stateGm,
    uint64_t gmOffset)
{
    constexpr uint32_t D = 128;
    constexpr uint32_t BV = 16;
    constexpr uint32_t STAGING_STRIDE = 24;
    constexpr uint32_t STATE_STRIDE = 144;

    // Step 1: UB [16,144] -> UB [128,24]
    // Transpose [V,K] -> [K,V]

    AscendC::PipeBarrier<PIPE_V>();

    AscendC::TransDataTo5HDParams params{};
    params.repeatTimes = 16;
    params.srcRepStride = 1;
    params.dstRepStride = 24;

    uint64_t srcList[16];
    uint64_t dstList[16];

    #pragma unroll
    for (uint32_t i = 0; i < 16; ++i) {
        // 16 V rows, each contributing 8 K values.
        srcList[i] = (uint64_t)(
            state[i * STATE_STRIDE].GetPhyAddr()
        );

        // 8 output K rows, each containing 16 V values.
        const uint32_t k = i / 2;
        const uint32_t v = (i % 2) * 8;

        dstList[i] = (uint64_t)(
            staging[k * STAGING_STRIDE + v].GetPhyAddr()
        );
    }

    AscendC::TransDataTo5HD<float>(
        dstList,
        srcList,
        params
    );

    // Wait until Vector writes to staging are complete.
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);

    // Step 2: UB [128,24] -> GM [128,128]
    // Each row writes 16 FP32 values (64 bytes).

    AscendC::DataCopyParams copyParams{
        128,  // blockCount: 128 K rows
        2,    // blockLen: 16 FP32 = 2 DataBlocks
        1,    // srcGap: skip 8 FP32 in staging
        14    // dstGap: skip 112 FP32 in GM
    };

    AscendC::DataCopy(
        stateGm[gmOffset],
        staging,
        copyParams
    );

    // Ensure MTE3 has finished reading staging
    // before the buffer is reused.
}

__aicore__ inline void PreparedRecurrent::Process()
{
    AscendC::InitSocState();

    const uint32_t blockIdx = AscendC::GetBlockIdx();
    constexpr uint32_t NUM_V_TILES = config::NUM_V_TILES;
    constexpr uint32_t NUM_REQ = config::NUM_REQ;
    constexpr uint32_t H = config::H;
    constexpr uint32_t STAGING_STRIDE = config::STAGING_STRIDE;
    constexpr uint32_t STATE_STRIDE = config::STATE_STRIDE;
    constexpr uint32_t D = config::D;
    constexpr uint32_t BV = config::BV;

    // iv, req, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    const uint32_t iv   = blockIdx % NUM_V_TILES;
    const uint32_t req  = (blockIdx / NUM_V_TILES) % NUM_REQ;
    const uint32_t head = blockIdx / (NUM_V_TILES * NUM_REQ);
    
    // bos = tl.load(STARTS + req).to(tl.int64)
    // eos = tl.load(STARTS + req + 1).to(tl.int64)
    // slot = tl.load(INDICES + req)
    const int32_t bos = startsGm.GetValue(req);
    const int32_t eos = startsGm.GetValue(req + 1);
    const int32_t slot = indicesGm.GetValue(req);

    // pstate points at: state[slot, head, iv*BV, 0]
    const uint64_t pstate =
        (static_cast<uint64_t>(slot) * H + head) * D * D 
        + static_cast<uint64_t>(iv*BV);

    if (slot > 0) {
        AscendC::LocalTensor<float> stagingLocal(AscendC::TPosition::VECCALC, UbLayout::UB_END, D * STAGING_STRIDE);
        LoadStateTile(stateLocal, stagingLocal, stateGm, pstate);
    } else {
        AscendC::Duplicate(stateLocal, 0.0f, BV * STATE_STRIDE);
    }
    AscendC::PipeBarrier<PIPE_V>();

    // Load Beta into UB first
    // Avoid GM Load latency and Vector waiting Scalar

    constexpr uint64_t tempBVDaddr = UbLayout::UB_END;
    constexpr uint64_t tempHDaddr = Align32(tempBVDaddr + BV * D * sizeof(float));
    constexpr uint64_t tempReduceAddr = Align32(tempHDaddr + D / 2 * sizeof(float));
    constexpr uint64_t tempBVaddr = Align32(tempReduceAddr + BV * sizeof(float));
    constexpr uint64_t tempBrcbAddr = Align32(tempBVaddr + BV * sizeof(float));
    AscendC::LocalTensor<float> tmpBVDLocal(AscendC::TPosition::VECCALC, tempBVDaddr, BV*D);
    AscendC::LocalTensor<float> tmpHDLocal(AscendC::TPosition::VECCALC, tempHDaddr, D / 2);
    AscendC::LocalTensor<float> reduceLocal(AscendC::TPosition::VECCALC, tempReduceAddr, BV);
    AscendC::LocalTensor<float> tmpBVLocal(AscendC::TPosition::VECCALC, tempBVaddr, BV);
    AscendC::LocalTensor<float> tmpBrcbLocal(AscendC::TPosition::VECCALC, tempBrcbAddr, 128);

    constexpr uint64_t BANK_PAD = 480;
    constexpr uint64_t tempMulReduceAddr = Align32(tempBrcbAddr + 128* sizeof(float)) + BANK_PAD;
    AscendC::LocalTensor<float> tmpMulReduceLocal(AscendC::TPosition::VECCALC, tempMulReduceAddr, BV*D /2);


    // Prefetch first token into buf[0].
    if (bos < eos) {
        const uint64_t row0 = static_cast<uint64_t>(bos) * H + head;
        AscendC::DataCopy(qLocal[0], qnGm[row0 * D], D);
        AscendC::DataCopy(kLocal[0], knGm[row0 * D], D);
        AscendC::DataCopy(decayLocal[0], decayGm[row0 * D], D);
        AscendC::DataCopy(vLocal[0], vGm[row0 * D + iv * BV], BV);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V_QKVD_0);
    }

    AscendC::BinaryRepeatParams brcb_muladddst_params = {
        STATE_STRIDE / 8, // dstBlkStride = 18
        0,               // src0BlkStride: reuse same k block
        1,               // src1BlkStride: next v broadcast block
        1,                // dstRepStride: next column block
        1,               // src0RepStride: next k block
        0                // src1RepStride: reuse v0..v7
    };

    // stride 的单位为 32B，即 8 个 float：
    // tmp 下一行 +64；state 下一行 +144；x 每行复用。
    AscendC::BinaryRepeatParams mul_reduce_params{
        1, 
        1, 
        1,
        8, 
        STATE_STRIDE / 8, 
        0
    };
    AscendC::BinaryRepeatParams decay_params{
        1, 
        1, 
        1,
        STATE_STRIDE / 8, 
        STATE_STRIDE / 8, 
        0
    };
    for (int32_t token = bos; token < eos; ++token)
    {
        bool pos2 = token > bos;
        bool posn_1 = token + 1 < eos;

        // q = tl.load(QN + row * D + ks)
        // k = tl.load(KN + row * D + ks)
        // v = tl.load(V + row * D + vs).to(tl.float32)
        // decay = tl.load(GDECAY + row * D + ks)
        const uint64_t row = static_cast<uint64_t>(token) * H + head;
        const uint32_t cur = (token - bos) & 1;
        const uint32_t nxt = 1 - cur;
        int32_t cur_buf_e = cur == 0 ? EVENT_MTE2_V_QKVD_0 : EVENT_MTE2_V_QKVD_1;
        int32_t nxt_buf_e = nxt == 0 ? EVENT_MTE2_V_QKVD_0 : EVENT_MTE2_V_QKVD_1;

        if (pos2) {
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_V_S_BETA);
        }
        const float betaValue = betaGm.GetValue(row);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_S_V);

        // Ensure Vector is done with buf[nxt] (used in the previous iteration)
        // before MTE2 overwrites it with the prefetch.
        if (pos2) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_V_MTE2);
        }

        // Prefetch next token into buf[nxt].
        // MTE2 loads run concurrently with the Vector computation below.
        if (posn_1) {
            const uint64_t nextRow = static_cast<uint64_t>(token + 1) * H + head;
            AscendC::DataCopy(qLocal[nxt], qnGm[nextRow * D], D);
            AscendC::DataCopy(kLocal[nxt], knGm[nextRow * D], D);
            AscendC::DataCopy(decayLocal[nxt], decayGm[nextRow * D], D);
            AscendC::DataCopy(vLocal[nxt], vGm[nextRow * D + iv * BV], BV);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(nxt_buf_e);
        }

        // Wait for all current-buffer loads (issued before the loop or as
        // prefetch at the end of the previous iteration).
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(cur_buf_e);

        // state *= decay[None, :]
        //////////////////////////////////////////
        // #pragma unroll
        // for (uint32_t j = 0; j < BV; ++j) {
        //     auto stateRow = stateLocal[j * STATE_STRIDE];
        //     AscendC::Mul(stateRow, stateRow, decayLocal[cur], D);
        // }
        // AscendC::PipeBarrier<PIPE_V>();
        //////////////////////////////////////////
        AscendC::Mul(
            stateLocal, stateLocal, decayLocal[cur],
            64, BV, decay_params);
        AscendC::Mul(
            stateLocal[64], stateLocal[64], decayLocal[cur][64],
            64, BV, decay_params);
        AscendC::PipeBarrier<PIPE_V>();

        // v -= tl.sum(state * k[None, :], axis=1)
        //////////////////////////////////////////
        // #pragma unroll
        // for (uint32_t j = 0; j < BV; ++j) {
            // auto stateRow = stateLocal[j * STATE_STRIDE];
            // auto tempRow = tmpBVDLocal[j * D];
            // AscendC::Mul(tempRow, stateRow, kLocal[cur], D);
        // }
        // AscendC::PipeBarrier<PIPE_V>();
        // ReduceRows128_AS<BV>(tmpBVDLocal, tmpHDLocal, reduceLocal);
        //////////////////////////////////////////
        AscendC::Mul(tmpMulReduceLocal, stateLocal, kLocal[cur], 64, BV, mul_reduce_params);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::MulAddDst(tmpMulReduceLocal, stateLocal[64], kLocal[cur][64], 64, BV, mul_reduce_params);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum<float>(reduceLocal, tmpMulReduceLocal, 
                                      64, BV, 1, 1, 
                                      8);
                                      
        AscendC::Cast(tmpBVLocal, vLocal[cur], AscendC::RoundMode::CAST_NONE, BV);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(tmpBVLocal, tmpBVLocal, reduceLocal, BV);
        AscendC::PipeBarrier<PIPE_V>();

        // v *= beta
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_S_V);
        AscendC::Muls(tmpBVLocal, tmpBVLocal, betaValue, BV);
        if (posn_1) {
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_V_S_BETA);
        }

        // state += k[None, :] * v[:, None]
        AscendC::PipeBarrier<PIPE_V>();
        // Broadcast 16 FP32 values into 16 DataBlocks.
        AscendC::Brcb(tmpBrcbLocal, tmpBVLocal, 2, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        // rows 0..7
        AscendC::MulAddDst(stateLocal, kLocal[cur], tmpBrcbLocal, 
                          64, 16, brcb_muladddst_params);
        // rows 8..15
        AscendC::MulAddDst(stateLocal[8 * STATE_STRIDE], kLocal[cur], tmpBrcbLocal[8 * 8], 
                          64, 16, brcb_muladddst_params);
        AscendC::PipeBarrier<PIPE_V>();

        // out = tl.sum(state * q[None, :], axis=1)
        //////////////////////////////////////////
        // #pragma unroll
        // for (uint32_t j = 0; j < BV; ++j) {
        //     auto stateRow = stateLocal[j * STATE_STRIDE];
        //     auto tempRow = tmpBVDLocal[j * D];
        //     AscendC::Mul(tempRow, stateRow, qLocal[cur], D);
        // }
        // AscendC::PipeBarrier<PIPE_V>();
        // ReduceRows128_AS<BV>(tmpBVDLocal, tmpHDLocal, reduceLocal);
        // AscendC::PipeBarrier<PIPE_V>();
        //////////////////////////////////////////
        AscendC::Mul(tmpMulReduceLocal, stateLocal, qLocal[cur], 64, BV, mul_reduce_params);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::MulAddDst(tmpMulReduceLocal, stateLocal[64], qLocal[cur][64], 64, BV, mul_reduce_params);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum<float>(reduceLocal, tmpMulReduceLocal, 
                                      64, BV, 1, 1, 
                                      8);

        // Signal that Vector is done reading buf[cur].
        // The next iteration's prefetch (into buf[cur]) will wait for this.
        if (posn_1) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_V_MTE2);
        }

        // tl.store(OUT + row * D + vs, out.to(OUT.dtype.element_ty))
        if (pos2) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_MTE3_V);
        }
        AscendC::Cast(outLocal, reduceLocal, AscendC::RoundMode::CAST_RINT, BV);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_MTE3);
        AscendC::DataCopy(outGm[row * D + iv * BV], outLocal, BV);
        if (posn_1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_MTE3_V);
        }

    }

    if (slot > 0) {
        AscendC::LocalTensor<float> stagingLocal(AscendC::TPosition::VECCALC, UbLayout::UB_END, D * STAGING_STRIDE);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_MTE3);
        StoreStateTile(stateLocal, stagingLocal, stateGm, pstate);
    }
    
}


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
    PreparedRecurrent op;
    op.Init(QN, KN,         // fp32
            V,                      // bf16
            GDECAY, BETA,   // fp32
            OUT,                    // bf16
            STATE,                  // fp32
            INDICES,                // int32/int64
            STARTS                 // int32
    );
    op.Process();
}

// ── Host-side: torch custom op registration ────────────────────────────

#ifndef __CCE_KT_TEST__

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
    // printf("start\n");
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