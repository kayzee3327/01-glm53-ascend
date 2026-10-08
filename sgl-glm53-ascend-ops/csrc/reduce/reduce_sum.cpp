#include "utils.h"
#include "kernel_operator.h"

constexpr int32_t eventMTE2V = 0;
constexpr int32_t eventVMTE3 = 1;

// ── Kernel: ReduceRows128 (optimized, from utils.h) ─────────────────────
template <typename Reducer, uint32_t ROWS>
__aicore__ inline
void reduce_rows_128_impl(GM_ADDR SRC, GM_ADDR DST, int32_t numRows) {
    AscendC::InitSocState();

    // constexpr uint32_t BV = 16;

    constexpr uint32_t D = 128;
    constexpr uint32_t HALF_D = 64;

    constexpr uint32_t SRC_ADDR  = 0;
    constexpr uint32_t SRC_SIZE  = ROWS * D * sizeof(float);
    constexpr uint32_t TMP_ADDR  = Align32(SRC_ADDR + SRC_SIZE);
    constexpr uint32_t TMP_SIZE  = HALF_D * sizeof(float);
    constexpr uint32_t DST_ADDR  = Align32(TMP_ADDR + TMP_SIZE);
    constexpr uint32_t DST_SIZE  = ROWS * sizeof(float);

    AscendC::LocalTensor<float> srcLocal(AscendC::TPosition::VECCALC, SRC_ADDR, ROWS * D);
    AscendC::LocalTensor<float> tmpLocal(AscendC::TPosition::VECCALC, TMP_ADDR, HALF_D);
    AscendC::LocalTensor<float> dstLocal(AscendC::TPosition::VECCALC, DST_ADDR, ROWS);

    AscendC::GlobalTensor<float> srcGm;
    AscendC::GlobalTensor<float> dstGm;
    srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(SRC));
    dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(DST));

    const uint32_t blockIdx = AscendC::GetBlockIdx();
    const uint32_t numTiles = (numRows + ROWS - 1) / ROWS;

    for (uint32_t tile = blockIdx; tile < numTiles; tile += AscendC::GetBlockNum()) {
        const uint32_t rowStart = tile * ROWS;
        const uint32_t rowsThisTile = (rowStart + ROWS <= static_cast<uint32_t>(numRows))
                                        ? ROWS : (numRows - rowStart);

        AscendC::DataCopy(srcLocal, srcGm[rowStart * D], rowsThisTile * D);
        // MTE2 -> V
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventMTE2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventMTE2V);


        // if (rowsThisTile < ROWS) {
        //     AscendC::Duplicate(srcLocal[rowsThisTile * D], 0.0f, (ROWS - rowsThisTile) * D);
        //     AscendC::PipeBarrier<PIPE_V>();
        // }
        Reducer::template run<ROWS>(srcLocal, tmpLocal, dstLocal);

        // V -> MTE3
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventVMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventVMTE3);

        AscendC::DataCopy(dstGm[rowStart], dstLocal, ROWS);
    }
}

extern "C" __global__ __vector__
void reduce_as_16(GM_ADDR src, GM_ADDR dst, int32_t numRows) 
{
    reduce_rows_128_impl<Reduce_AS, 16>(src, dst, numRows);
}

extern "C" __global__ __vector__
void reduce_ssa_16(GM_ADDR src, GM_ADDR dst, int32_t numRows) 
{
    reduce_rows_128_impl<Reduce_SSA, 16>(src, dst, numRows);
}



// ── Host-side: torch custom op registration ────────────────────────────

#ifndef __CCE_KT_TEST__

#include <acl/acl.h>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

at::Tensor reduce_rows_128(const at::Tensor& src, int64_t impl) {
    at::Tensor dst;
                                                                        
    TORCH_CHECK((src).is_privateuseone(), "src must be an NPU tensor");     
    TORCH_CHECK((src).scalar_type() == at::kFloat, "src must be float32");  
    TORCH_CHECK((src).dim() == 2, "src must be 2-D [numRows, D]");         
    TORCH_CHECK((src).is_contiguous(), "src must be contiguous");           
    TORCH_CHECK((src).size(1) == 128, "reduce_rows_128 requires D=128");   
                                                                            
    const int32_t numRows = static_cast<int32_t>((src).size(0));           
    dst = at::empty({numRows}, (src).options());                                                                      
                                                                                                           
                                                                            
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();                            
                                                                            
    constexpr int32_t blockDim = 8;     
    switch (impl) {
        case 0:
            reduce_as_16<<<blockDim, nullptr, stream>>>(                               
                reinterpret_cast<uint8_t*>(src.data_ptr()),    
                reinterpret_cast<uint8_t*>(dst.data_ptr()),                  
                numRows);
            break;
        case 1:
            reduce_ssa_16<<<blockDim, nullptr, stream>>>(                               
                reinterpret_cast<uint8_t*>(src.data_ptr()),    
                reinterpret_cast<uint8_t*>(dst.data_ptr()),                         
                numRows); 
            break;
    }                                  
                                 
                                                                            
    aclrtSynchronizeStream(stream);                                                                                         

    return dst;
}

TORCH_LIBRARY_FRAGMENT(sgl_glm53_ascend, m) {
    m.def("reduce_rows_128(Tensor src, int impl=0) -> Tensor");
}

TORCH_LIBRARY_IMPL(sgl_glm53_ascend, PrivateUse1, m) {
    m.impl("reduce_rows_128", &reduce_rows_128);
}

#endif
