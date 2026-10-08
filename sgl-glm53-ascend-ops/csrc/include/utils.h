#pragma once
#include "kernel_operator.h"

__aicore__ inline constexpr uint32_t Align32(uint32_t x) { return (x + 31u) & ~31u; }

template<uint32_t ROWS>
__aicore__ inline void ReduceRows128_AS(
    const AscendC::LocalTensor<float>& src,
    const AscendC::LocalTensor<float>& tmp,
    const AscendC::LocalTensor<float>& dst)
{
    constexpr uint32_t D = 128;

    #pragma unroll
    for (uint32_t row = 0; row < ROWS; ++row) {
        auto srcLo = src[row * D];
        auto srcHi = src[row * D + 64];
        auto dstRow = dst[row];

        // tmp[k] = src[row][k] + src[row][k + 64]
        // k = 0 ... 63
        AscendC::Add(tmp, srcLo, srcHi, 64);
        AscendC::PipeBarrier<PIPE_V>();
        // Sum tmp[0:64].
        AscendC::WholeReduceSum<float>(dstRow, tmp, 64, 
                                       1, 1, 1, 8);
    }
}

struct Reduce_AS {
    template<uint32_t ROWS>
    static __aicore__ inline void run(
        const AscendC::LocalTensor<float>& src,
        const AscendC::LocalTensor<float>& tmp,
        const AscendC::LocalTensor<float>& dst)
    {
        ReduceRows128_AS<ROWS>(src, tmp, dst);
    }
};

template<uint32_t ROWS>
__aicore__ inline void ReduceRows128_SSA(
    const AscendC::LocalTensor<float>& src, // [ROWS, 128]
    const AscendC::LocalTensor<float>& tmp, // [ROWS, 2]
    const AscendC::LocalTensor<float>& dst) // [ROWS]
{
    constexpr uint32_t D = 128;

    #pragma unroll
    for (uint32_t row = 0; row < ROWS; ++row) {
        auto srcRow = src[row * 128];
        auto tmpRow = tmp[row * 2];
        auto dstRow = dst[row];
        AscendC::WholeReduceSum<float>(tmpRow, src, 64, 
                                       2, 1, 1, 8);
        AscendC::WholeReduceSum<float>(dstRow, tmpRow, 2,
                                       1, 1, 1, 8);
    }
}

struct Reduce_SSA {
    template<uint32_t ROWS>
    static __aicore__ inline void run(
        const AscendC::LocalTensor<float>& src,
        const AscendC::LocalTensor<float>& tmp,
        const AscendC::LocalTensor<float>& dst)
    {
        ReduceRows128_SSA<ROWS>(src, tmp, dst);
    }
};



