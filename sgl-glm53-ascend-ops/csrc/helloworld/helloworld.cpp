// ── Device kernel (compiled by bisheng for NPU) ────────────────────────
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t ALIGN_BYTES = 32;

#pragma pack(push, 1)
struct HelloWorldTiling {
    int32_t totalLength;
    float scalar;
};
#pragma pack(pop)

class HelloWorldOp {
public:
    __aicore__ inline HelloWorldOp() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                int32_t totalLength, half scalar)
    {
        int32_t blockNum = AscendC::GetBlockNum();
        int32_t blockIdx = AscendC::GetBlockIdx();
        int32_t tileLength = totalLength / blockNum;
        // Align to 32-byte boundary (16 half elements)
        int32_t tileLengthAligned = (tileLength + 15) / 16 * 16;

        xGm.SetGlobalBuffer((__gm__ half *)x + blockIdx * tileLength, tileLength);
        yGm.SetGlobalBuffer((__gm__ half *)y + blockIdx * tileLength, tileLength);

        pipe.InitBuffer(inQueue, BUFFER_NUM, tileLengthAligned * sizeof(half));
        pipe.InitBuffer(outQueue, BUFFER_NUM, tileLengthAligned * sizeof(half));

        this->tileLength = tileLength;
        this->tileLengthAligned = tileLengthAligned;
        this->scalar = scalar;
    }

    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<half> xLocal = inQueue.AllocTensor<half>();
        AscendC::DataCopy(xLocal, xGm, tileLengthAligned);
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> xLocal = inQueue.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = outQueue.AllocTensor<half>();
        AscendC::Adds(yLocal, xLocal, scalar, tileLengthAligned);
        outQueue.EnQue(yLocal);
        inQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<half> yLocal = outQueue.DeQue<half>();
        AscendC::DataCopy(yGm, yLocal, tileLengthAligned);
        outQueue.FreeTensor(yLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> yGm;
    int32_t tileLength;
    int32_t tileLengthAligned;
    half scalar;
};

extern "C" __global__ __aicore__
void helloworld_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    auto *td = (const __gm__ HelloWorldTiling *)tiling;
    int32_t totalLength = td->totalLength;
    float scalar = td->scalar;

    HelloWorldOp op;
    op.Init(x, y, totalLength, static_cast<half>(scalar));
    op.Process();
}

// ── Host-side: torch custom op registration ────────────────────────────

#ifndef __CCE_KT_TEST__

#include <acl/acl.h>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

at::Tensor helloworld(const at::Tensor &x, double scalar)
{
    TORCH_CHECK(x.is_privateuseone(), "Input must be an NPU tensor");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "Input must be float16");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto y = at::empty_like(x);

    int32_t totalLength = static_cast<int32_t>(x.numel());
    constexpr int32_t blockDim = 8;
    TORCH_CHECK(totalLength % blockDim == 0,
                "numel must be divisible by blockDim (", blockDim, ")");

    // Tiling data must be in device-accessible memory
    HelloWorldTiling hostTd;
    hostTd.totalLength = totalLength;
    hostTd.scalar = static_cast<float>(scalar);

    void *devTd = nullptr;
    aclrtMalloc(&devTd, sizeof(HelloWorldTiling), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    aclrtMemcpyAsync(devTd, sizeof(HelloWorldTiling),
                     &hostTd, sizeof(HelloWorldTiling),
                     ACL_MEMCPY_HOST_TO_DEVICE, stream);

    helloworld_kernel<<<blockDim, nullptr, stream>>>(
        reinterpret_cast<uint8_t *>(const_cast<void *>(x.data_ptr())),
        reinterpret_cast<uint8_t *>(y.data_ptr()),
        nullptr,
        reinterpret_cast<uint8_t *>(devTd));

    // Free tiling buffer after kernel completes on stream
    aclrtSynchronizeStream(stream);
    aclrtFree(devTd);

    return y;
}

TORCH_LIBRARY(sgl_glm53_ascend, m) {
    m.def("helloworld(Tensor x, float scalar=1.0) -> Tensor");
}

TORCH_LIBRARY_IMPL(sgl_glm53_ascend, PrivateUse1, m) {
    m.impl("helloworld", &helloworld);
}

#endif
