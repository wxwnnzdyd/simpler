#include <cstdint>

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#ifdef MEMORY_BASE
#undef MEMORY_BASE
#endif
#ifndef REGISTER_BASE
#define REGISTER_BASE
#endif

#include <pto/pto-inst.hpp>

#include "backend/urma/urma_completion_kernel.h"
#include "platform_comm/comm_context.h"
#include "tensor.h"

using namespace pto;

namespace {

constexpr int kElems = 128 * 128;

template <typename T>
static inline __aicore__ __gm__ T *tensor_data(__gm__ Tensor *tensor) {
    return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) + tensor->start_offset;
}

template <typename T>
static inline __aicore__ __gm__ T *comm_remote_ptr(__gm__ CommContext *ctx, __gm__ T *local_ptr, uint32_t peer_rank) {
    uint64_t local_base = ctx->windowsIn[ctx->rankId];
    uint64_t offset = reinterpret_cast<uint64_t>(local_ptr) - local_base;
    uint64_t peer_base =
        pto2::urma_backend::peer_mr_base_addr(reinterpret_cast<__gm__ uint8_t *>(ctx->workSpace), peer_rank);
    return reinterpret_cast<__gm__ T *>(peer_base + offset);
}

}  // namespace

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *input_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ CommContext *comm_ctx = reinterpret_cast<__gm__ CommContext *>(args[2]);

    if (comm_ctx == nullptr || comm_ctx->rankNum != 2 || comm_ctx->rankId >= comm_ctx->rankNum ||
        comm_ctx->workSpace == 0 || comm_ctx->windowsIn[comm_ctx->rankId] == 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    __gm__ float *local_input = tensor_data<float>(input_tensor);
    __gm__ float *local_out = tensor_data<float>(out_tensor);
    uint32_t peer_rank = 1u - comm_ctx->rankId;
    __gm__ float *remote_input = comm_remote_ptr(comm_ctx, local_input, peer_rank);

    using FlatShape = Shape<1, 1, 1, 1, kElems>;
    using FlatStride = pto::Stride<kElems, kElems, kElems, kElems, 1>;
    using GlobalData = GlobalTensor<float, FlatShape, FlatStride>;

    GlobalData remote_global(remote_input);
    GlobalData local_global(local_out);

    AsyncCtx async_ctx = get_async_ctx(args);
    (void)send_request_entry(
        async_ctx,
        UrmaTget(local_global, remote_global, reinterpret_cast<__gm__ uint8_t *>(comm_ctx->workSpace), peer_rank)
    );
}
