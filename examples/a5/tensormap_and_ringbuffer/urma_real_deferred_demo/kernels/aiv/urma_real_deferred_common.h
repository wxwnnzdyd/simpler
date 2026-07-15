#pragma once

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
#include "pto/comm/comm_types.hpp"
#include "pto/comm/pto_comm_inst.hpp"
#ifdef PTO_URMA_SUPPORTED
#include "pto/comm/async/urma/urma_async_intrin.hpp"
#endif

#include "backend/urma/urma_completion_kernel.h"
#include "platform_comm/comm_context.h"
#include "tensor.h"

namespace urma_real_deferred {

using ShapeDyn = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
using GlobalFloat = pto::GlobalTensor<float, ShapeDyn, StrideDyn>;

constexpr uint32_t kMaxRemoteWritePollIters = 10000000;

enum class Status : int32_t {
    kOk = 0,
    kUnsupported = 1,
    kInvalidComm = 2,
    kInvalidElementCount = 3,
    kSubmitFailed = 4,
    kTgetMismatch = 100,
    kTputMismatch = 200,
};

inline __aicore__ void set_status(__gm__ int32_t *status, Status code, int32_t detail0 = 0, int32_t detail1 = 0) {
    status[0] = static_cast<int32_t>(code);
    status[1] = detail0;
    status[2] = detail1;
}

template <typename T>
inline __aicore__ __gm__ T *tensor_data(__gm__ Tensor *tensor) {
    return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) + tensor->start_offset;
}

inline __aicore__ bool validate_comm(__gm__ CommContext *comm_ctx, uint32_t elem_count, __gm__ int32_t *status) {
#ifndef PTO_URMA_SUPPORTED
    (void)comm_ctx;
    (void)elem_count;
    set_status(status, Status::kUnsupported);
    return false;
#else
    if (comm_ctx == nullptr || comm_ctx->rankNum != 2 || comm_ctx->rankId >= comm_ctx->rankNum ||
        comm_ctx->workSpace == 0 || comm_ctx->windowsIn[comm_ctx->rankId] == 0) {
        set_status(status, Status::kInvalidComm);
        return false;
    }
    if (elem_count == 0) {
        set_status(status, Status::kInvalidElementCount);
        return false;
    }
    return true;
#endif
}

inline __aicore__ uint32_t peer_rank(__gm__ CommContext *comm_ctx) {
    return (comm_ctx->rankId + 1u) % comm_ctx->rankNum;
}

inline __aicore__ GlobalFloat global_float(__gm__ float *ptr, uint32_t elem_count) {
    ShapeDyn shape(1, 1, 1, 1, elem_count);
    StrideDyn stride(elem_count, elem_count, elem_count, elem_count, 1);
    return GlobalFloat(ptr, shape, stride);
}

inline __aicore__ uint64_t remote_base(__gm__ CommContext *comm_ctx, uint32_t peer) {
#ifdef PTO_URMA_SUPPORTED
    return pto::comm::urma::UrmaPeerMrBaseAddr(reinterpret_cast<__gm__ uint8_t *>(comm_ctx->workSpace), peer);
#else
    (void)comm_ctx;
    (void)peer;
    return 0;
#endif
}

template <typename T>
inline __aicore__ uint64_t local_offset(__gm__ CommContext *comm_ctx, __gm__ T *ptr) {
    return reinterpret_cast<uint64_t>(ptr) - comm_ctx->windowsIn[comm_ctx->rankId];
}

}  // namespace urma_real_deferred
