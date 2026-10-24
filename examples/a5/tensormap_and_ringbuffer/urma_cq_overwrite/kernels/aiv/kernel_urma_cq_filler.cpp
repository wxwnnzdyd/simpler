/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

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

namespace {

constexpr int kScratchElems = 1024;

template <typename T>
static inline __aicore__ __gm__ T *tensor_data(__gm__ Tensor *tensor) {
    return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) + tensor->start_offset;
}

}  // namespace

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *input_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *scratch_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *flag_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ CommContext *comm_ctx = reinterpret_cast<__gm__ CommContext *>(args[3]);
    uint32_t wrap = (static_cast<uint32_t>(args[4]) != 0U) ? 1U : 0U;

    if (comm_ctx == nullptr || comm_ctx->rankNum != 2 || comm_ctx->rankId >= comm_ctx->rankNum ||
        comm_ctx->workSpace == 0 || comm_ctx->windowsIn[comm_ctx->rankId] == 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    __gm__ float *local_input = tensor_data<float>(input_tensor);
    __gm__ float *local_scratch = tensor_data<float>(scratch_tensor);
    __gm__ int32_t *posted = tensor_data<int32_t>(flag_tensor);
    uint32_t peer_rank = 1u - comm_ctx->rankId;
    uint64_t input_offset = reinterpret_cast<uint64_t>(local_input) - comm_ctx->windowsIn[comm_ctx->rankId];
    __gm__ uint8_t *workspace = reinterpret_cast<__gm__ uint8_t *>(comm_ctx->workSpace);
    __gm__ float *remote_input = pto2::urma_backend::peer_mr_ptr<float>(workspace, peer_rank, input_offset);

    while (ld_dev(reinterpret_cast<__gm__ uint32_t *>(posted), 0) == 0U) {}
    dsb(DSB_DDR);

    using FlatShape = pto::Shape<1, 1, 1, 1, kScratchElems>;
    using FlatStride = pto::Stride<kScratchElems, kScratchElems, kScratchElems, kScratchElems, 1>;
    using GlobalData = pto::GlobalTensor<float, FlatShape, FlatStride>;

    GlobalData remote_global(remote_input);
    GlobalData local_global(local_scratch);

    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(workspace, peer_rank, session)) {
        return;
    }

    __gm__ pto::comm::urma::UrmaCqCtx *cq = pto::comm::urma::detail::GetCqContext(session, peer_rank);
    if (cq == nullptr) {
        return;
    }
    pto::comm::urma::DcciCachelines(reinterpret_cast<__gm__ uint8_t *>(cq), sizeof(*cq));
    if (cq->depth < 2U) {
        return;
    }
    const uint32_t posts = cq->depth - 1U + wrap;
    for (uint32_t i = 0; i < posts; ++i) {
        (void)pto::comm::TGET_ASYNC<pto::comm::DmaEngine::URMA>(local_global, remote_global, session);
    }
}
