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

// The sequence task retires only after every slot it posted has been acked, so
// submitted and acked must both equal the total WQE count of the run.
constexpr int32_t kSummaryAckedTwoLaps = 1;
constexpr int32_t kSummaryAckedOneLap = 2;
constexpr int32_t kSummaryAckedNoWrap = 3;
constexpr int32_t kSummaryMismatch = 0;

template <typename T>
static inline __aicore__ __gm__ T *tensor_data(__gm__ Tensor *tensor) {
    return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) + tensor->start_offset;
}

}  // namespace

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *summary_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ CommContext *comm_ctx = reinterpret_cast<__gm__ CommContext *>(args[1]);
    const uint32_t expected = static_cast<uint32_t>(args[2]);

    __gm__ int32_t *summary = tensor_data<int32_t>(summary_tensor);
    if (comm_ctx == nullptr || comm_ctx->rankNum != 2 || comm_ctx->rankId >= comm_ctx->rankNum ||
        comm_ctx->workSpace == 0 || comm_ctx->windowsIn[comm_ctx->rankId] == 0) {
        st_dev(kSummaryMismatch, summary + 2, 0);
        dsb(DSB_DDR);
        pipe_barrier(PIPE_ALL);
        return;
    }

    const uint32_t peer_rank = 1u - comm_ctx->rankId;
    __gm__ uint8_t *workspace = reinterpret_cast<__gm__ uint8_t *>(comm_ctx->workSpace);

    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(workspace, peer_rank, session)) {
        st_dev(kSummaryMismatch, summary + 2, 0);
        dsb(DSB_DDR);
        pipe_barrier(PIPE_ALL);
        return;
    }

    __gm__ pto::comm::urma::UrmaCqCtx *cq = pto::comm::urma::detail::GetCqContext(session, peer_rank);
    __gm__ pto::comm::urma::UrmaWQCtx *wq = pto::comm::urma::detail::GetWqContext(session, peer_rank);
    pto::comm::urma::DcciCachelines(reinterpret_cast<__gm__ uint8_t *>(cq), sizeof(*cq));
    pto::comm::urma::DcciCachelines(reinterpret_cast<__gm__ uint8_t *>(wq), sizeof(*wq));

    const uint32_t submitted = wq->submittedWqeCount;
    const uint32_t acked = pto::comm::urma::detail::AdvanceAcked(cq, submitted);

    int32_t code = kSummaryMismatch;
    if (submitted == expected && acked == expected) {
        if (expected > 2U * cq->depth) {
            code = kSummaryAckedTwoLaps;
        } else if (expected > cq->depth) {
            code = kSummaryAckedOneLap;
        } else {
            code = kSummaryAckedNoWrap;
        }
    }

    st_dev(static_cast<int32_t>(submitted), summary + 0, 0);
    st_dev(static_cast<int32_t>(acked), summary + 1, 0);
    st_dev(code, summary + 2, 0);
    dsb(DSB_DDR);
    pipe_barrier(PIPE_ALL);
}
