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

// Reads the software CQ state after a deferred record has been retired.
// Output words:
//   [0] wq->submittedWqeCount
//   [1] AdvanceAcked(...) return value (the contiguous acked prefix)
//   [2] ack ring slot 0
//   [3] ack ring slot 1
//
// The dependency tensor (victim's output) makes the scheduler run this after
// the victim task completes, which means the AICPU has already observed and
// retired the victim record.
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *dep_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *probe_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ CommContext *comm_ctx = reinterpret_cast<__gm__ CommContext *>(args[2]);
    (void)dep_tensor;

    __gm__ int32_t *probe = reinterpret_cast<__gm__ int32_t *>(probe_tensor->buffer.addr) + probe_tensor->start_offset;
    if (comm_ctx == nullptr || comm_ctx->rankNum != 2 || comm_ctx->rankId >= comm_ctx->rankNum ||
        comm_ctx->workSpace == 0) {
        st_dev(-1, probe, 0);
        return;
    }

    uint32_t peer_rank = 1u - comm_ctx->rankId;
    __gm__ uint8_t *workspace = reinterpret_cast<__gm__ uint8_t *>(comm_ctx->workSpace);

    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(workspace, peer_rank, session)) {
        st_dev(-2, probe, 0);
        return;
    }

    __gm__ pto::comm::urma::UrmaCqCtx *cq = pto::comm::urma::detail::GetCqContext(session, peer_rank);
    __gm__ pto::comm::urma::UrmaWQCtx *wq = pto::comm::urma::detail::GetWqContext(session, peer_rank);
    pto::comm::urma::DcciCachelines(reinterpret_cast<__gm__ uint8_t *>(cq), sizeof(*cq));
    pto::comm::urma::DcciCachelines(reinterpret_cast<__gm__ uint8_t *>(wq), sizeof(*wq));

    const uint32_t submitted = wq->submittedWqeCount;
    const uint32_t acked = pto::comm::urma::detail::AdvanceAcked(cq, submitted);

    uint32_t ack0 = 0;
    uint32_t ack1 = 0;
    if (cq->ackAddr != 0U && cq->depth >= 2U) {
        __gm__ uint32_t *base = reinterpret_cast<__gm__ uint32_t *>(cq->ackAddr);
        pto::comm::urma::DcciCachelines(reinterpret_cast<__gm__ uint8_t *>(base), 2U * sizeof(uint32_t));
        ack0 = ld_dev(base + 0U, 0);
        ack1 = ld_dev(base + 1U, 0);
    }

    st_dev(static_cast<int32_t>(submitted), probe + 0, 0);
    st_dev(static_cast<int32_t>(acked), probe + 1, 0);
    st_dev(static_cast<int32_t>(ack0), probe + 2, 0);
    st_dev(static_cast<int32_t>(ack1), probe + 3, 0);
    dsb(DSB_DDR);
}
