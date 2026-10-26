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

// 257 MB of float payload: one TGET slice that exceeds the 256 MB single-WQE
// cap, so UrmaPostSliceOnJetty splits it into two WQEs (seq 1 and seq 2) but
// GetUrmaCompletionRecordAt still publishes one record whose sequence is 2.
constexpr int kElems = 67371008;

template <typename T>
static inline __aicore__ __gm__ T *tensor_data(__gm__ Tensor *tensor) {
    return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) + tensor->start_offset;
}

}  // namespace

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *input_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *flag_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ CommContext *comm_ctx = reinterpret_cast<__gm__ CommContext *>(args[3]);

    if (comm_ctx == nullptr || comm_ctx->rankNum != 2 || comm_ctx->rankId >= comm_ctx->rankNum ||
        comm_ctx->workSpace == 0 || comm_ctx->windowsIn[comm_ctx->rankId] == 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    __gm__ float *local_input = tensor_data<float>(input_tensor);
    __gm__ float *local_out = tensor_data<float>(out_tensor);
    __gm__ int32_t *posted = tensor_data<int32_t>(flag_tensor);
    uint32_t peer_rank = 1u - comm_ctx->rankId;
    uint64_t input_offset = reinterpret_cast<uint64_t>(local_input) - comm_ctx->windowsIn[comm_ctx->rankId];
    __gm__ uint8_t *workspace = reinterpret_cast<__gm__ uint8_t *>(comm_ctx->workSpace);
    __gm__ float *remote_input = pto2::urma_backend::peer_mr_ptr<float>(workspace, peer_rank, input_offset);

    using FlatShape = pto::Shape<1, 1, 1, 1, kElems>;
    using FlatStride = pto::Stride<kElems, kElems, kElems, kElems, 1>;
    using GlobalData = pto::GlobalTensor<float, FlatShape, FlatStride>;

    GlobalData remote_global(remote_input);
    GlobalData local_global(local_out);

    AsyncCtx async_ctx = get_async_ctx(args);
    (void)send_request_entry(async_ctx, UrmaTget(local_global, remote_global, workspace, peer_rank));

    st_dev(1U, reinterpret_cast<__gm__ uint32_t *>(posted), 0);
    dsb(DSB_DDR);
    pipe_barrier(PIPE_ALL);

    // Keep the victim task alive long enough for the AICPU to observe its
    // completion after the poll delay.
    const auto hold_until = get_sys_cnt() + 5'000'000LL;
    while (get_sys_cnt() < hold_until) {}
}
