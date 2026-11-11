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

// Posts kPostsPerTask single-WQE transfers for one chunk of the sequence.
// Chained runs call it with increasing task_index; the parallel run calls it
// once per AIV with the same scalar layout, which is why the chunk offset is a
// scalar instead of a tensor view.
//
// Nothing here waits: the completion is deferred, so the data is checked by the
// summary task once the whole sequence is acked.

#include "rdma_cq_common.h"

namespace {

using namespace rdma_cq_sequence;

constexpr int kArgSend = 0;
constexpr int kArgRecv = 1;
constexpr int kArgTput = 2;
constexpr int kArgCommCtx = 3;
constexpr int kArgPosts = 4;
constexpr int kArgTaskIndex = 5;

}  // namespace

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *send_tensor = reinterpret_cast<__gm__ Tensor *>(args[kArgSend]);
    __gm__ Tensor *recv_tensor = reinterpret_cast<__gm__ Tensor *>(args[kArgRecv]);
    __gm__ Tensor *tput_tensor = reinterpret_cast<__gm__ Tensor *>(args[kArgTput]);
    __gm__ CommContext *comm_ctx = reinterpret_cast<__gm__ CommContext *>(args[kArgCommCtx]);
    const uint32_t posts = static_cast<uint32_t>(args[kArgPosts]);
    const uint32_t task_index = static_cast<uint32_t>(args[kArgTaskIndex]);

    // The summary task owns the only host status tensor; a post task cannot write
    // one without becoming a second producer of the same tensor.
    if (!validate_comm(comm_ctx, nullptr)) {
        return;
    }
    if (posts == 0U || posts > 64U) {
        // 64 is MAX_COMPLETIONS_PER_TASK: one deferred record per post.
        return;
    }

    __gm__ float *send = tensor_data<float>(send_tensor);
    __gm__ float *recv = tensor_data<float>(recv_tensor);
    __gm__ float *tput = tensor_data<float>(tput_tensor);
    const uint32_t peer = peer_rank(comm_ctx);
    AsyncCtx async_ctx = get_async_ctx(args);

    const uint32_t base_post = task_index * posts;
    for (uint32_t i = 0; i < posts; ++i) {
        const uint32_t post = base_post + i;
        const auto submit_status = submit_post(async_ctx, comm_ctx, send, recv, tput, peer, post);
        if (submit_status != pto2::rdma_backend::RdmaSubmitStatus::OK) {
            return;
        }
    }

}
