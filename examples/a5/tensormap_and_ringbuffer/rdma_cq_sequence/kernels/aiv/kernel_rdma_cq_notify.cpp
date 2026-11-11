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

// Notify posts: each one is payload + fenced signal, i.e. two signalled WQEs, so
// one event owns two CQ slots and its record carries (ack_begin, ack_sequence].
// The post goes through pto-isa's own TPUT_ASYNC_NOTIFY (simpler's RDMA
// descriptor only exposes TGET/TPUT) and is registered as a deferred completion
// here, which is the mixed-API case: native submit, runtime-owned completion.

#include "rdma_cq_common.h"

namespace {

using namespace rdma_cq_sequence;

constexpr int kArgSend = 0;
constexpr int kArgTput = 1;
constexpr int kArgSignal = 2;
constexpr int kArgStatus = 3;
constexpr int kArgCommCtx = 4;
constexpr int kArgPosts = 5;
constexpr int kArgTaskIndex = 6;

constexpr uint32_t kNotifySyncId = 2;

}  // namespace

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *send_tensor = reinterpret_cast<__gm__ Tensor *>(args[kArgSend]);
    __gm__ Tensor *tput_tensor = reinterpret_cast<__gm__ Tensor *>(args[kArgTput]);
    __gm__ Tensor *signal_tensor = reinterpret_cast<__gm__ Tensor *>(args[kArgSignal]);
    __gm__ Tensor *status_tensor = reinterpret_cast<__gm__ Tensor *>(args[kArgStatus]);
    __gm__ CommContext *comm_ctx = reinterpret_cast<__gm__ CommContext *>(args[kArgCommCtx]);
    const uint32_t posts = static_cast<uint32_t>(args[kArgPosts]);
    const uint32_t task_index = static_cast<uint32_t>(args[kArgTaskIndex]);

    __gm__ int32_t *status = tensor_data<int32_t>(status_tensor) + static_cast<uint64_t>(task_index) * 3U;
    if (!validate_comm(comm_ctx, status)) {
        return;
    }
    if (posts == 0U || posts > 64U) {
        set_status(status, Status::kBadArguments, static_cast<int32_t>(posts));
        return;
    }

    __gm__ float *send = tensor_data<float>(send_tensor);
    __gm__ float *tput = tensor_data<float>(tput_tensor);
    __gm__ int32_t *signal = tensor_data<int32_t>(signal_tensor);
    const uint32_t peer = peer_rank(comm_ctx);
    AsyncCtx async_ctx = get_async_ctx(args);

    auto scratch = rdma_scratch_tile();
    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::RDMA>(
            scratch, reinterpret_cast<__gm__ uint8_t *>(comm_ctx->workSpace), comm_ctx->rankId, session,
            kNotifySyncId, 0, /*deferred=*/true
        )) {
        set_status(status, Status::kBadArguments, -1);
        return;
    }

    const uint32_t base_post = task_index * posts;
    for (uint32_t i = 0; i < posts; ++i) {
        const uint32_t post = base_post + i;
        __gm__ float *send_slice = send + static_cast<uint64_t>(post) * kSliceFloats;
        // TPUT_ASYNC_NOTIFY takes non-const lvalue references, so the tensors have
        // to be named locals rather than temporaries.
        auto remote_tput_slice = global_float(remote_slice(comm_ctx, peer, tput, post), kSliceFloats);
        auto local_send_slice = global_float(send_slice, kSliceFloats);
        pto::comm::Signal remote_signal(signal + post);
        auto event = pto::comm::TPUT_ASYNC_NOTIFY<pto::comm::DmaEngine::RDMA>(
            remote_tput_slice, local_send_slice, remote_signal, static_cast<int32_t>(post),
            pto::comm::NotifyOp::Set, session, peer
        );
        const auto registered = pto2::detail::register_rdma_async_event_status(async_ctx, event, session);
        if (registered != pto2::detail::RdmaEventRegistrationResult::OK) {
            set_status(
                status, Status::kRegisterFailed, static_cast<int32_t>(registered), static_cast<int32_t>(post)
            );
            return;
        }
    }

    set_status(status, Status::kOk, static_cast<int32_t>(posts));
}
