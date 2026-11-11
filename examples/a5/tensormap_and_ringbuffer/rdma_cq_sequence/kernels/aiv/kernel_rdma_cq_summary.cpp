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

// Runs after the whole sequence and answers two questions the per-task kernels
// cannot: did every completion get confirmed, and did the payloads land in the
// post they belong to?
//
// The counters are read straight out of the queue state, which is pto-isa
// internal. A test is the right place for that: asserting "the run finished" is
// too weak, because a deferred producer would also finish after silently
// retiring nothing.

#include "rdma_cq_common.h"

namespace {

using namespace rdma_cq_sequence;

constexpr int kArgRecv = 0;
constexpr int kArgTput = 1;
constexpr int kArgSignal = 2;
constexpr int kArgSummary = 3;
constexpr int kArgStatus = 4;
constexpr int kArgCommCtx = 5;
constexpr int kArgTotalPosts = 6;
constexpr int kArgMode = 7;
constexpr int kArgStatusIndex = 8;

// 0: chain / parallel posts (TPUT on even posts, TGET on odd ones).
// 1: notify posts (payload + signal per post).
constexpr uint32_t kModePosts = 0;
constexpr uint32_t kModeNotify = 1;

constexpr uint32_t kSummarySyncId = 3;

#ifdef PTO_RDMA_SUPPORTED
inline __aicore__ bool verify_slice(
    __gm__ int32_t *status, __gm__ const float *buffer, uint32_t post, uint32_t source_rank
) {
    const float expected = expected_value(source_rank, post);
    const __gm__ float *got = buffer + static_cast<uint64_t>(post) * kSliceFloats;
    for (uint32_t e = 0; e < kSliceFloats; ++e) {
        if (got[e] != expected) {
            set_status(status, Status::kTgetMismatch, static_cast<int32_t>(post), static_cast<int32_t>(e));
            return false;
        }
    }
    return true;
}
#endif

}  // namespace

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *recv_tensor = reinterpret_cast<__gm__ Tensor *>(args[kArgRecv]);
    __gm__ Tensor *tput_tensor = reinterpret_cast<__gm__ Tensor *>(args[kArgTput]);
    __gm__ Tensor *signal_tensor = reinterpret_cast<__gm__ Tensor *>(args[kArgSignal]);
    __gm__ Tensor *summary_tensor = reinterpret_cast<__gm__ Tensor *>(args[kArgSummary]);
    __gm__ Tensor *status_tensor = reinterpret_cast<__gm__ Tensor *>(args[kArgStatus]);
    __gm__ CommContext *comm_ctx = reinterpret_cast<__gm__ CommContext *>(args[kArgCommCtx]);
    const uint32_t total_posts = static_cast<uint32_t>(args[kArgTotalPosts]);
    const uint32_t mode = static_cast<uint32_t>(args[kArgMode]);
    const uint32_t status_index = static_cast<uint32_t>(args[kArgStatusIndex]);

    __gm__ int32_t *status = tensor_data<int32_t>(status_tensor) + static_cast<uint64_t>(status_index) * 3U;
    if (!validate_comm(comm_ctx, status)) {
        return;
    }
#ifndef PTO_RDMA_SUPPORTED
    set_status(status, Status::kUnsupported);
    return;
#else
    const uint32_t peer = peer_rank(comm_ctx);
    const uint32_t qp = 0;
    __gm__ pto::comm::rdma::RdmaInfo *info =
        reinterpret_cast<__gm__ pto::comm::rdma::RdmaInfo *>(comm_ctx->workSpace);
    const uint64_t index = static_cast<uint64_t>(peer) * info->qpNum + qp;
    __gm__ pto::comm::rdma::hns_1825::RoceSqCtx *sq =
        reinterpret_cast<__gm__ pto::comm::rdma::hns_1825::RoceSqCtx *>(info->sqPtr) + index;
    __gm__ pto::comm::rdma::hns_1825::RoceCqCtx *cq =
        reinterpret_cast<__gm__ pto::comm::rdma::hns_1825::RoceCqCtx *>(info->scqPtr) + index;
    __gm__ pto::comm::rdma::hns_1825::QueueState *state =
        pto::comm::rdma::hns_1825::detail::GetQueueState(sq);

    // A deferred producer only pushes the consumer cursor when it runs out of
    // room, so the tail of the sequence may still be unpublished here. Publish
    // once so ackedHead reflects every confirmation the consumer wrote.
    auto scratch = rdma_scratch_tile();
    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::RDMA>(
            scratch, reinterpret_cast<__gm__ uint8_t *>(comm_ctx->workSpace), comm_ctx->rankId, session,
            kSummarySyncId, qp, /*deferred=*/true
        )) {
        set_status(status, Status::kBadArguments, -2);
        return;
    }
    // One notify owns two CQ entries (payload + signal); everything else owns one.
    const uint64_t expected_acked =
        mode == kModeNotify ? static_cast<uint64_t>(total_posts) * 2U : static_cast<uint64_t>(total_posts);

    // PublishAckedProgress bails out when another consumer holds the CQ lock, so
    // retry until the prefix reflects every confirmation. The run is otherwise
    // over, which makes the retry cheap and bounded.
    for (uint32_t attempt = 0; attempt < 256U; ++attempt) {
        pto::comm::rdma::hns_1825::detail::PublishAckedProgress(
            info, peer, qp, session.tmpBufAddr, kSummarySyncId
        );
        pto::comm::rdma::hns_1825::detail::QueueFence();
        if (static_cast<uint64_t>(
                pto::comm::rdma::hns_1825::detail::AtomicLoad(&state->ackedHead)
            ) >= expected_acked) {
            break;
        }
    }

    __gm__ int64_t *summary = tensor_data<int64_t>(summary_tensor);
    summary[kSummaryReserve] = static_cast<int64_t>(state->reserveHead);
    summary[kSummaryReady] = static_cast<int64_t>(state->readyHead);
    summary[kSummaryPosted] = static_cast<int64_t>(state->postedHead);
    summary[kSummaryCompleted] = static_cast<int64_t>(state->completedHead);
    summary[kSummaryAcked] = static_cast<int64_t>(state->ackedHead);
    summary[kSummaryQpNum] = static_cast<int64_t>(info->qpNum);
    summary[kSummaryCqDepth] = static_cast<int64_t>(cq->depth);
    pto2::detail::defer_flush_range(summary, kSummaryWords * sizeof(int64_t));

    if (static_cast<uint64_t>(state->postedHead) != expected_acked ||
        static_cast<uint64_t>(state->ackedHead) != expected_acked) {
        set_status(
            status, Status::kAckedMismatch, static_cast<int32_t>(state->postedHead),
            static_cast<int32_t>(state->ackedHead)
        );
        return;
    }

    // Every completion is confirmed, so every payload has landed: check that each
    // post wrote its own index instead of some other post's value.
    if (mode == kModeNotify) {
        __gm__ const int32_t *signal = tensor_data<int32_t>(signal_tensor);
        for (uint32_t post = 0; post < total_posts; ++post) {
            if (signal[post] != static_cast<int32_t>(post)) {
                set_status(status, Status::kSignalMismatch, static_cast<int32_t>(post), signal[post]);
                return;
            }
        }
    } else {
        __gm__ const float *recv = tensor_data<float>(recv_tensor);
        __gm__ const float *tput = tensor_data<float>(tput_tensor);
        for (uint32_t post = 0; post < total_posts; ++post) {
            // Even posts were pushed by the peer into our tput buffer; odd ones
            // were pulled from the peer's send buffer into our recv buffer.
            if ((post & 1U) == 0U) {
                if (!verify_slice(status, tput, post, peer)) {
                    return;
                }
            } else if (!verify_slice(status, recv, post, peer)) {
                return;
            }
        }
    }

    set_status(status, Status::kOk, static_cast<int32_t>(total_posts));
#endif
}
