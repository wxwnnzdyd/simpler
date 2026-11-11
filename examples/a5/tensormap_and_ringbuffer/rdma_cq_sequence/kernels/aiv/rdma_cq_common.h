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

// Shared pieces of the RDMA CQ sequence cases: every post is one 1 WQE transfer,
// so a run's post index is its CQ sequence and the whole run is one monotone
// sequence across the ring laps.

#pragma once

#include <cstdint>

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include <pto/pto-inst.hpp>

#include "pto/comm/comm_types.hpp"
#include "pto/comm/pto_comm_inst.hpp"
#include "pto/comm/async/rdma/rdma_async_intrin.hpp"
#ifdef PTO_RDMA_SUPPORTED
#include "pto/comm/async/rdma/backends/hns_1825/hns_1825_backend.hpp"
#endif

#include "backend/rdma/rdma_completion_kernel.h"
#include "platform_comm/comm_context.h"
#include "tensor.h"

namespace rdma_cq_sequence {

using ShapeDyn = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
using GlobalFloat = pto::GlobalTensor<float, ShapeDyn, StrideDyn>;

#ifdef PTO_RDMA_SUPPORTED
constexpr uint32_t kRdmaScratchBytes = pto::comm::rdma::kRdmaScratchBytes;
#else
constexpr uint32_t kRdmaScratchBytes = pto::comm::sdma::UB_ALIGN_SIZE;
#endif
using RdmaScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, kRdmaScratchBytes>;

constexpr uint32_t kRdmaScratchOffset = 0;
// Slice size of one post. Any size below the 256 MB WQE limit is a single WQE,
// so a small slice keeps the buffers modest while the post index still equals
// the CQ sequence.
constexpr uint32_t kSliceFloats = 64;

// Layout of the summary tensor the summary task writes.
enum SummaryWord : uint32_t {
    kSummaryReserve = 0,
    kSummaryReady = 1,
    kSummaryPosted = 2,
    kSummaryCompleted = 3,
    kSummaryAcked = 4,
    kSummaryQpNum = 5,
    kSummaryCqDepth = 6,
    kSummaryWords = 7,
};

enum class Status : int32_t {
    kOk = 0,
    kUnsupported = 1,
    kInvalidComm = 2,
    kSubmitFailed = 3,
    kRegisterFailed = 4,
    kBadArguments = 5,
    kAckedMismatch = 6,
    kTgetMismatch = 100,
    kTputMismatch = 200,
    kSignalMismatch = 300,
};

// Each task owns four words: [0] status, [1]/[2] details, [3] a marker written
// before anything can fail. A slice still holding the host sentinel means the
// kernel never reached its body, which is otherwise indistinguishable from a
// kernel that ran and wrote nothing.
constexpr int32_t kStatusWordsPerTask = 4;
constexpr int32_t kKernelEntered = 0x5a5a5a5a;

// A task that owns no status slice passes nullptr: the summary is the only
// writer of the host tensor, so post tasks report failures by not posting.
inline __aicore__ void mark_entered(__gm__ int32_t *status) {
    if (status == nullptr) {
        return;
    }
    status[3] = kKernelEntered;
    pto2::detail::defer_flush_range(status, kStatusWordsPerTask * sizeof(int32_t));
}

inline __aicore__ void set_status(__gm__ int32_t *status, Status code, int32_t detail0 = 0, int32_t detail1 = 0) {
    if (status == nullptr) {
        return;
    }
    status[0] = static_cast<int32_t>(code);
    status[1] = detail0;
    status[2] = detail1;
    pto2::detail::defer_flush_range(status, kStatusWordsPerTask * sizeof(int32_t));
}

template <typename T>
inline __aicore__ __gm__ T *tensor_data(__gm__ Tensor *tensor) {
    return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) + tensor->start_offset;
}

inline __aicore__ bool validate_comm(__gm__ CommContext *comm_ctx, __gm__ int32_t *status) {
#ifndef PTO_RDMA_SUPPORTED
    (void)comm_ctx;
    set_status(status, Status::kUnsupported);
    return false;
#else
    if (comm_ctx == nullptr || comm_ctx->rankNum != 2 || comm_ctx->rankId >= comm_ctx->rankNum ||
        comm_ctx->workSpace == 0 || comm_ctx->windowsIn[comm_ctx->rankId] == 0) {
        set_status(status, Status::kInvalidComm);
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

inline __aicore__ RdmaScratchTile rdma_scratch_tile() {
    RdmaScratchTile scratch;
    TASSIGN(scratch, kRdmaScratchOffset);
    return scratch;
}

inline __aicore__ uint64_t remote_base(__gm__ CommContext *comm_ctx, uint32_t peer) {
#ifdef PTO_RDMA_SUPPORTED
    return pto2::rdma_backend::peer_mr_base_addr(reinterpret_cast<__gm__ uint8_t *>(comm_ctx->workSpace), peer);
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

inline __aicore__ __gm__ float *remote_slice(
    __gm__ CommContext *comm_ctx, uint32_t peer, __gm__ float *local_ptr, uint32_t post
) {
    return reinterpret_cast<__gm__ float *>(
        remote_base(comm_ctx, peer) + local_offset(comm_ctx, local_ptr) +
        static_cast<uint64_t>(post) * kSliceFloats * sizeof(float)
    );
}

// The host writes send[post] = rank * kRankStride + post, so a transfer that
// lands on the wrong post index is caught by the value check instead of passing
// as "some data arrived".
constexpr float kRankStride = 1000000.0f;

inline __aicore__ float expected_value(uint32_t source_rank, uint32_t post) {
    return static_cast<float>(source_rank) * kRankStride + static_cast<float>(post);
}

// One 1 WQE transfer, alternating direction so both opcodes share the CQ and a
// single run covers TPUT and TGET on the same sequence. Even posts push the
// local send slice into the peer's tput buffer; odd posts pull the peer's send
// slice into the local recv buffer. Both are checked by the summary task, which
// expects the peer's rank value in both buffers.
inline __aicore__ pto2::rdma_backend::RdmaSubmitStatus submit_post(
    AsyncCtx &async_ctx, __gm__ CommContext *comm_ctx, __gm__ float *local_send, __gm__ float *local_recv,
    __gm__ float *local_tput, uint32_t peer, uint32_t post
) {
    const uint32_t sync_id = post & 1U;
    auto scratch = rdma_scratch_tile();
    __gm__ uint8_t *workspace = reinterpret_cast<__gm__ uint8_t *>(comm_ctx->workSpace);
    __gm__ float *send_slice = local_send + static_cast<uint64_t>(post) * kSliceFloats;
    __gm__ float *recv_slice = local_recv + static_cast<uint64_t>(post) * kSliceFloats;
    if ((post & 1U) == 0U) {
        return pto2::rdma_backend::submit_rdma_request_status(
            async_ctx, RdmaTput(
                           global_float(remote_slice(comm_ctx, peer, local_tput, post), kSliceFloats),
                           global_float(send_slice, kSliceFloats), scratch, workspace, peer, comm_ctx->rankId,
                           sync_id
                       )
        );
    }
    return pto2::rdma_backend::submit_rdma_request_status(
        async_ctx, RdmaTget(
                       global_float(recv_slice, kSliceFloats),
                       global_float(remote_slice(comm_ctx, peer, local_send, post), kSliceFloats), scratch, workspace,
                       peer, comm_ctx->rankId, sync_id
                   )
    );
}

}  // namespace rdma_cq_sequence
