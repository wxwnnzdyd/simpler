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

#include <pto/pto-inst.hpp>
#include "pto/common/pto_tile.hpp"
#include "pto/comm/comm_types.hpp"
#include "pto/comm/pto_comm_inst.hpp"
#ifdef PTO_URMA_SUPPORTED
#include "pto/comm/async/urma/urma_async_intrin.hpp"
#endif

#include "platform_comm/comm_context.h"
#include "tensor.h"

namespace {

constexpr int kMaxSupportedRanks = 16;

enum class UrmaRealStatus : int32_t {
    kOk = 0,
    kUnsupported = 1,
    kInvalidRankCount = 2,
    kMissingWorkspace = 3,
    kInvalidElementCount = 4,
    kTgetWaitFailed = 10,
    kTputWaitFailed = 20,
    kTgetPostBegin = 30,
    kTgetPostDone = 31,
    kProbeWorkspaceInfo = 32,
    kProbeWqCtx = 33,
    kProbeQueueIndexBegin = 34,
    kProbeQueueIndexDone = 35,
    kProbeWqeWriteBegin = 36,
    kProbeWqeWriteDone = 37,
    kProbeQueueIndexStoreBegin = 38,
    kProbeQueueIndexStoreDone = 39,
    kTputPostBegin = 40,
    kTputPostDone = 41,
    kProbeFillSqeDone = 42,
    kProbeEidReadDone = 43,
    kProbeRawSqeHeaderDone = 44,
    kProbeWqeOriginalStoreDone = 45,
    kProbeFullSqeDone = 46,
    kProbeDcciDone = 47,
    kTgetMismatch = 100,
    kTputMismatch = 200,
};

constexpr uint32_t kMaxUrmaPollIters = 1000000;

template <typename T>
AICORE inline __gm__ T *CommRemotePtr(__gm__ CommContext *ctx, __gm__ T *local_ptr, int peer) {
    uint64_t local_base = ctx->windowsIn[ctx->rankId];
    uint64_t offset = reinterpret_cast<uint64_t>(local_ptr) - local_base;
    return reinterpret_cast<__gm__ T *>(ctx->windowsIn[peer] + offset);
}

AICORE inline void SetStatus(__gm__ int32_t *status, UrmaRealStatus code, int32_t detail0 = 0, int32_t detail1 = 0) {
    status[0] = static_cast<int32_t>(code);
    status[1] = detail0;
    status[2] = detail1;
}

template <typename Event, typename Session>
AICORE inline bool
WaitUrmaBounded(const Event &event, const Session &session, __gm__ int32_t *status, UrmaRealStatus timeout_code) {
    for (uint32_t i = 0; i < kMaxUrmaPollIters; ++i) {
        if (event.Test(session)) {
            return true;
        }
    }
    SetStatus(status, timeout_code, static_cast<int32_t>(kMaxUrmaPollIters));
    return false;
}

AICORE inline void DeviceBarrier(__gm__ CommContext *ctx, __gm__ int32_t *signal_base, int my_rank, int nranks) {
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) continue;
        __gm__ int32_t *remote_signal = CommRemotePtr(ctx, signal_base + my_rank, peer);
        pto::comm::Signal signal(remote_signal);
        pto::comm::TNOTIFY(signal, static_cast<int32_t>(1), pto::comm::NotifyOp::AtomicAdd);
    }

    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) continue;
        pto::comm::Signal signal(signal_base + peer);
        pto::comm::TWAIT(signal, static_cast<int32_t>(1), pto::comm::WaitCmp::GE);
    }
    pipe_barrier(PIPE_ALL);
}

}  // namespace

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *send_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *tget_recv_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *tput_recv_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *signal_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ Tensor *status_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ CommContext *comm_ctx = reinterpret_cast<__gm__ CommContext *>(args[5]);
    uint32_t elem_count = static_cast<uint32_t>(args[6]);

    __gm__ float *send = reinterpret_cast<__gm__ float *>(send_tensor->buffer.addr) + send_tensor->start_offset;
    __gm__ float *tget_recv =
        reinterpret_cast<__gm__ float *>(tget_recv_tensor->buffer.addr) + tget_recv_tensor->start_offset;
    __gm__ float *tput_recv =
        reinterpret_cast<__gm__ float *>(tput_recv_tensor->buffer.addr) + tput_recv_tensor->start_offset;
    __gm__ int32_t *signal =
        reinterpret_cast<__gm__ int32_t *>(signal_tensor->buffer.addr) + signal_tensor->start_offset;
    __gm__ int32_t *status =
        reinterpret_cast<__gm__ int32_t *>(status_tensor->buffer.addr) + status_tensor->start_offset;

    SetStatus(status, UrmaRealStatus::kOk);

    int nranks = static_cast<int>(comm_ctx->rankNum);
    int my_rank = static_cast<int>(comm_ctx->rankId);
    if (nranks != 2 || nranks > kMaxSupportedRanks || my_rank < 0 || my_rank >= nranks) {
        SetStatus(status, UrmaRealStatus::kInvalidRankCount, my_rank, nranks);
        return;
    }
    if (comm_ctx->workSpace == 0) {
        SetStatus(status, UrmaRealStatus::kMissingWorkspace);
        return;
    }
    if (elem_count == 0) {
        SetStatus(status, UrmaRealStatus::kInvalidElementCount);
        return;
    }

    int peer = 1 - my_rank;

#ifdef PTO_URMA_SUPPORTED
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;

    ShapeDyn shape(1, 1, 1, 1, static_cast<int>(elem_count));
    StrideDyn stride(elem_count, elem_count, elem_count, elem_count, 1);
    __gm__ uint8_t *workspace = reinterpret_cast<__gm__ uint8_t *>(comm_ctx->workSpace);

    uint64_t local_window_base = comm_ctx->windowsIn[comm_ctx->rankId];
    uint64_t peer_base = pto::comm::urma::UrmaPeerMrBaseAddr(workspace, static_cast<uint32_t>(peer));
    uint64_t send_offset = reinterpret_cast<uint64_t>(send) - local_window_base;
    uint64_t tput_slot_offset =
        (reinterpret_cast<uint64_t>(tput_recv) - local_window_base) + static_cast<uint64_t>(my_rank) * elem_count * 4u;

    __gm__ float *remote_send = reinterpret_cast<__gm__ float *>(peer_base + send_offset);
    __gm__ float *remote_tput_slot = reinterpret_cast<__gm__ float *>(peer_base + tput_slot_offset);

    pto::comm::AsyncSession tget_session;
    pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(workspace, static_cast<uint32_t>(peer), tget_session);
    Global local_tget_g(tget_recv, shape, stride);
    Global remote_send_g(remote_send, shape, stride);
    SetStatus(status, UrmaRealStatus::kTgetPostBegin, my_rank, peer);
    __gm__ pto::comm::urma::UrmaInfo *urma_info = reinterpret_cast<__gm__ pto::comm::urma::UrmaInfo *>(workspace);
    uint32_t qp_num = urma_info->qpNum;
    uint64_t sq_ptr = urma_info->sqPtr;
    SetStatus(status, UrmaRealStatus::kProbeWorkspaceInfo, static_cast<int32_t>(qp_num), peer);
    __gm__ pto::comm::urma::UrmaWQCtx *wq_ctx = reinterpret_cast<__gm__ pto::comm::urma::UrmaWQCtx *>(
        sq_ptr + (static_cast<uint32_t>(peer) * qp_num) * sizeof(pto::comm::urma::UrmaWQCtx)
    );
    uint64_t head_addr = wq_ctx->headAddr;
    uint64_t tail_addr = wq_ctx->tailAddr;
    SetStatus(
        status, UrmaRealStatus::kProbeWqCtx, static_cast<int32_t>(wq_ctx->depth),
        static_cast<int32_t>(wq_ctx->wqeShiftSize)
    );
    SetStatus(
        status, UrmaRealStatus::kProbeQueueIndexBegin, static_cast<int32_t>(head_addr & 0xFFFFFFFFu),
        static_cast<int32_t>(tail_addr & 0xFFFFFFFFu)
    );
    uint32_t head = *reinterpret_cast<__gm__ uint32_t *>(head_addr);
    uint32_t tail = *reinterpret_cast<__gm__ uint32_t *>(tail_addr);
    SetStatus(status, UrmaRealStatus::kProbeQueueIndexDone, static_cast<int32_t>(head), static_cast<int32_t>(tail));
    uint32_t wqe_size = 1U << wq_ctx->wqeShiftSize;
    uint64_t wqe_addr = wq_ctx->bufAddr + static_cast<uint64_t>(wqe_size) * (head % wq_ctx->depth);
    SetStatus(
        status, UrmaRealStatus::kProbeWqeWriteBegin, static_cast<int32_t>(wqe_addr & 0xFFFFFFFFu),
        static_cast<int32_t>(wqe_size)
    );
    __gm__ uint64_t *wqe_word0 = reinterpret_cast<__gm__ uint64_t *>(wqe_addr);
    uint64_t old_wqe_word0 = *wqe_word0;
    SetStatus(
        status, UrmaRealStatus::kProbeWqeWriteDone, static_cast<int32_t>(old_wqe_word0 & 0xFFFFFFFFu),
        static_cast<int32_t>(head)
    );
    return;
    *wqe_word0 = old_wqe_word0 ^ 1ULL;
    *wqe_word0 = old_wqe_word0;
    uint64_t second_wqe_word0 = *wqe_word0;
    SetStatus(
        status, UrmaRealStatus::kProbeQueueIndexStoreBegin, static_cast<int32_t>(head), static_cast<int32_t>(tail)
    );
    *reinterpret_cast<__gm__ uint32_t *>(head_addr) = head;
    *reinterpret_cast<__gm__ uint32_t *>(tail_addr) = tail;
    SetStatus(
        status, UrmaRealStatus::kProbeQueueIndexStoreDone, static_cast<int32_t>(head), static_cast<int32_t>(tail)
    );
    __gm__ pto::comm::urma::UrmaMemInfo *remote_mem =
        reinterpret_cast<__gm__ pto::comm::urma::UrmaMemInfo *>(urma_info->memPtr) + peer;
    SetStatus(
        status, UrmaRealStatus::kProbeFillSqeDone, static_cast<int32_t>(remote_mem->tpn),
        static_cast<int32_t>(remote_mem->tid)
    );
    __gm__ uint64_t *remote_eid_probe = reinterpret_cast<__gm__ uint64_t *>(remote_mem->eidAddr);
    uint64_t eid0 = remote_eid_probe[0];
    uint64_t eid1 = remote_eid_probe[1];
    SetStatus(
        status, UrmaRealStatus::kProbeEidReadDone, static_cast<int32_t>(eid0 & 0xFFFFFFFFu),
        static_cast<int32_t>(eid1 & 0xFFFFFFFFu)
    );
    uint32_t sqe_dw0_without_owner = static_cast<uint32_t>(head % wq_ctx->depth) | (0x20U << 16) |
                                     ((remote_mem->tokenValueValid ? 1U : 0U) << 28) |
                                     ((remote_mem->rmtJettyType & 0x3U) << 29);
    SetStatus(
        status, UrmaRealStatus::kProbeRawSqeHeaderDone, static_cast<int32_t>(sqe_dw0_without_owner),
        static_cast<int32_t>(head)
    );
    uint64_t late_wqe_word0 = second_wqe_word0;
    uint64_t remote_addr_value = reinterpret_cast<uint64_t>(remote_send);
    __gm__ uint8_t *sqe_bytes = reinterpret_cast<__gm__ uint8_t *>(wqe_addr);
    __gm__ uint32_t *sqe_dw = reinterpret_cast<__gm__ uint32_t *>(sqe_bytes);
    uint32_t old_sqe_dw0 = static_cast<uint32_t>(late_wqe_word0 & 0xFFFFFFFFu);
    sqe_dw[0] = old_sqe_dw0;
    uint32_t sqe_owner = ((head & wq_ctx->depth) == 0U ? 1U : 0U);
    uint32_t sqe_dw0 = sqe_dw0_without_owner | (sqe_owner << 31);
    uint32_t sqe_dw1 = (static_cast<uint32_t>(remote_mem->targetHint) & 0xFFU) |
                       (static_cast<uint32_t>(pto::comm::urma::UrmaOpcode::READ) << 8);
    uint32_t sqe_dw2 = (remote_mem->tpn & 0xFFFFFFU) | (1U << 24);
    uint32_t sqe_dw3 = remote_mem->tid & 0xFFFFFU;
    sqe_dw[0] = sqe_dw0;
    sqe_dw[1] = sqe_dw1;
    sqe_dw[2] = sqe_dw2;
    sqe_dw[3] = sqe_dw3;
    sqe_dw[4] = static_cast<uint32_t>(eid0 & 0xFFFFFFFFU);
    sqe_dw[5] = static_cast<uint32_t>((eid0 >> 32) & 0xFFFFFFFFU);
    sqe_dw[6] = static_cast<uint32_t>(eid1 & 0xFFFFFFFFU);
    sqe_dw[7] = static_cast<uint32_t>((eid1 >> 32) & 0xFFFFFFFFU);
    sqe_dw[8] = remote_mem->rmtTokenValue;
    sqe_dw[9] = 0;
    sqe_dw[10] = static_cast<uint32_t>(remote_addr_value & 0xFFFFFFFFU);
    sqe_dw[11] = static_cast<uint32_t>((remote_addr_value >> 32) & 0xFFFFFFFFU);
    __gm__ pto::comm::urma::UrmaSgeCtx *sge =
        reinterpret_cast<__gm__ pto::comm::urma::UrmaSgeCtx *>(sqe_bytes + pto::comm::urma::kUrmaSqeSizeBytes);
    sge->len = elem_count * 4U;
    sge->tokenId = urma_info->localTokenId;
    sge->va = reinterpret_cast<uint64_t>(tget_recv);
    SetStatus(
        status, UrmaRealStatus::kProbeFullSqeDone, static_cast<int32_t>(remote_mem->tpn),
        static_cast<int32_t>(remote_mem->tid)
    );
    return;
    pipe_barrier(PIPE_ALL);
    pto::comm::urma::DcciCachelines(sqe_bytes, pto::comm::urma::kUrmaSqeSizeBytes + pto::comm::urma::kUrmaSgeSizeBytes);
    pipe_barrier(PIPE_ALL);
    SetStatus(status, UrmaRealStatus::kProbeDcciDone, static_cast<int32_t>(head), static_cast<int32_t>(tail));
    return;
    auto tget_event = pto::comm::TGET_ASYNC<pto::comm::DmaEngine::URMA>(local_tget_g, remote_send_g, tget_session);
    SetStatus(status, UrmaRealStatus::kTgetPostDone, static_cast<int32_t>(tget_event.handle & 0xFFFFFFFFu), peer);
    if (!WaitUrmaBounded(tget_event, tget_session, status, UrmaRealStatus::kTgetWaitFailed)) {
        return;
    }

    pto::comm::AsyncSession tput_session;
    pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(workspace, static_cast<uint32_t>(peer), tput_session);
    Global remote_tput_g(remote_tput_slot, shape, stride);
    Global local_send_g(send, shape, stride);
    SetStatus(status, UrmaRealStatus::kTputPostBegin, my_rank, peer);
    auto tput_event = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(remote_tput_g, local_send_g, tput_session);
    SetStatus(status, UrmaRealStatus::kTputPostDone, static_cast<int32_t>(tput_event.handle & 0xFFFFFFFFu), peer);
    if (!WaitUrmaBounded(tput_event, tput_session, status, UrmaRealStatus::kTputWaitFailed)) {
        return;
    }

    DeviceBarrier(comm_ctx, signal, my_rank, nranks);

    for (uint32_t i = 0; i < elem_count; ++i) {
        float expected = static_cast<float>(peer * 100000 + static_cast<int>(i));
        if (tget_recv[i] != expected) {
            SetStatus(status, UrmaRealStatus::kTgetMismatch, static_cast<int32_t>(i), peer);
            return;
        }
    }

    __gm__ float *incoming_tput = tput_recv + static_cast<uint64_t>(peer) * elem_count;
    for (uint32_t i = 0; i < elem_count; ++i) {
        float expected = static_cast<float>(peer * 100000 + static_cast<int>(i));
        if (incoming_tput[i] != expected) {
            SetStatus(status, UrmaRealStatus::kTputMismatch, static_cast<int32_t>(i), peer);
            return;
        }
    }
    SetStatus(status, UrmaRealStatus::kOk, static_cast<int32_t>(elem_count), peer);
#else
    (void)send;
    (void)tget_recv;
    (void)tput_recv;
    (void)signal;
    (void)elem_count;
    (void)peer;
    SetStatus(status, UrmaRealStatus::kUnsupported);
#endif
}
