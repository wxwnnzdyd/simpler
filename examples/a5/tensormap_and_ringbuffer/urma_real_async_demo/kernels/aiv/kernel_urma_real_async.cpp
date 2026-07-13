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
    kTputPostBegin = 40,
    kTputPostDone = 41,
    kBarrierWaitFailed = 50,
    kUnsafeWqeAccess = 60,
    kTgetMismatch = 100,
    kTputMismatch = 200,
};

enum class ProbeStage : uint32_t {
    kFull = 0,
    kWorkspace = 1,
    kBuildSession = 2,
    kWorkspaceInfo = 3,
    kWqCtx = 4,
    kQueueIndexRead = 5,
    kWqeRead = 6,
    kWqeWriteRestore = 7,
    kRemoteMem = 8,
    kEidRead = 9,
    kTgetPost = 10,
    kTgetTestOnce = 11,
    kTputPost = 12,
    kTputTestOnce = 13,
    kQueueIndexLdDev = 14,
    kWqeAddr = 15,
    kWqeFirstStore = 16,
    kWqeFirstStDev = 17,
    kWqeMteStore = 18,
};

constexpr uint32_t kMaxUrmaPollIters = 10000000;
constexpr uint32_t kMaxBarrierPollIters = 10000000;

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
            return event.Wait(session);
        }
    }
    SetStatus(
        status, timeout_code, static_cast<int32_t>(event.handle & 0xFFFFFFFFu), static_cast<int32_t>(event.handle >> 32)
    );
    return false;
}

AICORE inline bool DeviceBarrierBounded(__gm__ CommContext *ctx, __gm__ int32_t *signal_base, int my_rank, int nranks) {
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) continue;
        __gm__ int32_t *remote_signal = CommRemotePtr(ctx, signal_base + my_rank, peer);
        pto::comm::Signal signal(remote_signal);
        pto::comm::TNOTIFY(signal, static_cast<int32_t>(1), pto::comm::NotifyOp::AtomicAdd);
    }

    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) continue;
        __gm__ volatile int32_t *peer_signal = signal_base + peer;
        bool observed = false;
        for (uint32_t i = 0; i < kMaxBarrierPollIters; ++i) {
            if (*peer_signal >= 1) {
                observed = true;
                break;
            }
        }
        if (!observed) {
            return false;
        }
    }
    pipe_barrier(PIPE_ALL);
    return true;
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
    uint32_t probe_stage = static_cast<uint32_t>(args[7]);

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
    __gm__ uint8_t *workspace = reinterpret_cast<__gm__ uint8_t *>(comm_ctx->workSpace);
    uint64_t transfer_bytes = static_cast<uint64_t>(elem_count) * sizeof(float);

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using GlobalFloat = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
    ShapeDyn shape(1, 1, 1, 1, elem_count);
    StrideDyn stride(elem_count, elem_count, elem_count, elem_count, 1);

    uint64_t local_window_base = comm_ctx->windowsIn[comm_ctx->rankId];
    uint64_t peer_base = pto::comm::urma::UrmaPeerMrBaseAddr(workspace, static_cast<uint32_t>(peer));
    uint64_t send_offset = reinterpret_cast<uint64_t>(send) - local_window_base;
    uint64_t tput_slot_offset =
        (reinterpret_cast<uint64_t>(tput_recv) - local_window_base) + static_cast<uint64_t>(my_rank) * elem_count * 4u;

    __gm__ float *remote_send = reinterpret_cast<__gm__ float *>(peer_base + send_offset);
    __gm__ float *remote_tput_slot = reinterpret_cast<__gm__ float *>(peer_base + tput_slot_offset);
    GlobalFloat local_send_g(send, shape, stride);
    GlobalFloat local_tget_recv_g(tget_recv, shape, stride);
    GlobalFloat remote_send_g(remote_send, shape, stride);
    GlobalFloat remote_tput_slot_g(remote_tput_slot, shape, stride);

    (void)transfer_bytes;
    if (probe_stage == static_cast<uint32_t>(ProbeStage::kWorkspace)) {
        SetStatus(status, UrmaRealStatus::kOk, static_cast<int32_t>(probe_stage), peer);
        return;
    }

    pto::comm::AsyncSession tget_session;
    pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(workspace, static_cast<uint32_t>(peer), tget_session);
    if (probe_stage == static_cast<uint32_t>(ProbeStage::kBuildSession)) {
        SetStatus(status, UrmaRealStatus::kOk, static_cast<int32_t>(probe_stage), peer);
        return;
    }

    __gm__ pto::comm::urma::UrmaInfo *urma_info = reinterpret_cast<__gm__ pto::comm::urma::UrmaInfo *>(workspace);
    uint32_t qp_num = urma_info->qpNum;
    uint64_t sq_ptr = urma_info->sqPtr;
    if (probe_stage == static_cast<uint32_t>(ProbeStage::kWorkspaceInfo)) {
        SetStatus(status, UrmaRealStatus::kOk, static_cast<int32_t>(qp_num), peer);
        status[3] = static_cast<int32_t>(sq_ptr & 0xFFFFFFFFu);
        return;
    }

    __gm__ pto::comm::urma::UrmaMemInfo *remote_mem =
        reinterpret_cast<__gm__ pto::comm::urma::UrmaMemInfo *>(urma_info->memPtr) + peer;
    if (probe_stage == static_cast<uint32_t>(ProbeStage::kRemoteMem)) {
        SetStatus(
            status, UrmaRealStatus::kOk, static_cast<int32_t>(remote_mem->tpn), static_cast<int32_t>(remote_mem->tid)
        );
        status[3] = static_cast<int32_t>(remote_mem->rmtTokenValue);
        return;
    }
    if (probe_stage == static_cast<uint32_t>(ProbeStage::kEidRead)) {
        __gm__ uint64_t *remote_eid_probe = reinterpret_cast<__gm__ uint64_t *>(remote_mem->eidAddr);
        uint64_t eid0 = remote_eid_probe[0];
        SetStatus(
            status, UrmaRealStatus::kOk, static_cast<int32_t>(eid0 & 0xFFFFFFFFu), static_cast<int32_t>(eid0 >> 32)
        );
        return;
    }

    const bool run_tget = probe_stage == static_cast<uint32_t>(ProbeStage::kFull) ||
                          probe_stage == static_cast<uint32_t>(ProbeStage::kTgetPost) ||
                          probe_stage == static_cast<uint32_t>(ProbeStage::kTgetTestOnce);
    const bool run_tput = probe_stage == static_cast<uint32_t>(ProbeStage::kFull) ||
                          probe_stage == static_cast<uint32_t>(ProbeStage::kTputPost) ||
                          probe_stage == static_cast<uint32_t>(ProbeStage::kTputTestOnce);
    if (run_tget) {
        auto event = pto::comm::TGET_ASYNC<pto::comm::DmaEngine::URMA>(local_tget_recv_g, remote_send_g, tget_session);
        bool observed_ready = event.Test(tget_session);
        if (!WaitUrmaBounded(event, tget_session, status, UrmaRealStatus::kTgetWaitFailed)) {
            return;
        }
        if (probe_stage == static_cast<uint32_t>(ProbeStage::kTgetPost) ||
            probe_stage == static_cast<uint32_t>(ProbeStage::kTgetTestOnce)) {
            SetStatus(
                status, UrmaRealStatus::kOk, static_cast<int32_t>(event.handle & 0xFFFFFFFFu),
                observed_ready ? 1 : 0
            );
            return;
        }
    }

    if (run_tput) {
        pto::comm::AsyncSession tput_session;
        pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(workspace, static_cast<uint32_t>(peer), tput_session);
        auto event = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(remote_tput_slot_g, local_send_g, tput_session);
        bool observed_ready = event.Test(tput_session);
        if (!WaitUrmaBounded(event, tput_session, status, UrmaRealStatus::kTputWaitFailed)) {
            return;
        }
        if (probe_stage == static_cast<uint32_t>(ProbeStage::kTputPost) ||
            probe_stage == static_cast<uint32_t>(ProbeStage::kTputTestOnce)) {
            SetStatus(
                status, UrmaRealStatus::kOk, static_cast<int32_t>(event.handle & 0xFFFFFFFFu),
                observed_ready ? 1 : 0
            );
            return;
        }
    }

    if (probe_stage == static_cast<uint32_t>(ProbeStage::kFull)) {
        if (!DeviceBarrierBounded(comm_ctx, signal, my_rank, nranks)) {
            SetStatus(status, UrmaRealStatus::kBarrierWaitFailed, my_rank, peer);
            return;
        }
        for (uint32_t i = 0; i < elem_count; ++i) {
            float expected = static_cast<float>(peer * 100000 + static_cast<int>(i));
            if (tget_recv[i] != expected) {
                SetStatus(status, UrmaRealStatus::kTgetMismatch, static_cast<int32_t>(i), peer);
                status[3] = static_cast<int32_t>(tget_recv[i]);
                return;
            }
            float tput_value = tput_recv[static_cast<uint32_t>(peer) * elem_count + i];
            if (tput_value != expected) {
                SetStatus(status, UrmaRealStatus::kTputMismatch, static_cast<int32_t>(i), peer);
                status[3] = static_cast<int32_t>(tput_value);
                return;
            }
        }
        SetStatus(status, UrmaRealStatus::kOk, static_cast<int32_t>(elem_count), peer);
        return;
    }

    __gm__ pto::comm::urma::UrmaWQCtx *wq_ctx = reinterpret_cast<__gm__ pto::comm::urma::UrmaWQCtx *>(
        sq_ptr + (static_cast<uint32_t>(peer) * qp_num) * sizeof(pto::comm::urma::UrmaWQCtx)
    );
    uint64_t head_addr = wq_ctx->headAddr;
    uint64_t tail_addr = wq_ctx->tailAddr;
    if (probe_stage == static_cast<uint32_t>(ProbeStage::kWqCtx)) {
        SetStatus(
            status, UrmaRealStatus::kOk, static_cast<int32_t>(wq_ctx->depth), static_cast<int32_t>(wq_ctx->wqeShiftSize)
        );
        status[3] = static_cast<int32_t>(head_addr & 0xFFFFFFFFu);
        status[4] = static_cast<int32_t>(tail_addr & 0xFFFFFFFFu);
        return;
    }

    uint32_t head = *reinterpret_cast<__gm__ uint32_t *>(head_addr);
    uint32_t tail = *reinterpret_cast<__gm__ uint32_t *>(tail_addr);
    if (probe_stage == static_cast<uint32_t>(ProbeStage::kQueueIndexRead)) {
        SetStatus(status, UrmaRealStatus::kOk, static_cast<int32_t>(head), static_cast<int32_t>(tail));
        return;
    }
    if (probe_stage == static_cast<uint32_t>(ProbeStage::kQueueIndexLdDev)) {
        uint32_t head_ld = ld_dev(reinterpret_cast<__gm__ uint32_t *>(head_addr), 0);
        uint32_t tail_ld = ld_dev(reinterpret_cast<__gm__ uint32_t *>(tail_addr), 0);
        SetStatus(status, UrmaRealStatus::kOk, static_cast<int32_t>(head_ld), static_cast<int32_t>(tail_ld));
        return;
    }

    uint32_t wqe_size = 1U << wq_ctx->wqeShiftSize;
    uint64_t wqe_addr = wq_ctx->bufAddr + static_cast<uint64_t>(wqe_size) * (head % wq_ctx->depth);
    if (probe_stage == static_cast<uint32_t>(ProbeStage::kWqeAddr)) {
        SetStatus(
            status, UrmaRealStatus::kOk, static_cast<int32_t>(wqe_addr & 0xFFFFFFFFu), static_cast<int32_t>(head)
        );
        status[3] = static_cast<int32_t>(wqe_size);
        status[4] = static_cast<int32_t>(wq_ctx->depth);
        return;
    }
    if (probe_stage == static_cast<uint32_t>(ProbeStage::kWqeFirstStore)) {
        SetStatus(status, UrmaRealStatus::kUnsafeWqeAccess, static_cast<int32_t>(wqe_addr & 0xFFFFFFFFu), head);
        return;
    }
    if (probe_stage == static_cast<uint32_t>(ProbeStage::kWqeFirstStDev)) {
        SetStatus(status, UrmaRealStatus::kUnsafeWqeAccess, static_cast<int32_t>(wqe_addr & 0xFFFFFFFFu), head);
        return;
    }
    if (probe_stage == static_cast<uint32_t>(ProbeStage::kWqeMteStore)) {
        SetStatus(status, UrmaRealStatus::kUnsafeWqeAccess, static_cast<int32_t>(wqe_addr & 0xFFFFFFFFu), head);
        return;
    }
    if (probe_stage == static_cast<uint32_t>(ProbeStage::kWqeRead)) {
        SetStatus(status, UrmaRealStatus::kUnsafeWqeAccess, static_cast<int32_t>(wqe_addr & 0xFFFFFFFFu), head);
        return;
    }
    if (probe_stage == static_cast<uint32_t>(ProbeStage::kWqeWriteRestore)) {
        SetStatus(status, UrmaRealStatus::kUnsafeWqeAccess, static_cast<int32_t>(wqe_addr & 0xFFFFFFFFu), head);
        return;
    }

    SetStatus(status, UrmaRealStatus::kUnsafeWqeAccess, static_cast<int32_t>(probe_stage), peer);
    return;
#else
    (void)send;
    (void)tget_recv;
    (void)tput_recv;
    (void)signal;
    (void)elem_count;
    (void)peer;
    (void)probe_stage;
    SetStatus(status, UrmaRealStatus::kUnsupported);
#endif
}
