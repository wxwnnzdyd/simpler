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

#ifndef SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_RDMA_RDMA_COMPLETION_SCHEDULER_H_
#define SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_RDMA_RDMA_COMPLETION_SCHEDULER_H_

#include <cstddef>
#include <cstdint>

#include "aicpu/platform_regs.h"
#include "aicore_completion_mailbox.h"
#include "common/unified_log.h"
#include "pto_completion_token.h"
#include "pto_runtime_status.h"

namespace pto2::rdma_backend {

inline constexpr uint32_t kHandleRankIdShift = 32;
inline constexpr uint32_t kCqeBytes = 64;
inline constexpr uint32_t kCqUpdateCiMask = 0xffffffu;
inline constexpr uint32_t kRdmaWorkspaceMagic = 0x52444d41u;
inline constexpr uint32_t kRdmaWorkspaceVersion = 1u;
inline constexpr uint32_t kRdmaBackendHns1825 = 1u;
inline constexpr uint32_t kRdmaErrorRank = 0xffffffffu;

enum class RdmaDbMode : int32_t {
    INVALID_DB = -1,
    HW_DB = 0,
    SW_DB = 1,
};

struct RdmaInfo {
    uint32_t magic;
    uint32_t version;
    uint32_t backend;
    uint32_t qp_num;
    uint32_t rank_count;
    uint32_t reserved;
    uint64_t sq_ptr;
    uint64_t rq_ptr;
    uint64_t scq_ptr;
    uint64_t rcq_ptr;
    uint64_t mem_ptr;
};

struct RdmaWqCtx {
    uint32_t wqn;
    uint64_t buf_addr;
    uint32_t wqe_size;
    uint32_t depth;
    uint64_t head_addr;
    uint64_t tail_addr;
    RdmaDbMode db_mode;
    uint64_t db_addr;
    uint32_t sl;
    uint64_t amo_addr;
    uint32_t amo_lkey;
    uint64_t db_sw_addr;
    uint8_t mtu_shift;
    uint8_t reserved[7];
};

struct RdmaCqCtx {
    uint32_t cqn;
    uint64_t buf_addr;
    uint32_t cqe_size;
    uint32_t depth;
    uint64_t head_addr;
    uint64_t tail_addr;
    RdmaDbMode db_mode;
    uint64_t db_addr;
    uint64_t db_sw_addr;
};

struct Hns1825Cqe {
    uint32_t owner_id_qpn;
    uint32_t op_sr_wqebb;
    uint32_t byte_cnt;
    uint32_t imm_data;
    uint32_t rsvd_dw5;
    uint32_t wqe_num;
    uint32_t vlan_queue_index;
    uint8_t syndrome;
    uint8_t rsvd;
    uint16_t wqe_counter;
};

static_assert(sizeof(RdmaInfo) == 64, "RDMA info ABI drift");
static_assert(offsetof(RdmaInfo, sq_ptr) == 24, "RDMA info ABI drift");
static_assert(sizeof(RdmaWqCtx) == 96, "RDMA WQ context ABI drift");
static_assert(offsetof(RdmaWqCtx, db_addr) == 48, "RDMA WQ context ABI drift");
static_assert(offsetof(RdmaWqCtx, db_sw_addr) == 80, "RDMA WQ context ABI drift");
static_assert(sizeof(RdmaCqCtx) == 64, "RDMA CQ context ABI drift");
static_assert(offsetof(RdmaCqCtx, db_addr) == 48, "RDMA CQ context ABI drift");
static_assert(offsetof(RdmaCqCtx, db_sw_addr) == 56, "RDMA CQ context ABI drift");
static_assert(sizeof(Hns1825Cqe) == 32, "RDMA CQE ABI drift");

inline uint64_t encode_rdma_event_handle(uint32_t dest_rank, uint32_t target_head) {
    return (static_cast<uint64_t>(dest_rank) << kHandleRankIdShift) | static_cast<uint64_t>(target_head);
}

inline bool is_rdma_error_handle(uint64_t handle) {
    return static_cast<uint32_t>(handle >> kHandleRankIdShift) == kRdmaErrorRank;
}

inline void decode_rdma_event_handle(uint64_t handle, uint32_t &dest_rank, uint32_t &target_head) {
    dest_rank = static_cast<uint32_t>(handle >> kHandleRankIdShift);
    target_head = static_cast<uint32_t>(handle & 0xFFFFFFFFu);
}

inline uintptr_t cache_line(const volatile void *addr) {
    return reinterpret_cast<uintptr_t>(addr) & ~(uintptr_t(PTO2_ALIGN_SIZE) - 1u);
}

inline void invalidate_object(const volatile void *addr, std::size_t size) {
    const uintptr_t object_addr = reinterpret_cast<uintptr_t>(addr);
    const uintptr_t begin = cache_line(addr);
    const uintptr_t end = (object_addr + size + PTO2_ALIGN_SIZE - 1u) & ~(uintptr_t(PTO2_ALIGN_SIZE) - 1u);
    cache_invalidate_range(reinterpret_cast<const void *>(begin), end - begin);
}

inline bool has_reached(uint32_t current, uint32_t target) { return static_cast<int32_t>(current - target) >= 0; }

inline bool is_power_of_two(uint32_t value) { return value != 0 && (value & (value - 1u)) == 0; }

inline bool is_hns1825_cqe_owner_ready(bool owner, uint32_t cur_tail, uint32_t cq_ring) {
    const bool expected_owner = (cur_tail & cq_ring) == 0;
    return owner != expected_owner;
}

inline uint32_t htobe32(uint32_t value) { return __builtin_bswap32(value); }

inline uint32_t load_device_u32(uint64_t addr) {
    auto *ptr = reinterpret_cast<volatile uint32_t *>(static_cast<uintptr_t>(addr));
    return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
}

inline uint32_t load_device_u32_or_zero(uint64_t addr) { return addr != 0 ? load_device_u32(addr) : 0; }

inline void store_device_u32(uint64_t addr, uint32_t value) {
    auto *ptr = reinterpret_cast<volatile uint32_t *>(static_cast<uintptr_t>(addr));
    __atomic_store_n(ptr, value, __ATOMIC_RELEASE);
#if defined(__aarch64__)
    __asm__ __volatile__("dsb sy" ::: "memory");
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

inline void update_tail_info(const RdmaCqCtx &cq_ctx, const RdmaWqCtx &wq_ctx, uint32_t cur_tail) {
    if (cq_ctx.tail_addr != 0) {
        store_device_u32(cq_ctx.tail_addr, cur_tail);
    }
    if (cq_ctx.db_sw_addr != 0) {
        store_device_u32(cq_ctx.db_sw_addr, htobe32(cur_tail & kCqUpdateCiMask));
    }
    if (wq_ctx.tail_addr != 0) {
        store_device_u32(wq_ctx.tail_addr, cur_tail);
    }
}

inline CompletionPollResult poll_rdma_event_handle(uint64_t event_handle, uint64_t workspace_addr) {
    if (event_handle == 0) {
        return {CompletionPollState::READY, PTO2_ERROR_NONE};
    }
    if (is_rdma_error_handle(event_handle) || workspace_addr == 0) {
        LOG_ERROR(
            "RDMA completion invalid: bad handle/workspace handle=0x%llx workspace=0x%llx is_error=%u",
            static_cast<unsigned long long>(event_handle), static_cast<unsigned long long>(workspace_addr),
            static_cast<unsigned>(is_rdma_error_handle(event_handle))
        );
        return {CompletionPollState::FAILED, PTO2_ERROR_ASYNC_COMPLETION_INVALID};
    }

    uint32_t dest_rank = 0;
    uint32_t target_head = 0;
    decode_rdma_event_handle(event_handle, dest_rank, target_head);
    if (target_head == 0) {
        return {CompletionPollState::READY, PTO2_ERROR_NONE};
    }

    auto *info = reinterpret_cast<volatile RdmaInfo *>(static_cast<uintptr_t>(workspace_addr));
    invalidate_object(info, sizeof(*info));
    const uint32_t magic = __atomic_load_n(&info->magic, __ATOMIC_ACQUIRE);
    const uint32_t version = __atomic_load_n(&info->version, __ATOMIC_ACQUIRE);
    const uint32_t backend = __atomic_load_n(&info->backend, __ATOMIC_ACQUIRE);
    const uint32_t qp_num = __atomic_load_n(&info->qp_num, __ATOMIC_ACQUIRE);
    const uint32_t rank_count = __atomic_load_n(&info->rank_count, __ATOMIC_ACQUIRE);
    const uint64_t sq_ptr = __atomic_load_n(&info->sq_ptr, __ATOMIC_ACQUIRE);
    const uint64_t scq_ptr = __atomic_load_n(&info->scq_ptr, __ATOMIC_ACQUIRE);
    if (magic != kRdmaWorkspaceMagic || version != kRdmaWorkspaceVersion || backend != kRdmaBackendHns1825 ||
        qp_num == 0 || dest_rank >= rank_count || sq_ptr == 0 || scq_ptr == 0) {
        LOG_ERROR(
            "RDMA completion invalid: workspace metadata handle=0x%llx workspace=0x%llx dest_rank=%u target_head=%u "
            "magic=0x%x version=%u backend=%u qp_num=%u rank_count=%u sq=0x%llx scq=0x%llx",
            static_cast<unsigned long long>(event_handle), static_cast<unsigned long long>(workspace_addr), dest_rank,
            target_head, magic, version, backend, qp_num, rank_count, static_cast<unsigned long long>(sq_ptr),
            static_cast<unsigned long long>(scq_ptr)
        );
        return {CompletionPollState::FAILED, PTO2_ERROR_ASYNC_COMPLETION_INVALID};
    }

    const uint64_t ctx_index = static_cast<uint64_t>(dest_rank) * qp_num;
    auto *cq_entry = reinterpret_cast<volatile RdmaCqCtx *>(
        static_cast<uintptr_t>(scq_ptr + ctx_index * static_cast<uint64_t>(sizeof(RdmaCqCtx)))
    );
    auto *wq_entry = reinterpret_cast<volatile RdmaWqCtx *>(
        static_cast<uintptr_t>(sq_ptr + ctx_index * static_cast<uint64_t>(sizeof(RdmaWqCtx)))
    );
    invalidate_object(cq_entry, sizeof(*cq_entry));
    invalidate_object(wq_entry, sizeof(*wq_entry));

    RdmaCqCtx cq_ctx{};
    cq_ctx.buf_addr = __atomic_load_n(&cq_entry->buf_addr, __ATOMIC_ACQUIRE);
    cq_ctx.cqe_size = __atomic_load_n(&cq_entry->cqe_size, __ATOMIC_ACQUIRE);
    cq_ctx.depth = __atomic_load_n(&cq_entry->depth, __ATOMIC_ACQUIRE);
    cq_ctx.tail_addr = __atomic_load_n(&cq_entry->tail_addr, __ATOMIC_ACQUIRE);
    cq_ctx.db_addr = __atomic_load_n(&cq_entry->db_addr, __ATOMIC_ACQUIRE);
    cq_ctx.db_sw_addr = __atomic_load_n(&cq_entry->db_sw_addr, __ATOMIC_ACQUIRE);

    RdmaWqCtx wq_ctx{};
    wq_ctx.tail_addr = __atomic_load_n(&wq_entry->tail_addr, __ATOMIC_ACQUIRE);
    const uint32_t cqe_size = cq_ctx.cqe_size == 0 ? kCqeBytes : cq_ctx.cqe_size;
    const uint32_t cq_ring = cq_ctx.depth;
    if (cqe_size < sizeof(Hns1825Cqe) || cqe_size > kCqeBytes || cq_ctx.buf_addr == 0 || cq_ctx.tail_addr == 0 ||
        cq_ctx.db_sw_addr == 0 || !is_power_of_two(cq_ring)) {
        LOG_ERROR(
            "RDMA completion invalid: cq context handle=0x%llx workspace=0x%llx dest_rank=%u target_head=%u "
            "cqe_size=%u depth=%u cq_ring=%u cq_buf=0x%llx cq_tail=0x%llx cq_db_sw=0x%llx sq_tail=0x%llx",
            static_cast<unsigned long long>(event_handle), static_cast<unsigned long long>(workspace_addr), dest_rank,
            target_head, cqe_size, cq_ctx.depth, cq_ring, static_cast<unsigned long long>(cq_ctx.buf_addr),
            static_cast<unsigned long long>(cq_ctx.tail_addr), static_cast<unsigned long long>(cq_ctx.db_sw_addr),
            static_cast<unsigned long long>(wq_ctx.tail_addr)
        );
        return {CompletionPollState::FAILED, PTO2_ERROR_ASYNC_COMPLETION_INVALID};
    }

    uint32_t cur_tail = load_device_u32(cq_ctx.tail_addr);
    if (has_reached(cur_tail, target_head)) {
        return {CompletionPollState::READY, PTO2_ERROR_NONE};
    }

    uint32_t next_tail = cur_tail;
    while (!has_reached(next_tail, target_head)) {
        const uint64_t cqe_addr =
            cq_ctx.buf_addr + static_cast<uint64_t>(cqe_size) * static_cast<uint64_t>(next_tail & (cq_ring - 1));
        auto *cqe = reinterpret_cast<volatile Hns1825Cqe *>(static_cast<uintptr_t>(cqe_addr));
        invalidate_object(cqe, sizeof(*cqe));
        const uint32_t owner_id_qpn = __atomic_load_n(&cqe->owner_id_qpn, __ATOMIC_ACQUIRE);
        const uint32_t op_sr_wqebb = __atomic_load_n(&cqe->op_sr_wqebb, __ATOMIC_ACQUIRE);
        constexpr uint32_t kOwnerShift = 31;
        constexpr uint32_t kCqeOpcodeShift = 27;
        constexpr uint32_t kCqeOpcodeMask = 0x1f;
        constexpr uint32_t kCqeOptypeError = 0x1e;
        constexpr uint32_t kCqeOptypeInvalid = 0x1f;
        const uint32_t cqe_type = (op_sr_wqebb >> kCqeOpcodeShift) & kCqeOpcodeMask;
        const bool owner = (owner_id_qpn & (1u << kOwnerShift)) != 0;
        if (cqe_type == kCqeOptypeInvalid || !is_hns1825_cqe_owner_ready(owner, next_tail, cq_ring)) {
            break;
        }

        if (cqe_type == kCqeOptypeError) {
            const uint8_t syndrome = __atomic_load_n(&cqe->syndrome, __ATOMIC_ACQUIRE);
            const uint32_t byte_cnt = __atomic_load_n(&cqe->byte_cnt, __ATOMIC_ACQUIRE);
            const uint32_t imm_data = __atomic_load_n(&cqe->imm_data, __ATOMIC_ACQUIRE);
            const uint32_t wqe_num = __atomic_load_n(&cqe->wqe_num, __ATOMIC_ACQUIRE);
            const uint32_t vlan_queue_index = __atomic_load_n(&cqe->vlan_queue_index, __ATOMIC_ACQUIRE);
            LOG_ERROR(
                "RDMA completion invalid: CQE error handle=0x%llx workspace=0x%llx dest_rank=%u target_head=%u "
                "cur_tail=%u next_tail=%u cqe_addr=0x%llx owner_id_qpn=0x%x op_sr_wqebb=0x%x "
                "byte_cnt=0x%x imm_data=0x%x wqe_num=0x%x vlan_queue_index=0x%x syndrome=0x%x",
                static_cast<unsigned long long>(event_handle), static_cast<unsigned long long>(workspace_addr),
                dest_rank, target_head, cur_tail, next_tail, static_cast<unsigned long long>(cqe_addr), owner_id_qpn,
                op_sr_wqebb, byte_cnt, imm_data, wqe_num, vlan_queue_index, static_cast<unsigned>(syndrome)
            );
            ++next_tail;
            update_tail_info(cq_ctx, wq_ctx, next_tail);
            return {CompletionPollState::FAILED, PTO2_ERROR_ASYNC_COMPLETION_INVALID};
        }
        next_tail++;
    }

    if (next_tail != cur_tail) {
        update_tail_info(cq_ctx, wq_ctx, next_tail);
    }
    return {
        has_reached(next_tail, target_head) ? CompletionPollState::READY : CompletionPollState::PENDING, PTO2_ERROR_NONE
    };
}

inline void
log_rdma_event_handle_snapshot(uint64_t event_handle, uint64_t workspace_addr, int32_t entry_idx, int32_t cond_idx) {
    uint32_t dest_rank = 0;
    uint32_t target_head = 0;
    decode_rdma_event_handle(event_handle, dest_rank, target_head);
    LOG_INFO_V9(
        "[ASYNC_WAIT RDMA entry=%d cond=%d] handle=0x%llx workspace=0x%llx dest_rank=%u target_head=%u", entry_idx,
        cond_idx, static_cast<unsigned long long>(event_handle), static_cast<unsigned long long>(workspace_addr),
        dest_rank, target_head
    );

    if (event_handle == 0 || workspace_addr == 0 || is_rdma_error_handle(event_handle)) {
        return;
    }

    auto *info = reinterpret_cast<volatile RdmaInfo *>(static_cast<uintptr_t>(workspace_addr));
    invalidate_object(info, sizeof(*info));
    const uint32_t magic = __atomic_load_n(&info->magic, __ATOMIC_ACQUIRE);
    const uint32_t version = __atomic_load_n(&info->version, __ATOMIC_ACQUIRE);
    const uint32_t backend = __atomic_load_n(&info->backend, __ATOMIC_ACQUIRE);
    const uint32_t qp_num = __atomic_load_n(&info->qp_num, __ATOMIC_ACQUIRE);
    const uint32_t rank_count = __atomic_load_n(&info->rank_count, __ATOMIC_ACQUIRE);
    const uint64_t sq_ptr = __atomic_load_n(&info->sq_ptr, __ATOMIC_ACQUIRE);
    const uint64_t scq_ptr = __atomic_load_n(&info->scq_ptr, __ATOMIC_ACQUIRE);
    LOG_INFO_V9(
        "[ASYNC_WAIT RDMA entry=%d cond=%d] info magic=0x%x version=%u backend=%u qp_num=%u rank_count=%u sq=0x%llx "
        "scq=0x%llx",
        entry_idx, cond_idx, magic, version, backend, qp_num, rank_count, static_cast<unsigned long long>(sq_ptr),
        static_cast<unsigned long long>(scq_ptr)
    );
    if (magic != kRdmaWorkspaceMagic || version != kRdmaWorkspaceVersion || backend != kRdmaBackendHns1825 ||
        qp_num == 0 || dest_rank >= rank_count || sq_ptr == 0 || scq_ptr == 0) {
        return;
    }

    const uint64_t ctx_index = static_cast<uint64_t>(dest_rank) * qp_num;
    auto *cq_entry = reinterpret_cast<volatile RdmaCqCtx *>(
        static_cast<uintptr_t>(scq_ptr + ctx_index * static_cast<uint64_t>(sizeof(RdmaCqCtx)))
    );
    auto *wq_entry = reinterpret_cast<volatile RdmaWqCtx *>(
        static_cast<uintptr_t>(sq_ptr + ctx_index * static_cast<uint64_t>(sizeof(RdmaWqCtx)))
    );
    invalidate_object(cq_entry, sizeof(*cq_entry));
    invalidate_object(wq_entry, sizeof(*wq_entry));

    RdmaCqCtx cq_ctx{};
    cq_ctx.buf_addr = __atomic_load_n(&cq_entry->buf_addr, __ATOMIC_ACQUIRE);
    cq_ctx.cqe_size = __atomic_load_n(&cq_entry->cqe_size, __ATOMIC_ACQUIRE);
    cq_ctx.depth = __atomic_load_n(&cq_entry->depth, __ATOMIC_ACQUIRE);
    cq_ctx.tail_addr = __atomic_load_n(&cq_entry->tail_addr, __ATOMIC_ACQUIRE);
    cq_ctx.db_sw_addr = __atomic_load_n(&cq_entry->db_sw_addr, __ATOMIC_ACQUIRE);
    RdmaWqCtx wq_ctx{};
    wq_ctx.wqn = __atomic_load_n(&wq_entry->wqn, __ATOMIC_ACQUIRE);
    wq_ctx.head_addr = __atomic_load_n(&wq_entry->head_addr, __ATOMIC_ACQUIRE);
    wq_ctx.tail_addr = __atomic_load_n(&wq_entry->tail_addr, __ATOMIC_ACQUIRE);
    wq_ctx.db_addr = __atomic_load_n(&wq_entry->db_addr, __ATOMIC_ACQUIRE);
    wq_ctx.db_sw_addr = __atomic_load_n(&wq_entry->db_sw_addr, __ATOMIC_ACQUIRE);
    const uint32_t cqe_size = cq_ctx.cqe_size == 0 ? kCqeBytes : cq_ctx.cqe_size;
    const uint32_t cur_tail = cq_ctx.tail_addr != 0 ? load_device_u32(cq_ctx.tail_addr) : 0;
    const uint32_t sq_head = load_device_u32_or_zero(wq_ctx.head_addr);
    const uint32_t sq_tail = load_device_u32_or_zero(wq_ctx.tail_addr);
    const uint32_t sq_db_sw = load_device_u32_or_zero(wq_ctx.db_sw_addr);
    const uint32_t cq_db_sw = load_device_u32_or_zero(cq_ctx.db_sw_addr);
    const bool ctx_valid = cqe_size >= sizeof(Hns1825Cqe) && cqe_size <= kCqeBytes && cq_ctx.buf_addr != 0 &&
                           cq_ctx.tail_addr != 0 && cq_ctx.db_sw_addr != 0 && is_power_of_two(cq_ctx.depth);
    LOG_INFO_V9(
        "[ASYNC_WAIT RDMA entry=%d cond=%d] sq wqn=%u head=%u tail=%u db_sw_be=0x%x head_addr=0x%llx "
        "tail_addr=0x%llx db_hw=0x%llx db_sw=0x%llx",
        entry_idx, cond_idx, wq_ctx.wqn, sq_head, sq_tail, sq_db_sw, static_cast<unsigned long long>(wq_ctx.head_addr),
        static_cast<unsigned long long>(wq_ctx.tail_addr), static_cast<unsigned long long>(wq_ctx.db_addr),
        static_cast<unsigned long long>(wq_ctx.db_sw_addr)
    );
    LOG_INFO_V9(
        "[ASYNC_WAIT RDMA entry=%d cond=%d] cq cqn=%u depth=%u cqe_size=%u cur_tail=%u target_head=%u "
        "db_sw_be=0x%x cq_buf=0x%llx cq_tail=0x%llx cq_db_sw=0x%llx valid=%u",
        entry_idx, cond_idx, cq_ctx.cqn, cq_ctx.depth, cqe_size, cur_tail, target_head, cq_db_sw,
        static_cast<unsigned long long>(cq_ctx.buf_addr), static_cast<unsigned long long>(cq_ctx.tail_addr),
        static_cast<unsigned long long>(cq_ctx.db_sw_addr), static_cast<unsigned>(ctx_valid)
    );
    if (!ctx_valid || has_reached(cur_tail, target_head)) {
        return;
    }

    const uint64_t cqe_addr =
        cq_ctx.buf_addr + static_cast<uint64_t>(cqe_size) * static_cast<uint64_t>(cur_tail & (cq_ctx.depth - 1));
    auto *cqe = reinterpret_cast<volatile Hns1825Cqe *>(static_cast<uintptr_t>(cqe_addr));
    invalidate_object(cqe, sizeof(*cqe));
    const uint32_t owner_id_qpn = __atomic_load_n(&cqe->owner_id_qpn, __ATOMIC_ACQUIRE);
    const uint32_t op_sr_wqebb = __atomic_load_n(&cqe->op_sr_wqebb, __ATOMIC_ACQUIRE);
    constexpr uint32_t kOwnerShift = 31;
    constexpr uint32_t kCqeOpcodeShift = 27;
    constexpr uint32_t kCqeOpcodeMask = 0x1f;
    const bool owner = (owner_id_qpn & (1u << kOwnerShift)) != 0;
    const uint32_t cqe_type = (op_sr_wqebb >> kCqeOpcodeShift) & kCqeOpcodeMask;
    LOG_INFO_V9(
        "[ASYNC_WAIT RDMA entry=%d cond=%d] cqe addr=0x%llx owner=%u ready=%u type=0x%x owner_id_qpn=0x%x "
        "op_sr_wqebb=0x%x syndrome=0x%x",
        entry_idx, cond_idx, static_cast<unsigned long long>(cqe_addr), static_cast<unsigned>(owner),
        static_cast<unsigned>(is_hns1825_cqe_owner_ready(owner, cur_tail, cq_ctx.depth)), cqe_type, owner_id_qpn,
        op_sr_wqebb, static_cast<unsigned>(__atomic_load_n(&cqe->syndrome, __ATOMIC_ACQUIRE))
    );
}

inline void retire_rdma_event_handle(uint64_t /*event_handle*/, uint64_t /*workspace_addr*/) {}

}  // namespace pto2::rdma_backend

#endif  // SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_RDMA_RDMA_COMPLETION_SCHEDULER_H_
