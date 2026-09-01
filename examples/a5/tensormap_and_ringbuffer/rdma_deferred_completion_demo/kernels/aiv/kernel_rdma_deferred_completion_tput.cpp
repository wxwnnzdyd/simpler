#include "rdma_deferred_completion_common.h"

// DIAG: synchronous AICore wait (equivalent to native STATUS_WAIT_EACH).
// Posts the TPUT then blocks in AICore until the CQE arrives, instead of
// registering a deferred completion for the AICPU poller. Distinguishes
// "WRITE never completes" from "WRITE completes but AICPU cannot observe it".

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *send_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *tput_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *marker_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ CommContext *comm_ctx = reinterpret_cast<__gm__ CommContext *>(args[4]);
    uint32_t elem_count = static_cast<uint32_t>(args[5]);

    __gm__ float *send = rdma_deferred_completion::tensor_data<float>(send_tensor);
    __gm__ float *tput = rdma_deferred_completion::tensor_data<float>(tput_tensor);
    __gm__ int32_t *marker = rdma_deferred_completion::tensor_data<int32_t>(marker_tensor);
    uint32_t peer = rdma_deferred_completion::peer_rank(comm_ctx);
    uint64_t peer_base = rdma_deferred_completion::remote_base(comm_ctx, peer);
    uint64_t tput_slot_offset = rdma_deferred_completion::local_offset(comm_ctx, tput) +
                                static_cast<uint64_t>(comm_ctx->rankId) * elem_count * sizeof(float);
    __gm__ float *remote_tput_slot = reinterpret_cast<__gm__ float *>(peer_base + tput_slot_offset);

    uint32_t first_count = rdma_deferred_completion::first_chunk_count(elem_count);
    uint32_t second_count = rdma_deferred_completion::second_chunk_count(elem_count);
    auto rdma_scratch = rdma_deferred_completion::rdma_scratch_tile();

    // DIAG: one-way discriminator (mirror of the TGET producer). rank 0 posts
    // the TPUT and waits; rank 1 posts nothing. Confirms whether the wire path
    // is fine in isolation vs. bidirectional RDMA being the failure trigger.
    if (comm_ctx->rankId != 0) {
        rdma_deferred_completion::store_marker(marker, 1);
        return;
    }

    auto local_send = rdma_deferred_completion::global_float(send, first_count);
    auto remote_tput = rdma_deferred_completion::global_float(remote_tput_slot, first_count);
    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::RDMA>(
            rdma_scratch, reinterpret_cast<__gm__ uint8_t *>(comm_ctx->workSpace), comm_ctx->rankId, session, 0
        )) {
        rdma_deferred_completion::store_marker(marker, -1);
        return;
    }
    auto ev0 = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::RDMA>(remote_tput, local_send, session, peer);
    if (!rdma_deferred_completion::wait_rdma_bounded(ev0, session)) {
        rdma_deferred_completion::store_marker(marker, -50);
        return;
    }
    uint32_t request_count = 1;
    if (second_count != 0) {
        request_count++;
        auto local_send_tail = rdma_deferred_completion::global_float(send + first_count, second_count);
        auto remote_tput_tail = rdma_deferred_completion::global_float(remote_tput_slot + first_count, second_count);
        pto::comm::AsyncSession tail_session;
        if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::RDMA>(
                rdma_scratch, reinterpret_cast<__gm__ uint8_t *>(comm_ctx->workSpace), comm_ctx->rankId, tail_session,
                1
            )) {
            rdma_deferred_completion::store_marker(marker, -1);
            return;
        }
        auto ev1 = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::RDMA>(
            remote_tput_tail, local_send_tail, tail_session, peer
        );
        if (!rdma_deferred_completion::wait_rdma_bounded(ev1, tail_session)) {
            rdma_deferred_completion::store_marker(marker, -50);
            return;
        }
    }
    rdma_deferred_completion::store_marker(marker, static_cast<int32_t>(request_count));
}
