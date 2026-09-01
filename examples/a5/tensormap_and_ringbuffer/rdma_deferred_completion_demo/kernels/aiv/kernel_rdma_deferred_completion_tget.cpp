#include "rdma_deferred_completion_common.h"

// DIAG: synchronous AICore wait (equivalent to native STATUS_WAIT_EACH).
// Posts the TGET then blocks in AICore until the CQE arrives, instead of
// registering a deferred completion for the AICPU poller. Distinguishes
// "READ never completes" from "READ completes but AICPU cannot observe it".

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *send_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *tget_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *marker_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ CommContext *comm_ctx = reinterpret_cast<__gm__ CommContext *>(args[4]);
    uint32_t elem_count = static_cast<uint32_t>(args[5]);

    __gm__ float *send = rdma_deferred_completion::tensor_data<float>(send_tensor);
    __gm__ float *tget = rdma_deferred_completion::tensor_data<float>(tget_tensor);
    __gm__ int32_t *marker = rdma_deferred_completion::tensor_data<int32_t>(marker_tensor);
    uint32_t peer = rdma_deferred_completion::peer_rank(comm_ctx);
    uint64_t peer_base = rdma_deferred_completion::remote_base(comm_ctx, peer);
    uint64_t send_offset = rdma_deferred_completion::local_offset(comm_ctx, send);
    __gm__ float *remote_send = reinterpret_cast<__gm__ float *>(peer_base + send_offset);

    uint32_t first_count = rdma_deferred_completion::first_chunk_count(elem_count);
    uint32_t second_count = rdma_deferred_completion::second_chunk_count(elem_count);
    auto rdma_scratch = rdma_deferred_completion::rdma_scratch_tile();

    // DIAG: one-way discriminator. rank 0 posts the TGET and waits in AICore
    // (equivalent to native root-only TGET); rank 1 posts nothing. If rank 0's
    // wait succeeds, the wire path is fine and the failure is bidirectional
    // RDMA; if rank 0 also times out, the AICore doorbell/compile is suspect.
    if (comm_ctx->rankId != 0) {
        rdma_deferred_completion::store_marker(marker, 1);
        return;
    }

    auto local_tget = rdma_deferred_completion::global_float(tget, first_count);
    auto remote_send_g = rdma_deferred_completion::global_float(remote_send, first_count);
    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::RDMA>(
            rdma_scratch, reinterpret_cast<__gm__ uint8_t *>(comm_ctx->workSpace), comm_ctx->rankId, session, 0
        )) {
        rdma_deferred_completion::store_marker(marker, -1);
        return;
    }
    auto ev0 = pto::comm::TGET_ASYNC<pto::comm::DmaEngine::RDMA>(local_tget, remote_send_g, session, peer);
    if (!rdma_deferred_completion::wait_rdma_bounded(ev0, session)) {
        rdma_deferred_completion::store_marker(marker, -50);
        return;
    }
    uint32_t request_count = 1;
    if (second_count != 0) {
        request_count++;
        auto local_tget_tail = rdma_deferred_completion::global_float(tget + first_count, second_count);
        auto remote_send_tail = rdma_deferred_completion::global_float(remote_send + first_count, second_count);
        pto::comm::AsyncSession tail_session;
        if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::RDMA>(
                rdma_scratch, reinterpret_cast<__gm__ uint8_t *>(comm_ctx->workSpace), comm_ctx->rankId, tail_session,
                1
            )) {
            rdma_deferred_completion::store_marker(marker, -1);
            return;
        }
        auto ev1 = pto::comm::TGET_ASYNC<pto::comm::DmaEngine::RDMA>(
            local_tget_tail, remote_send_tail, tail_session, peer
        );
        if (!rdma_deferred_completion::wait_rdma_bounded(ev1, tail_session)) {
            rdma_deferred_completion::store_marker(marker, -50);
            return;
        }
    }
    rdma_deferred_completion::store_marker(marker, static_cast<int32_t>(request_count));
}
