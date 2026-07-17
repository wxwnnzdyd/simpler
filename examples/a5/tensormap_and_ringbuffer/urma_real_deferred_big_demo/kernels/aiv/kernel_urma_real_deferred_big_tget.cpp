#include "urma_real_deferred_demo/kernels/aiv/urma_real_deferred_common.h"

namespace {

inline __aicore__ uint32_t first_urma_chunk_elems() { return (256u * 1024u * 1024u) / sizeof(float); }

inline __aicore__ float expected_value(uint32_t rank, uint32_t idx) {
    return static_cast<float>(rank * 100000 + static_cast<int>(idx & 0xFFFFu));
}

inline __aicore__ void write_sentinel(__gm__ float *send, uint32_t elem_count, uint32_t rank) {
    send[0] = expected_value(rank, 0);
    uint32_t chunk_tail = first_urma_chunk_elems() - 1u;
    if (chunk_tail < elem_count) {
        send[chunk_tail] = expected_value(rank, chunk_tail);
    }
    uint32_t chunk_head = first_urma_chunk_elems();
    if (chunk_head < elem_count) {
        send[chunk_head] = expected_value(rank, chunk_head);
    }
    send[elem_count - 1u] = expected_value(rank, elem_count - 1u);
    pto2::detail::defer_flush_range(send, sizeof(float));
    if (chunk_tail < elem_count) {
        pto2::detail::defer_flush_range(send + chunk_tail, sizeof(float));
    }
    if (chunk_head < elem_count) {
        pto2::detail::defer_flush_range(send + chunk_head, sizeof(float));
    }
    pto2::detail::defer_flush_range(send + elem_count - 1u, sizeof(float));
}

inline __aicore__ bool wait_peer_ready(
    __gm__ CommContext *comm_ctx, __gm__ int32_t *signal, uint32_t peer, uint32_t token, __gm__ int32_t *status
) {
    urma_real_deferred::store_marker(signal, static_cast<int32_t>(token));
    uint64_t peer_base = urma_real_deferred::remote_base(comm_ctx, peer);
    uint64_t signal_offset = urma_real_deferred::local_offset(comm_ctx, signal);
    __gm__ int32_t *remote_signal = reinterpret_cast<__gm__ int32_t *>(peer_base + signal_offset);
    for (uint32_t iter = 0; iter < urma_real_deferred::kMaxRemoteWritePollIters; ++iter) {
        __asm__ __volatile__("" ::: "memory");
        dcci((__gm__ void *)remote_signal, SINGLE_CACHE_LINE);
        __asm__ __volatile__("" ::: "memory");
        if (remote_signal[0] == static_cast<int32_t>(token)) {
            return true;
        }
    }
    urma_real_deferred::set_status(status, urma_real_deferred::Status::kSubmitFailed, static_cast<int32_t>(token), peer);
    return false;
}

}  // namespace

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *send_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *recv_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *signal_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *marker_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ Tensor *status_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ CommContext *comm_ctx = reinterpret_cast<__gm__ CommContext *>(args[5]);
    uint32_t elem_count = static_cast<uint32_t>(args[6]);

    __gm__ float *send = urma_real_deferred::tensor_data<float>(send_tensor);
    __gm__ float *recv = urma_real_deferred::tensor_data<float>(recv_tensor);
    __gm__ int32_t *signal = urma_real_deferred::tensor_data<int32_t>(signal_tensor);
    __gm__ int32_t *marker = urma_real_deferred::tensor_data<int32_t>(marker_tensor);
    __gm__ int32_t *status = urma_real_deferred::tensor_data<int32_t>(status_tensor);

    if (!urma_real_deferred::validate_comm(comm_ctx, elem_count, status)) {
        return;
    }

    uint32_t peer = urma_real_deferred::peer_rank(comm_ctx);
    write_sentinel(send, elem_count, comm_ctx->rankId);
    uint32_t token = 0x5A000000u | (elem_count & 0x00FFFFFFu);
    if (!wait_peer_ready(comm_ctx, signal, peer, token, status)) {
        return;
    }

    uint64_t send_offset = urma_real_deferred::local_offset(comm_ctx, send);
    __gm__ float *remote_send = pto2::urma_backend::peer_mr_ptr<float>(
        reinterpret_cast<__gm__ uint8_t *>(comm_ctx->workSpace), peer, send_offset
    );

    AsyncCtx async_ctx = get_async_ctx(args);
    auto local_recv = urma_real_deferred::global_float(recv, elem_count);
    auto remote_send_g = urma_real_deferred::global_float(remote_send, elem_count);
    bool submitted = send_request_entry(
        async_ctx, UrmaTget(local_recv, remote_send_g, reinterpret_cast<__gm__ uint8_t *>(comm_ctx->workSpace), peer)
    );
    if (!submitted) {
        urma_real_deferred::set_status(status, urma_real_deferred::Status::kSubmitFailed, elem_count, peer);
        return;
    }

    urma_real_deferred::store_marker(marker, 2);
}
