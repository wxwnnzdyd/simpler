#include "urma_real_deferred_common.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *send_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *tget_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *marker_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ CommContext *comm_ctx = reinterpret_cast<__gm__ CommContext *>(args[3]);
    uint32_t elem_count = static_cast<uint32_t>(args[4]);

    __gm__ float *send = urma_real_deferred::tensor_data<float>(send_tensor);
    __gm__ float *tget = urma_real_deferred::tensor_data<float>(tget_tensor);
    __gm__ int32_t *marker = urma_real_deferred::tensor_data<int32_t>(marker_tensor);
    uint32_t peer = urma_real_deferred::peer_rank(comm_ctx);
    uint64_t peer_base = urma_real_deferred::remote_base(comm_ctx, peer);
    uint64_t send_offset = urma_real_deferred::local_offset(comm_ctx, send);
    __gm__ float *remote_send = reinterpret_cast<__gm__ float *>(peer_base + send_offset);

    uint32_t first_count = urma_real_deferred::first_chunk_count(elem_count);
    uint32_t second_count = urma_real_deferred::second_chunk_count(elem_count);
    AsyncCtx async_ctx = get_async_ctx(args);

    auto local_tget = urma_real_deferred::global_float(tget, first_count);
    auto remote_send_g = urma_real_deferred::global_float(remote_send, first_count);
    uint32_t request_count = 1;
    (void)send_request_entry(
        async_ctx, UrmaTget(local_tget, remote_send_g, reinterpret_cast<__gm__ uint8_t *>(comm_ctx->workSpace), peer)
    );
    if (second_count != 0) {
        request_count++;
        auto local_tget_tail = urma_real_deferred::global_float(tget + first_count, second_count);
        auto remote_send_tail = urma_real_deferred::global_float(remote_send + first_count, second_count);
        (void)send_request_entry(
            async_ctx,
            UrmaTget(local_tget_tail, remote_send_tail, reinterpret_cast<__gm__ uint8_t *>(comm_ctx->workSpace), peer)
        );
    }
    marker[0] = static_cast<int32_t>(request_count);
}
