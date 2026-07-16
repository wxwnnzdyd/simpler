#include "urma_real_deferred_demo/kernels/aiv/urma_real_deferred_common.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *send_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *recv_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *status_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ CommContext *comm_ctx = reinterpret_cast<__gm__ CommContext *>(args[3]);
    uint32_t elem_count = static_cast<uint32_t>(args[4]);

    __gm__ float *send = urma_real_deferred::tensor_data<float>(send_tensor);
    __gm__ float *recv = urma_real_deferred::tensor_data<float>(recv_tensor);
    __gm__ int32_t *status = urma_real_deferred::tensor_data<int32_t>(status_tensor);

    if (!urma_real_deferred::validate_comm(comm_ctx, elem_count, status)) {
        return;
    }

    uint32_t peer = urma_real_deferred::peer_rank(comm_ctx);
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

    urma_real_deferred::set_status(status, urma_real_deferred::Status::kOk, elem_count, peer);
}
