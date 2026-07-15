#include "urma_real_deferred_common.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *send_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *tput_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *marker_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ CommContext *comm_ctx = reinterpret_cast<__gm__ CommContext *>(args[3]);
    uint32_t elem_count = static_cast<uint32_t>(args[4]);

    __gm__ float *send = urma_real_deferred::tensor_data<float>(send_tensor);
    __gm__ float *tput = urma_real_deferred::tensor_data<float>(tput_tensor);
    __gm__ int32_t *marker = urma_real_deferred::tensor_data<int32_t>(marker_tensor);
    uint32_t peer = urma_real_deferred::peer_rank(comm_ctx);
    uint64_t peer_base = urma_real_deferred::remote_base(comm_ctx, peer);
    uint64_t tput_slot_offset = urma_real_deferred::local_offset(comm_ctx, tput) +
                                static_cast<uint64_t>(comm_ctx->rankId) * elem_count * sizeof(float);
    __gm__ float *remote_tput_slot = reinterpret_cast<__gm__ float *>(peer_base + tput_slot_offset);

    auto local_send = urma_real_deferred::global_float(send, elem_count);
    auto remote_tput = urma_real_deferred::global_float(remote_tput_slot, elem_count);
    AsyncCtx async_ctx = get_async_ctx(args);
    (void)send_request_entry(
        async_ctx, UrmaTput(remote_tput, local_send, reinterpret_cast<__gm__ uint8_t *>(comm_ctx->workSpace), peer)
    );
    marker[0] = 1;
}
