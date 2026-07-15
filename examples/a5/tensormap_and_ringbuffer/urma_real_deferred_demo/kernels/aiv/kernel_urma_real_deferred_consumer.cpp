#include "urma_real_deferred_common.h"

namespace {

inline __aicore__ bool tput_slot_matches(__gm__ float *slot, uint32_t elem_count, uint32_t peer) {
    for (uint32_t i = 0; i < elem_count; ++i) {
        float expected = static_cast<float>(peer * 100000 + static_cast<int>(i));
        if (slot[i] != expected) {
            return false;
        }
    }
    return true;
}

}  // namespace

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *tget_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *tput_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *marker_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *status_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ CommContext *comm_ctx = reinterpret_cast<__gm__ CommContext *>(args[4]);
    uint32_t elem_count = static_cast<uint32_t>(args[5]);

    __gm__ float *tget = urma_real_deferred::tensor_data<float>(tget_tensor);
    __gm__ float *tput = urma_real_deferred::tensor_data<float>(tput_tensor);
    __gm__ int32_t *marker = urma_real_deferred::tensor_data<int32_t>(marker_tensor);
    __gm__ int32_t *status = urma_real_deferred::tensor_data<int32_t>(status_tensor);

    if (!urma_real_deferred::validate_comm(comm_ctx, elem_count, status)) {
        return;
    }
    uint32_t peer = urma_real_deferred::peer_rank(comm_ctx);
    if (marker[0] != 1) {
        urma_real_deferred::set_status(status, urma_real_deferred::Status::kSubmitFailed, marker[0], peer);
        return;
    }

    for (uint32_t i = 0; i < elem_count; ++i) {
        float expected = static_cast<float>(peer * 100000 + static_cast<int>(i));
        if (tget[i] != expected) {
            urma_real_deferred::set_status(status, urma_real_deferred::Status::kTgetMismatch, i, peer);
            status[3] = static_cast<int32_t>(tget[i]);
            return;
        }
    }

    __gm__ float *peer_slot = tput + static_cast<uint64_t>(peer) * elem_count;
    for (uint32_t iter = 0; iter < urma_real_deferred::kMaxRemoteWritePollIters; ++iter) {
        if (tput_slot_matches(peer_slot, elem_count, peer)) {
            urma_real_deferred::set_status(status, urma_real_deferred::Status::kOk, elem_count, peer);
            return;
        }
    }

    urma_real_deferred::set_status(status, urma_real_deferred::Status::kTputMismatch, elem_count, peer);
}
