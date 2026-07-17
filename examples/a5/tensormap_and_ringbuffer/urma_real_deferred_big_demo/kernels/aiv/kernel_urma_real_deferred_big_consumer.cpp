#include "urma_real_deferred_demo/kernels/aiv/urma_real_deferred_common.h"

namespace {

inline __aicore__ uint32_t first_urma_chunk_elems() { return (256u * 1024u * 1024u) / sizeof(float); }

inline __aicore__ float expected_value(uint32_t rank, uint32_t idx) {
    return static_cast<float>(rank * 100000 + static_cast<int>(idx & 0xFFFFu));
}

inline __aicore__ bool check_one(
    __gm__ float *recv, uint32_t idx, uint32_t peer, __gm__ int32_t *status, int32_t which
) {
    float got = recv[idx];
    float expected = expected_value(peer, idx);
    if (got != expected) {
        urma_real_deferred::set_status(status, urma_real_deferred::Status::kTgetMismatch, static_cast<int32_t>(idx), peer);
        status[3] = static_cast<int32_t>(got);
        status[4] = static_cast<int32_t>(expected);
        status[5] = which;
        return false;
    }
    return true;
}

}  // namespace

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *recv_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *marker_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *status_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ CommContext *comm_ctx = reinterpret_cast<__gm__ CommContext *>(args[3]);
    uint32_t elem_count = static_cast<uint32_t>(args[4]);

    __gm__ float *recv = urma_real_deferred::tensor_data<float>(recv_tensor);
    __gm__ int32_t *marker = urma_real_deferred::tensor_data<int32_t>(marker_tensor);
    __gm__ int32_t *status = urma_real_deferred::tensor_data<int32_t>(status_tensor);

    if (!urma_real_deferred::validate_comm(comm_ctx, elem_count, status)) {
        return;
    }
    uint32_t peer = urma_real_deferred::peer_rank(comm_ctx);
    if (marker[0] != 2) {
        urma_real_deferred::set_status(status, urma_real_deferred::Status::kSubmitFailed, marker[0], peer);
        return;
    }

    if (!check_one(recv, 0, peer, status, 0)) return;
    uint32_t chunk_tail = first_urma_chunk_elems() - 1u;
    if (chunk_tail < elem_count && !check_one(recv, chunk_tail, peer, status, 1)) return;
    uint32_t chunk_head = first_urma_chunk_elems();
    if (chunk_head < elem_count && !check_one(recv, chunk_head, peer, status, 2)) return;
    if (!check_one(recv, elem_count - 1u, peer, status, 3)) return;

    urma_real_deferred::set_status(status, urma_real_deferred::Status::kOk, elem_count, peer);
    status[3] = marker[0];
    status[4] = static_cast<int32_t>(first_urma_chunk_elems());
}
