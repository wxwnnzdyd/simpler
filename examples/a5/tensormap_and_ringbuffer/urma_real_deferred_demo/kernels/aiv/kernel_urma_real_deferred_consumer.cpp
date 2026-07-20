#include "urma_real_deferred_common.h"

namespace {

inline __aicore__ bool tput_slot_matches(
    __gm__ float *slot, uint32_t elem_count, uint32_t peer, __gm__ int32_t *status
) {
    for (uint32_t i = 0; i < elem_count; ++i) {
        float expected = static_cast<float>(peer * 100000 + static_cast<int>(i));
        if (slot[i] != expected) {
            urma_real_deferred::set_status(status, urma_real_deferred::Status::kTputMismatch, i, peer);
            status[3] = static_cast<int32_t>(slot[i]);
            status[4] = static_cast<int32_t>(expected);
            status[5] = static_cast<int32_t>(slot[0]);
            status[6] = static_cast<int32_t>(slot[elem_count - 1]);
            return false;
        }
    }
    return true;
}

inline __aicore__ bool tget_matches(
    __gm__ float *slot, uint32_t elem_count, uint32_t peer, __gm__ int32_t *status, int32_t which,
    int32_t marker_sum, int32_t expected_marker_sum, __gm__ float *other_slot
) {
    for (uint32_t i = 0; i < elem_count; ++i) {
        float expected = static_cast<float>(peer * 100000 + static_cast<int>(i));
        if (slot[i] != expected) {
            urma_real_deferred::set_status(status, urma_real_deferred::Status::kTgetMismatch, i, peer);
            status[3] = static_cast<int32_t>(slot[i]);
            status[4] = which;
            status[5] = marker_sum;
            status[6] = expected_marker_sum;
            status[7] = static_cast<int32_t>(other_slot[0]);
            return false;
        }
    }
    return true;
}

}  // namespace

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *tget0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *tget1_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *tput_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *tget0_marker_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ Tensor *tget1_marker_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ Tensor *tput0_marker_tensor = reinterpret_cast<__gm__ Tensor *>(args[5]);
    __gm__ Tensor *tput1_marker_tensor = reinterpret_cast<__gm__ Tensor *>(args[6]);
    __gm__ Tensor *status_tensor = reinterpret_cast<__gm__ Tensor *>(args[7]);
    __gm__ CommContext *comm_ctx = reinterpret_cast<__gm__ CommContext *>(args[8]);
    uint32_t elem_count = static_cast<uint32_t>(args[9]);

    __gm__ float *tget0 = urma_real_deferred::tensor_data<float>(tget0_tensor);
    __gm__ float *tget1 = urma_real_deferred::tensor_data<float>(tget1_tensor);
    __gm__ float *tput = urma_real_deferred::tensor_data<float>(tput_tensor);
    __gm__ int32_t *tget0_marker = urma_real_deferred::tensor_data<int32_t>(tget0_marker_tensor);
    __gm__ int32_t *tget1_marker = urma_real_deferred::tensor_data<int32_t>(tget1_marker_tensor);
    __gm__ int32_t *tput0_marker = urma_real_deferred::tensor_data<int32_t>(tput0_marker_tensor);
    __gm__ int32_t *tput1_marker = urma_real_deferred::tensor_data<int32_t>(tput1_marker_tensor);
    __gm__ int32_t *status = urma_real_deferred::tensor_data<int32_t>(status_tensor);

    if (!urma_real_deferred::validate_comm(comm_ctx, elem_count, status)) {
        return;
    }
    uint32_t peer = urma_real_deferred::peer_rank(comm_ctx);
    int32_t expected_marker_sum = static_cast<int32_t>((elem_count > 1 ? 2 : 1) * 2);
    int32_t tget_marker_sum = tget0_marker[0] + tget1_marker[0];

    if (!tget_matches(tget0, elem_count, peer, status, 0, tget_marker_sum, expected_marker_sum, tget1)) {
        return;
    }
    if (!tget_matches(tget1, elem_count, peer, status, 1, tget_marker_sum, expected_marker_sum, tget0)) {
        return;
    }

    uint64_t peer_base = urma_real_deferred::remote_base(comm_ctx, peer);
    uint64_t tput_slot_offset = urma_real_deferred::local_offset(comm_ctx, tput) +
                                static_cast<uint64_t>(comm_ctx->rankId) * elem_count * sizeof(float);
    __gm__ float *remote_tput_slot = reinterpret_cast<__gm__ float *>(peer_base + tput_slot_offset);
    __gm__ float *scratch = tput + static_cast<uint64_t>(comm_ctx->rankNum) * elem_count;
    auto local_scratch = urma_real_deferred::global_float(scratch, elem_count);
    auto remote_tput = urma_real_deferred::global_float(remote_tput_slot, elem_count);
    pto::comm::AsyncSession readback_session;
    pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(
        reinterpret_cast<__gm__ uint8_t *>(comm_ctx->workSpace), peer, readback_session
    );
    auto readback_event =
        pto::comm::TGET_ASYNC<pto::comm::DmaEngine::URMA>(local_scratch, remote_tput, readback_session);
    if (!urma_real_deferred::wait_urma_bounded(readback_event, readback_session)) {
        urma_real_deferred::set_status(status, urma_real_deferred::Status::kTputReadbackFailed, elem_count, peer);
        return;
    }

    if (tput_slot_matches(scratch, elem_count, comm_ctx->rankId, status)) {
        urma_real_deferred::set_status(status, urma_real_deferred::Status::kOk, elem_count, peer);
        status[3] = tget_marker_sum;
        status[4] = tput0_marker[0] + tput1_marker[0];
        status[5] = expected_marker_sum;
        return;
    }
}
