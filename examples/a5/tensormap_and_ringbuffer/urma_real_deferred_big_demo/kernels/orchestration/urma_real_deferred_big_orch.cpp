#include <stdint.h>

#include "platform_comm/comm_context.h"
#include "pto_orchestration_api.h"

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
urma_real_deferred_big_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{.expected_arg_count = 5};
}

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    return urma_real_deferred_big_orchestration_config(orch_args);
}

__attribute__((visibility("default"))) void urma_real_deferred_big_orchestration(const L2TaskArgs &orch_args) {
    if (orch_args.tensor_count() + orch_args.scalar_count() != 5) {
        LOG_ERROR("urma_real_deferred_big_demo: expected 5 args");
        return;
    }

    const Tensor &send = orch_args.tensor(0).ref();
    const Tensor &recv = orch_args.tensor(1).ref();
    const Tensor &status = orch_args.tensor(2).ref();
    auto *comm_ctx = reinterpret_cast<CommContext *>(static_cast<uintptr_t>(orch_args.scalar(0)));
    const uint32_t elem_count = static_cast<uint32_t>(orch_args.scalar(1));

    L0TaskArgs tget_args;
    tget_args.add_input(send);
    tget_args.add_output(recv);
    tget_args.add_output(status);
    tget_args.add_scalar(reinterpret_cast<uint64_t>(comm_ctx));
    tget_args.add_scalar(elem_count);
    rt_submit_aiv_task(0, tget_args);
}

}  // extern "C"
