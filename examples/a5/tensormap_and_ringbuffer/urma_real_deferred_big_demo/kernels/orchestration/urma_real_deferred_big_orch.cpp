#include <stdint.h>

#include "platform_comm/comm_context.h"
#include "pto_orchestration_api.h"

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
urma_real_deferred_big_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{.expected_arg_count = 6};
}

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    return urma_real_deferred_big_orchestration_config(orch_args);
}

__attribute__((visibility("default"))) void urma_real_deferred_big_orchestration(const L2TaskArgs &orch_args) {
    if (orch_args.tensor_count() + orch_args.scalar_count() != 6) {
        LOG_ERROR("urma_real_deferred_big_demo: expected 6 args");
        return;
    }

    const Tensor &send = orch_args.tensor(0).ref();
    const Tensor &recv = orch_args.tensor(1).ref();
    const Tensor &signal = orch_args.tensor(2).ref();
    const Tensor &status = orch_args.tensor(3).ref();
    auto *comm_ctx = reinterpret_cast<CommContext *>(static_cast<uintptr_t>(orch_args.scalar(0)));
    const uint32_t elem_count = static_cast<uint32_t>(orch_args.scalar(1));

    uint32_t marker_shape[1] = {1};
    TensorCreateInfo marker_info(marker_shape, 1, DataType::INT32);

    L0TaskArgs tget_args;
    tget_args.add_output(send);
    tget_args.add_output(recv);
    tget_args.add_output(signal);
    tget_args.add_output(marker_info);
    tget_args.add_output(status);
    tget_args.add_scalar(reinterpret_cast<uint64_t>(comm_ctx));
    tget_args.add_scalar(elem_count);
    TaskOutputTensors tget_outputs = rt_submit_aiv_task(0, tget_args);
    Tensor marker = tget_outputs.get_ref(3);

    L0TaskArgs consumer_args;
    consumer_args.add_input(recv);
    consumer_args.add_input(marker);
    consumer_args.add_output(status);
    consumer_args.add_scalar(reinterpret_cast<uint64_t>(comm_ctx));
    consumer_args.add_scalar(elem_count);
    rt_submit_aiv_task(1, consumer_args);
}

}  // extern "C"
