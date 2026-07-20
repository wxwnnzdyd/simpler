#include <stdint.h>

#include "platform_comm/comm_context.h"
#include "pto_orchestration_api.h"

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
urma_real_deferred_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{.expected_arg_count = 6};
}

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    return urma_real_deferred_orchestration_config(orch_args);
}

__attribute__((visibility("default"))) void urma_real_deferred_orchestration(const L2TaskArgs &orch_args) {
    if (orch_args.tensor_count() + orch_args.scalar_count() != 6) {
        LOG_ERROR("urma_real_deferred_demo: expected 6 args");
        return;
    }

    const Tensor &send = orch_args.tensor(0).ref();
    const Tensor &tget_recv = orch_args.tensor(1).ref();
    const Tensor &tput_recv = orch_args.tensor(2).ref();
    const Tensor &status = orch_args.tensor(3).ref();
    auto *comm_ctx = reinterpret_cast<CommContext *>(static_cast<uintptr_t>(orch_args.scalar(0)));
    const uint32_t elem_count = static_cast<uint32_t>(orch_args.scalar(1));

    uint32_t output_shape[1] = {elem_count};
    uint32_t marker_shape[1] = {1};

    L0TaskArgs tget0_args;
    TensorCreateInfo tget_output_info(output_shape, 1, DataType::FLOAT32);
    TensorCreateInfo tget_marker_info(marker_shape, 1, DataType::INT32);
    tget0_args.add_input(send);
    tget0_args.add_output(tget_output_info);
    tget0_args.add_output(tget_marker_info);
    tget0_args.add_scalar(reinterpret_cast<uint64_t>(comm_ctx));
    tget0_args.add_scalar(elem_count);
    TaskOutputTensors tget0_outputs = rt_submit_aiv_task(0, tget0_args);
    Tensor tget0_tmp = tget0_outputs.get_ref(0);
    Tensor tget0_marker = tget0_outputs.get_ref(1);

    L0TaskArgs tget1_args;
    tget1_args.add_input(send);
    tget1_args.add_output(tget_output_info);
    tget1_args.add_output(tget_marker_info);
    tget1_args.add_scalar(reinterpret_cast<uint64_t>(comm_ctx));
    tget1_args.add_scalar(elem_count);
    TaskOutputTensors tget1_outputs = rt_submit_aiv_task(0, tget1_args);
    Tensor tget1_tmp = tget1_outputs.get_ref(0);
    Tensor tget1_marker = tget1_outputs.get_ref(1);

    L0TaskArgs tput0_args;
    TensorCreateInfo marker_info(marker_shape, 1, DataType::INT32);
    tput0_args.add_input(send);
    tput0_args.add_input(tput_recv);
    tput0_args.add_output(marker_info);
    tput0_args.add_scalar(reinterpret_cast<uint64_t>(comm_ctx));
    tput0_args.add_scalar(elem_count);
    TaskOutputTensors tput0_outputs = rt_submit_aiv_task(1, tput0_args);
    Tensor tput0_marker = tput0_outputs.get_ref(0);

    L0TaskArgs tput1_args;
    tput1_args.add_input(send);
    tput1_args.add_input(tput_recv);
    tput1_args.add_output(marker_info);
    tput1_args.add_scalar(reinterpret_cast<uint64_t>(comm_ctx));
    tput1_args.add_scalar(elem_count);
    TaskOutputTensors tput1_outputs = rt_submit_aiv_task(1, tput1_args);
    Tensor tput1_marker = tput1_outputs.get_ref(0);

    L0TaskArgs consumer_args;
    consumer_args.add_input(tget0_tmp);
    consumer_args.add_input(tget1_tmp);
    consumer_args.add_input(tput_recv);
    consumer_args.add_input(tget0_marker);
    consumer_args.add_input(tget1_marker);
    consumer_args.add_input(tput0_marker);
    consumer_args.add_input(tput1_marker);
    consumer_args.add_output(status);
    consumer_args.add_scalar(reinterpret_cast<uint64_t>(comm_ctx));
    consumer_args.add_scalar(elem_count);
    rt_submit_aiv_task(2, consumer_args);
}

}  // extern "C"
