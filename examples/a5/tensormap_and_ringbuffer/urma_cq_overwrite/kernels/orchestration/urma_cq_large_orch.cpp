/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <stdint.h>

#include "platform_comm/comm_context.h"
#include "orchestration_api.h"

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig
urma_cq_large_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{.expected_arg_count = 5};
}

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    return urma_cq_large_orchestration_config(orch_args);
}

__attribute__((visibility("default"))) void urma_cq_large_orchestration(const ChipTaskArgs &orch_args) {
    if (orch_args.tensor_count() != 4 || orch_args.scalar_count() != 1) {
        LOG_ERROR("urma_cq_large: expected 4 tensors and 1 scalar");
        return;
    }

    const simpler::tmr::Tensor &input = orch_args.tensor(0).ref();
    const simpler::tmr::Tensor &out = orch_args.tensor(1).ref();
    const simpler::tmr::Tensor &probe = orch_args.tensor(2).ref();
    const simpler::tmr::Tensor &flag = orch_args.tensor(3).ref();
    auto *comm_ctx = reinterpret_cast<CommContext *>(orch_args.scalar<uintptr_t>(0));

    CoreTaskArgs victim_args;
    victim_args.add_input(input);
    victim_args.add_output(out);
    victim_args.add_input(flag);
    victim_args.add_scalar(reinterpret_cast<uint64_t>(comm_ctx));
    rt_submit_aiv_task(0, victim_args);

    CoreTaskArgs probe_args;
    probe_args.add_input(out);
    probe_args.add_output(probe);
    probe_args.add_scalar(reinterpret_cast<uint64_t>(comm_ctx));
    rt_submit_aiv_task(1, probe_args);
}

}  // extern "C"
