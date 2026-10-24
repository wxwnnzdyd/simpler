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
urma_cq_overwrite_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{.expected_arg_count = 7};
}

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    return urma_cq_overwrite_orchestration_config(orch_args);
}

__attribute__((visibility("default"))) void urma_cq_overwrite_orchestration(const ChipTaskArgs &orch_args) {
    if (orch_args.tensor_count() != 5 || orch_args.scalar_count() != 2) {
        LOG_ERROR("urma_cq_overwrite: expected 5 tensors and 2 scalars");
        return;
    }

    const simpler::tmr::Tensor &input = orch_args.tensor(0).ref();
    const simpler::tmr::Tensor &out = orch_args.tensor(1).ref();
    const simpler::tmr::Tensor &result = orch_args.tensor(2).ref();
    const simpler::tmr::Tensor &flag = orch_args.tensor(3).ref();
    const simpler::tmr::Tensor &scratch = orch_args.tensor(4).ref();
    auto *comm_ctx = reinterpret_cast<CommContext *>(orch_args.scalar<uintptr_t>(0));
    uint32_t wrap = static_cast<uint32_t>(orch_args.scalar<uint64_t>(1) != 0ULL ? 1U : 0U);

    CoreTaskArgs victim_args;
    victim_args.add_input(input);
    victim_args.add_output(out);
    victim_args.add_input(flag);
    victim_args.add_scalar(reinterpret_cast<uint64_t>(comm_ctx));
    rt_submit_aiv_task(0, victim_args);

    CoreTaskArgs filler_args;
    filler_args.add_input(input);
    filler_args.add_output(scratch);
    filler_args.add_input(flag);
    filler_args.add_scalar(reinterpret_cast<uint64_t>(comm_ctx));
    filler_args.add_scalar(static_cast<uint64_t>(wrap));
    rt_submit_aiv_task(1, filler_args);

    CoreTaskArgs consumer_args;
    consumer_args.add_input(out);
    consumer_args.add_output(result);
    rt_submit_aiv_task(2, consumer_args);
}

}  // extern "C"
