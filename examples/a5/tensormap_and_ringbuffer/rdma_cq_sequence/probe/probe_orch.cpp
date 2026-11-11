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

// Step 1 orchestrator: submit exactly one task that writes the host status
// tensor. Mirrors the demo's shape (plain CoreTaskArgs, one output tensor).

#include <stdint.h>

#include "orchestration_api.h"

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig probe_orch_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{.expected_arg_count = 1};
}

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    return probe_orch_config(orch_args);
}

__attribute__((visibility("default"))) void probe_orch(const ChipTaskArgs &orch_args) {
    const simpler::tmr::Tensor &status = orch_args.tensor(0).ref();

    CoreTaskArgs args;
    args.add_output(status);
    rt_submit_aiv_task(0, args);
}

}  // extern "C"
