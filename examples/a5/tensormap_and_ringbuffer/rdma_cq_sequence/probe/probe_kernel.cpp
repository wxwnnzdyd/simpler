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

// Step 1: the smallest possible AIV kernel. No RDMA, no comm context, no
// deferred completion -- it only proves that a task can be dispatched and write
// into a host tensor.

#include <cstdint>

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include "tensor.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *status_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ int32_t *status =
        reinterpret_cast<__gm__ int32_t *>(status_tensor->buffer.addr) + status_tensor->start_offset;
    status[0] = 0;
    status[1] = 1234;
    status[2] = 42;
}
