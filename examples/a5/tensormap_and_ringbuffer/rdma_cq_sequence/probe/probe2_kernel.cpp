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

// Step 2: same argument shape and domain as the full case (6 tensors + 4
// scalars, five comm buffers) but the kernel only echoes the scalars into the
// status tensor. No RDMA: this isolates the shape/domain from the transport.

#include <cstdint>

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include "tensor.h"

namespace {

constexpr int kArgSend = 0;
constexpr int kArgRecv = 1;
constexpr int kArgTput = 2;
constexpr int kArgSignal = 3;
constexpr int kArgSummary = 4;
constexpr int kArgStatus = 5;
constexpr int kArgCommCtx = 6;
constexpr int kArgMode = 7;
constexpr int kArgPosts = 8;
constexpr int kArgTasks = 9;

constexpr int32_t kEntered = 0x5a5a5a5a;

}  // namespace

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *send_tensor = reinterpret_cast<__gm__ Tensor *>(args[kArgSend]);
    __gm__ Tensor *recv_tensor = reinterpret_cast<__gm__ Tensor *>(args[kArgRecv]);
    __gm__ Tensor *tput_tensor = reinterpret_cast<__gm__ Tensor *>(args[kArgTput]);
    __gm__ Tensor *signal_tensor = reinterpret_cast<__gm__ Tensor *>(args[kArgSignal]);
    __gm__ Tensor *summary_tensor = reinterpret_cast<__gm__ Tensor *>(args[kArgSummary]);
    __gm__ Tensor *status_tensor = reinterpret_cast<__gm__ Tensor *>(args[kArgStatus]);
    const uint64_t comm_ctx = static_cast<uint64_t>(args[kArgCommCtx]);
    const uint32_t mode = static_cast<uint32_t>(args[kArgMode]);
    const uint32_t posts = static_cast<uint32_t>(args[kArgPosts]);
    const uint32_t tasks = static_cast<uint32_t>(args[kArgTasks]);

    __gm__ int32_t *status =
        reinterpret_cast<__gm__ int32_t *>(status_tensor->buffer.addr) + status_tensor->start_offset;
    status[3] = kEntered;
    status[0] = 0;
    status[1] = static_cast<int32_t>(mode);
    status[2] = static_cast<int32_t>(tasks);

    // Touch every tensor argument so a wrong descriptor shows up as a bad
    // address rather than as an unused argument.
    __gm__ int32_t *send = reinterpret_cast<__gm__ int32_t *>(send_tensor->buffer.addr) + send_tensor->start_offset;
    __gm__ int32_t *recv = reinterpret_cast<__gm__ int32_t *>(recv_tensor->buffer.addr) + recv_tensor->start_offset;
    __gm__ int32_t *tput = reinterpret_cast<__gm__ int32_t *>(tput_tensor->buffer.addr) + tput_tensor->start_offset;
    __gm__ int32_t *signal =
        reinterpret_cast<__gm__ int32_t *>(signal_tensor->buffer.addr) + signal_tensor->start_offset;
    __gm__ int64_t *summary =
        reinterpret_cast<__gm__ int64_t *>(summary_tensor->buffer.addr) + summary_tensor->start_offset;
    send[0] = static_cast<int32_t>(posts);
    recv[0] = send[0];
    tput[0] = recv[0];
    signal[0] = tput[0];
    summary[0] = static_cast<int64_t>(comm_ctx != 0U);
}
