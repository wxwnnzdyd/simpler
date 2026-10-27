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

#include <gtest/gtest.h>

#include "backend/urma/urma_completion_scheduler.h"

using pto2::urma_backend::poll_urma_cqe_record;

TEST(UrmaCompletionScheduler, PollsOneCqeDw0WithoutWritingIt) {
    alignas(CHIP_ALIGN_SIZE) uint32_t dw0 = 0;

    EXPECT_EQ(poll_urma_cqe_record(/*cqe_addr=*/0, /*expected_owner=*/0).state, CompletionPollState::FAILED);
    EXPECT_EQ(
        poll_urma_cqe_record(reinterpret_cast<uint64_t>(&dw0), /*expected_owner=*/2).state, CompletionPollState::FAILED
    );

    EXPECT_EQ(
        poll_urma_cqe_record(reinterpret_cast<uint64_t>(&dw0), /*expected_owner=*/1).state, CompletionPollState::PENDING
    );
    EXPECT_EQ(dw0, 0U);

    dw0 = 1U << 2U;
    EXPECT_EQ(
        poll_urma_cqe_record(reinterpret_cast<uint64_t>(&dw0), /*expected_owner=*/1).state, CompletionPollState::READY
    );
    EXPECT_EQ(dw0, 1U << 2U);

    dw0 = (1U << 2U) | (1U << 24U);
    EXPECT_EQ(
        poll_urma_cqe_record(reinterpret_cast<uint64_t>(&dw0), /*expected_owner=*/1).state, CompletionPollState::FAILED
    );
    EXPECT_EQ(dw0, (1U << 2U) | (1U << 24U));

    dw0 = (1U << 2U) | (1U << 16U);
    EXPECT_EQ(
        poll_urma_cqe_record(reinterpret_cast<uint64_t>(&dw0), /*expected_owner=*/1).state, CompletionPollState::FAILED
    );
    EXPECT_EQ(dw0, (1U << 2U) | (1U << 16U));
}

TEST(UrmaCompletionScheduler, RetireWritesOnlyTheRequestedRange) {
    constexpr uint16_t kMask = 7U;  // depth = 8
    alignas(CHIP_ALIGN_SIZE) uint32_t ack[8] = {};

    // sequence 6 -> tail slot (6 - 1) & 7 = 5; ack_begin = 2.
    retire_urma_cqe_record(
        reinterpret_cast<uint64_t>(&ack[5]), /*ack_begin=*/2U, /*sequence=*/6U, kMask);

    EXPECT_EQ(ack[0], 0U);
    EXPECT_EQ(ack[1], 0U);
    EXPECT_EQ(ack[2], 3U);
    EXPECT_EQ(ack[3], 4U);
    EXPECT_EQ(ack[4], 5U);
    EXPECT_EQ(ack[5], 6U);
    EXPECT_EQ(ack[6], 0U);
    EXPECT_EQ(ack[7], 0U);
}

TEST(UrmaCompletionScheduler, RetireWrapsAcrossTheRingEnd) {
    constexpr uint16_t kMask = 3U;  // depth = 4
    alignas(CHIP_ALIGN_SIZE) uint32_t ack[4] = {};

    // sequence 7 -> tail slot (7 - 1) & 3 = 2; ack_begin = 4.
    retire_urma_cqe_record(
        reinterpret_cast<uint64_t>(&ack[2]), /*ack_begin=*/4U, /*sequence=*/7U, kMask);

    EXPECT_EQ(ack[0], 5U);
    EXPECT_EQ(ack[1], 6U);
    EXPECT_EQ(ack[2], 7U);
    EXPECT_EQ(ack[3], 0U);
}

TEST(UrmaCompletionScheduler, RetireSingleSlot) {
    constexpr uint16_t kMask = 7U;
    alignas(CHIP_ALIGN_SIZE) uint32_t ack[8] = {};

    // Single WQE: ack_begin = sequence - 1, so only the tail slot is written.
    retire_urma_cqe_record(
        reinterpret_cast<uint64_t>(&ack[4]), /*ack_begin=*/4U, /*sequence=*/5U, kMask);

    EXPECT_EQ(ack[4], 5U);
    EXPECT_EQ(ack[3], 0U);
    EXPECT_EQ(ack[5], 0U);
}

TEST(UrmaCompletionScheduler, FailedPollStillRetiresTheWholeRange) {
    alignas(CHIP_ALIGN_SIZE) uint32_t cqe = (1U << 2U) | (1U << 24U);  // owner ok, status bad
    EXPECT_EQ(
        poll_urma_cqe_record(reinterpret_cast<uint64_t>(&cqe), /*expected_owner=*/1).state,
        CompletionPollState::FAILED);

    constexpr uint16_t kMask = 7U;
    alignas(CHIP_ALIGN_SIZE) uint32_t ack[8] = {};
    retire_urma_cqe_record(
        reinterpret_cast<uint64_t>(&ack[5]), /*ack_begin=*/2U, /*sequence=*/6U, kMask);

    EXPECT_EQ(ack[2], 3U);
    EXPECT_EQ(ack[3], 4U);
    EXPECT_EQ(ack[4], 5U);
    EXPECT_EQ(ack[5], 6U);
}
