#include <gtest/gtest.h>

#include <cstring>

#include "backend/urma/urma_completion_scheduler.h"

namespace {

using pto2::urma_backend::encode_fake_cqe_dw0;
using pto2::urma_backend::encode_fake_pending_cqe_dw0;
using pto2::urma_backend::encode_urma_event_handle;
using pto2::urma_backend::FakeUrmaWorkspace;
using pto2::urma_backend::kCqeBytes;
using pto2::urma_backend::kFakeUrmaCqDepth;
using pto2::urma_backend::kFakeUrmaCqeShift;
using pto2::urma_backend::poll_urma_event_handle;
using pto2::urma_backend::UrmaCqCtx;
using pto2::urma_backend::UrmaWqCtx;

struct SchedulerWorkspaceFixture {
    FakeUrmaWorkspace ws{};

    SchedulerWorkspaceFixture() {
        std::memset(&ws, 0, sizeof(ws));
        ws.info.qp_num = 1;
        ws.info.rank_count = 2;
        ws.info.sq_ptr = reinterpret_cast<uint64_t>(&ws.sq[0]);
        ws.info.scq_ptr = reinterpret_cast<uint64_t>(&ws.scq[0]);
        for (uint32_t rank = 0; rank < ws.info.rank_count; ++rank) {
            UrmaWqCtx &sq = ws.sq[rank];
            sq.wqe_shift_size = kFakeUrmaCqeShift;
            sq.depth = kFakeUrmaCqDepth;
            sq.tail_addr = reinterpret_cast<uint64_t>(&ws.sq_tail[rank]);

            UrmaCqCtx &cq = ws.scq[rank];
            cq.buf_addr = reinterpret_cast<uint64_t>(&ws.scq_entries[rank][0]);
            cq.cqe_shift_size = kFakeUrmaCqeShift;
            cq.depth = kFakeUrmaCqDepth;
            cq.tail_addr = reinterpret_cast<uint64_t>(&ws.cq_tail[rank]);
            cq.db_addr = reinterpret_cast<uint64_t>(&ws.cq_doorbell[rank]);
        }
    }

    uint64_t addr() { return reinterpret_cast<uint64_t>(&ws); }
};

}  // namespace

TEST(A5UrmaCompletionScheduler, ZeroHandleIsReady) {
    auto result = poll_urma_event_handle(0, 0);
    EXPECT_EQ(result.state, CompletionPollState::READY);
    EXPECT_EQ(result.error_code, PTO2_ERROR_NONE);
}

TEST(A5UrmaCompletionScheduler, NullWorkspaceFails) {
    auto result = poll_urma_event_handle(encode_urma_event_handle(1, 1), 0);
    EXPECT_EQ(result.state, CompletionPollState::FAILED);
    EXPECT_EQ(result.error_code, PTO2_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST(A5UrmaCompletionScheduler, InvalidRankFails) {
    SchedulerWorkspaceFixture fixture;
    auto result = poll_urma_event_handle(encode_urma_event_handle(3, 1), fixture.addr());
    EXPECT_EQ(result.state, CompletionPollState::FAILED);
    EXPECT_EQ(result.error_code, PTO2_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST(A5UrmaCompletionScheduler, TailAlreadyAtTargetIsReady) {
    SchedulerWorkspaceFixture fixture;
    fixture.ws.cq_tail[1] = 7;

    auto result = poll_urma_event_handle(encode_urma_event_handle(1, 7), fixture.addr());
    EXPECT_EQ(result.state, CompletionPollState::READY);
    EXPECT_EQ(result.error_code, PTO2_ERROR_NONE);
    EXPECT_EQ(fixture.ws.cq_doorbell[1], 0u);
    EXPECT_EQ(fixture.ws.sq_tail[1], 0u);
}

TEST(A5UrmaCompletionScheduler, OwnerNotReadyReturnsPending) {
    SchedulerWorkspaceFixture fixture;
    fixture.ws.scq_entries[1][0].dw[0] = encode_fake_pending_cqe_dw0(0);

    auto result = poll_urma_event_handle(encode_urma_event_handle(1, 1), fixture.addr());
    EXPECT_EQ(result.state, CompletionPollState::PENDING);
    EXPECT_EQ(result.error_code, PTO2_ERROR_NONE);
    EXPECT_EQ(fixture.ws.cq_tail[1], 0u);
    EXPECT_EQ(fixture.ws.cq_doorbell[1], 0u);
    EXPECT_EQ(fixture.ws.sq_tail[1], 0u);
}

TEST(A5UrmaCompletionScheduler, ReadyCqeAdvancesCqDoorbellAndSqTail) {
    SchedulerWorkspaceFixture fixture;
    fixture.ws.scq_entries[1][0].dw[0] = encode_fake_cqe_dw0(0, 0, 0);

    auto result = poll_urma_event_handle(encode_urma_event_handle(1, 1), fixture.addr());
    EXPECT_EQ(result.state, CompletionPollState::READY);
    EXPECT_EQ(result.error_code, PTO2_ERROR_NONE);
    EXPECT_EQ(fixture.ws.cq_tail[1], 1u);
    EXPECT_EQ(fixture.ws.cq_doorbell[1], 1u);
    EXPECT_EQ(fixture.ws.sq_tail[1], 1u);
}

TEST(A5UrmaCompletionScheduler, CqeErrorFailsWithoutAdvancingTail) {
    SchedulerWorkspaceFixture fixture;
    fixture.ws.scq_entries[1][0].dw[0] = encode_fake_cqe_dw0(0, 2, 9);

    auto result = poll_urma_event_handle(encode_urma_event_handle(1, 1), fixture.addr());
    EXPECT_EQ(result.state, CompletionPollState::FAILED);
    EXPECT_EQ(result.error_code, PTO2_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(fixture.ws.cq_tail[1], 0u);
    EXPECT_EQ(fixture.ws.cq_doorbell[1], 0u);
    EXPECT_EQ(fixture.ws.sq_tail[1], 0u);
}

static_assert(sizeof(pto2::urma_backend::FakeUrmaCqe) == kCqeBytes);
