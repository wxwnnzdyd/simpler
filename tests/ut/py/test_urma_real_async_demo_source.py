from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
KERNEL = (
    REPO_ROOT
    / "examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/kernels/aiv/kernel_urma_real_async.cpp"
)


def test_phase3_kernel_reaches_real_urma_submit_for_all_ranks() -> None:
    source = KERNEL.read_text()
    assert "kProbeRankBypass" not in source

    tget_submit = source.index("auto tget_event")
    urma_block_start = source.rindex("#ifdef PTO_URMA_SUPPORTED", 0, tget_submit)
    submit_preamble = source[urma_block_start:tget_submit]
    assert "return;" not in submit_preamble


def test_phase3_kernel_uses_bounded_waits_for_real_tget_and_tput() -> None:
    source = KERNEL.read_text()
    assert "TGET_ASYNC<pto::comm::DmaEngine::URMA>" in source
    assert "TPUT_ASYNC<pto::comm::DmaEngine::URMA>" in source
    assert "WaitUrmaBounded(tget_event, tget_session" in source
    assert "WaitUrmaBounded(tput_event, tput_session" in source
    assert "tget_event.Wait(tget_session)" not in source
    assert "tput_event.Wait(tput_session)" not in source
    assert "DeviceBarrierBounded" in source


def test_phase2_workspace_smoke_asserts_a5_acceptance_points() -> None:
    test_source = (REPO_ROOT / "tests/ut/py/test_worker/test_platform_comm.py").read_text()
    assert '@pytest.mark.platforms(["a2a3", "a5"])' in test_source
    assert "if platform == \"a5\":" in test_source
    assert "a5 base CommContext.workSpace is 0" in test_source
    assert "a5 base CommContext.workSpaceSize is 0" in test_source
    assert "dense derived workSpace" in test_source
    assert "non-dense derived context must disable URMA workspace" in test_source


def test_phase2_comm_hccl_initializes_and_propagates_urma_workspace() -> None:
    source = (REPO_ROOT / "src/a5/platform/onboard/host/comm_hccl.cpp").read_text()
    assert "ensure_base_urma_workspace(h);" in source
    assert "h->host_ctx.workSpace = reinterpret_cast<uint64_t>(h->urma_workspace->GetWorkspaceAddr())" in source
    assert "h->host_ctx.workSpaceSize = urma_workspace_bytes" in source
    assert "ctx.workSpace = reinterpret_cast<uint64_t>(out->urma_workspace->GetWorkspaceAddr())" in source
    assert "ctx.workSpace = h->host_ctx.workSpace" in source
    assert "rank_ids_are_dense_prefix" in source
    assert "URMA workspace disabled for non-dense rank mapping" in source
