from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
KERNEL = (
    REPO_ROOT
    / "examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/kernels/aiv/kernel_urma_real_async.cpp"
)
TEST_PY = REPO_ROOT / "examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/test_urma_real_async_demo.py"
ORCH = (
    REPO_ROOT
    / "examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/kernels/orchestration/urma_real_async_orch.cpp"
)


def test_phase3_kernel_does_not_bypass_ranks() -> None:
    source = KERNEL.read_text()
    assert "kProbeRankBypass" not in source


def test_phase3_kernel_failfasts_known_unsafe_wqe_access() -> None:
    source = KERNEL.read_text()
    assert "kUnsafeWqeAccess = 60" in source
    assert "UrmaGetAsyncViaMte" not in source
    assert "UrmaPutAsyncViaMte" not in source
    assert "copy_ubuf_to_gm_align_v2" not in source
    assert "TGET_ASYNC<pto::comm::DmaEngine::URMA>" not in source
    assert "TPUT_ASYNC<pto::comm::DmaEngine::URMA>" not in source


def test_phase3_wqe_write_probes_do_not_touch_wqe_memory() -> None:
    source = KERNEL.read_text()
    for unsafe in [
        "uint64_t old_wqe_word0 = *wqe_word0",
        "*reinterpret_cast<__gm__ uint32_t *>(wqe_addr) = 0",
        "st_dev(0U, reinterpret_cast<__gm__ uint32_t *>(wqe_addr), 0)",
        "*wqe_word0 = old_wqe_word0 ^ 1ULL",
    ]:
        assert unsafe not in source


def test_phase3_demo_exposes_probe_stage_cli_and_passes_it_to_kernel() -> None:
    test_source = TEST_PY.read_text()
    orch_source = ORCH.read_text()
    kernel_source = KERNEL.read_text()

    assert "PROBE_STAGES" in test_source
    assert 'parser.add_argument("--probe-stage"' in test_source
    assert "args.add_scalar(probe_stage)" in test_source
    assert "expected_arg_count = 8" in orch_source
    assert "params.add_scalar(probe_stage)" in orch_source
    assert "uint32_t probe_stage = static_cast<uint32_t>(args[7])" in kernel_source
    assert "enum class ProbeStage" in kernel_source
    for stage in [
        "kWorkspace",
        "kBuildSession",
        "kWorkspaceInfo",
        "kWqCtx",
        "kQueueIndexRead",
        "kWqeRead",
        "kWqeWriteRestore",
        "kRemoteMem",
        "kEidRead",
        "kTgetPost",
        "kTgetTestOnce",
        "kTputPost",
        "kTputTestOnce",
        "kQueueIndexLdDev",
        "kWqeAddr",
        "kWqeFirstStore",
        "kWqeFirstStDev",
        "kWqeMteStore",
    ]:
        assert stage in kernel_source


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


def test_phase3_comm_hccl_logs_hcomm_route_diagnostics() -> None:
    source = (REPO_ROOT / "src/a5/platform/onboard/host/comm_hccl.cpp").read_text()
    assert "link[%u/%u] protocol=%s(%d)" in source
    assert "selected channel protocol=%s(%d)" in source
    assert "entity peer=%u protocol=%s(%d)" in source
    assert "SqContext type=%s(%d)" in source
    assert "CqContext type=%s(%d)" in source
    assert "ProtocolName" in source
