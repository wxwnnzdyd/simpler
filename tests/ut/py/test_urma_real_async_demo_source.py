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
DEFERRED_DEMO = REPO_ROOT / "examples/a5/tensormap_and_ringbuffer/urma_real_deferred_demo"
DEFERRED_COMMON = DEFERRED_DEMO / "kernels/aiv/urma_real_deferred_common.h"
DEFERRED_TGET = DEFERRED_DEMO / "kernels/aiv/kernel_urma_real_deferred_tget.cpp"
DEFERRED_TPUT = DEFERRED_DEMO / "kernels/aiv/kernel_urma_real_deferred_tput.cpp"
DEFERRED_CONSUMER = DEFERRED_DEMO / "kernels/aiv/kernel_urma_real_deferred_consumer.cpp"
DEFERRED_ORCH = DEFERRED_DEMO / "kernels/orchestration/urma_real_deferred_orch.cpp"
DEFERRED_TEST_PY = DEFERRED_DEMO / "test_urma_real_deferred_demo.py"
DEFERRED_BIG_DEMO = REPO_ROOT / "examples/a5/tensormap_and_ringbuffer/urma_real_deferred_big_demo"
DEFERRED_BIG_RUN = DEFERRED_BIG_DEMO / "run_urma_real_deferred_big_demo.py"
DEFERRED_BIG_TGET = DEFERRED_BIG_DEMO / "kernels/aiv/kernel_urma_real_deferred_big_tget.cpp"
DEFERRED_BIG_ORCH = DEFERRED_BIG_DEMO / "kernels/orchestration/urma_real_deferred_big_orch.cpp"
URMA_BACKEND_DESIGN = (
    REPO_ROOT / "src/a5/runtime/tensormap_and_ringbuffer/runtime/backend/urma/design.md"
)
URMA_BACKEND_KERNEL = (
    REPO_ROOT / "src/a5/runtime/tensormap_and_ringbuffer/runtime/backend/urma/urma_completion_kernel.h"
)
STATUS_DOC = REPO_ROOT / "docs/urma-deferred-completion-status.md"


def test_phase3_kernel_does_not_bypass_ranks() -> None:
    source = KERNEL.read_text()
    assert "kProbeRankBypass" not in source


def test_phase3_kernel_failfasts_known_unsafe_wqe_access() -> None:
    source = KERNEL.read_text()
    assert "#undef MEMORY_BASE" in source
    assert "#define REGISTER_BASE" in source
    assert "kUnsafeWqeAccess = 60" in source
    assert "UrmaGetAsyncViaMte" not in source
    assert "UrmaPutAsyncViaMte" not in source
    assert "copy_ubuf_to_gm_align_v2" not in source
    assert "TGET_ASYNC<pto::comm::DmaEngine::URMA>" in source
    assert "TPUT_ASYNC<pto::comm::DmaEngine::URMA>" in source


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
        "kTgetRootPost",
        "kTputRootPost",
        "kQueueIndexLdDev",
        "kWqeAddr",
        "kWqeFirstStore",
        "kWqeFirstStDev",
        "kWqeMteStore",
    ]:
        assert stage in kernel_source


def test_phase3_hardware_pytest_runs_real_full_submit_by_default() -> None:
    test_source = TEST_PY.read_text()

    assert "SAFE_PROBE_STAGES" in test_source
    assert "UNSAFE_PROBE_STAGES" in test_source
    assert "REAL_SUBMIT_PROBE_STAGES" in test_source
    assert 'parser.add_argument("--probe-stage", choices=("suite", *PROBE_STAGES)' in test_source
    assert "def run_probe_suite(" in test_source
    assert "assert run(st_platform, [int(st_device_ids[0]), int(st_device_ids[1])]) == 0" in test_source
    assert "probe_stage: int | None = None" in test_source
    assert "if probe_stage is None:" in test_source
    assert 'probe_stage = PROBE_STAGES["full"]' in test_source
    assert "kUnsafeWqeAccess" in test_source


def test_phase3_urma_demo_keeps_native_data_offset() -> None:
    test_source = TEST_PY.read_text()

    assert "URMA_DATA_OFFSET_NBYTES = 64 * 4" in test_source
    assert "URMA_DATA_OFFSET_NBYTES + send_nbytes" in test_source
    assert 'name="urma_reserved"' in test_source
    assert "URMA_DATA_OFFSET_NBYTES // 4" in test_source


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
    assert "if (!ensure_base_urma_workspace(h)) return -1;" in source
    assert "h->host_ctx.workSpace = reinterpret_cast<uint64_t>(h->urma_workspace->GetWorkspaceAddr())" in source
    assert "h->host_ctx.workSpaceSize = urma_workspace_bytes" in source
    assert "if (rank_ids_are_dense_prefix(rank_ids, rank_count))" in source
    assert "ctx.workSpace = h->host_ctx.workSpace" in source
    assert "ctx.workSpaceSize = h->host_ctx.workSpaceSize" in source
    assert "rank_ids_are_dense_prefix" in source
    assert "URMA workspace disabled for non-dense rank mapping" in source


def test_phase3_comm_hccl_uses_native_urma_workspace_manager() -> None:
    source = (REPO_ROOT / "src/a5/platform/onboard/host/comm_hccl.cpp").read_text()
    cmake = (REPO_ROOT / "src/a5/platform/onboard/host/CMakeLists.txt").read_text()
    assert '#include "pto/comm/async/urma/urma_workspace_manager.hpp"' in source
    assert "pto::comm::urma::UrmaWorkspaceManager" in source
    assert "class A5UrmaWorkspaceManager" not in source
    assert "ResolveDeviceChannelEntity" not in source
    assert "refusing private conversion" not in source
    assert "BuildChannelEntityToDevice" not in source
    assert "GetUserRemoteMem" not in source
    assert "set(HCCL_LINK_TARGETS ${HCCL_LIB} ${HCOMM_LIB})" in cmake


def test_phase3_comm_alloc_windows_fails_without_urma_workspace() -> None:
    source = (REPO_ROOT / "src/a5/platform/onboard/host/comm_hccl.cpp").read_text()

    assert "static bool ensure_base_urma_workspace(CommHandle h)" in source
    assert "if (!ensure_base_urma_workspace(h)) return -1;" in source
    assert "CommContext::workSpace remains 0" not in source


def test_phase3_comm_hccl_keeps_helper_namespace_closed_before_c_api() -> None:
    source = (REPO_ROOT / "src/a5/platform/onboard/host/comm_hccl.cpp").read_text()

    assert source.count("namespace {") == source.count("}  // namespace")
    assert source.rfind("}  // namespace", 0, source.index('extern "C" int comm_alloc_windows')) != -1


def test_phase3_full_probe_uses_native_root_only_urma_pattern() -> None:
    source = (REPO_ROOT / "examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/kernels/aiv/kernel_urma_real_async.cpp").read_text()

    full_branch = source[source.index("const bool full_probe") : source.index("__gm__ pto::comm::urma::UrmaWQCtx")]
    assert "(tput_root_post || full_probe) && my_rank != 0" in full_branch
    assert "tget_root_post || full_probe" in full_branch
    assert "tput_root_post || full_probe" in full_branch
    assert "DeviceBarrierBounded" not in source
    assert "CommRemotePtr" not in source
    assert "kBarrierWaitFailed" not in source


def test_phase3_status_doc_marks_real_urma_correctness_complete() -> None:
    source = STATUS_DOC.read_text()

    phase3 = source[source.index("## Phase 3: Standalone Real URMA Correctness") : source.index("## Phase 4")]
    assert "Status: complete" in phase3
    assert "UrmaPeerMrBaseAddr(workspace, peer) + offset" in phase3
    assert "does not prove AICPU deferred CQ polling" in phase3


def test_phase4_real_deferred_demo_uses_deferred_urma_backend() -> None:
    tget = DEFERRED_TGET.read_text()
    tput = DEFERRED_TPUT.read_text()
    common = DEFERRED_COMMON.read_text()
    orch = DEFERRED_ORCH.read_text()
    backend = URMA_BACKEND_KERNEL.read_text()
    test_py = DEFERRED_TEST_PY.read_text()
    run_source = test_py[test_py.index("def run(") : test_py.index("\n\n@pytest.mark.requires_hardware")]

    assert "backend/urma/urma_completion_kernel.h" in common
    assert "CommContext" in common
    assert "comm_ctx->workSpace" in tget
    assert "comm_ctx->workSpace" in tput
    assert "peer_mr_base_addr" in backend
    assert "peer_mr_ptr" in backend
    assert "UrmaPeerMrBaseAddr" in backend
    assert "pto2::urma_backend::peer_mr_base_addr" in common
    assert "UrmaPeerMrBaseAddr" not in common
    assert "send_request_entry" in tget
    assert "send_request_entry" in tput
    assert tget.count("send_request_entry") >= 2
    assert tput.count("send_request_entry") >= 2
    assert "UrmaTget" in tget
    assert "UrmaTput" in tput
    assert "first_chunk_count" in common
    assert "second_chunk_count" in common
    assert "store_marker" in common
    assert "defer_flush_range(marker" in common
    assert "event.Wait" not in tget
    assert "event.Wait" not in tput
    assert "TPUT_ASYNC" not in tget
    assert "TPUT_ASYNC" not in tput
    assert "TGET_ASYNC" not in tget
    assert "TGET_ASYNC" not in tput
    assert "rt_submit_aiv_task(0" in orch
    assert "rt_submit_aiv_task(1" in orch
    assert "rt_submit_aiv_task(2" in orch
    assert "Tensor tget0_marker" in orch
    assert "Tensor tget1_marker" in orch
    assert "TaskOutputTensors tput0_outputs" in orch
    assert "TaskOutputTensors tput1_outputs" in orch
    assert "def run_case(" in test_py
    assert "for elem_count in CASES:" in test_py
    assert "ok = run_case(platform, device_ids, elem_count, build=build) and ok" in run_source
    assert "Worker(" not in run_source
    assert "build_chip_callable" not in run_source
    assert "worker = Worker(" in test_py
    assert "worker.init()" in test_py
    assert "worker.close()" in test_py


def test_phase5_urma_backend_exposes_chunked_submission_helpers() -> None:
    backend = URMA_BACKEND_KERNEL.read_text()

    assert "kUrmaMaxTransferBytes" in backend
    assert "chunk_count" in backend
    assert "submit_chunked_urma_request" in backend
    assert "send_request_entry" in backend


def test_phase5_real_deferred_big_smoke_exercises_backend_chunking() -> None:
    run_source = DEFERRED_BIG_RUN.read_text()
    kernel = DEFERRED_BIG_TGET.read_text()
    orch = DEFERRED_BIG_ORCH.read_text()

    assert "URMA_SINGLE_WQE_FLOATS = (256 * 1024 * 1024) // DTYPE_NBYTES" in run_source
    assert "BIG_COUNT = URMA_SINGLE_WQE_FLOATS + 1" in run_source
    assert "pytest.mark" not in run_source
    assert "rt_submit_aiv_task(0" in orch
    assert kernel.count("send_request_entry") == 1
    assert "first_chunk_count" not in kernel
    assert "second_chunk_count" not in kernel
    assert "UrmaTget" in kernel
    assert "UrmaTput" not in kernel


def test_phase4_real_deferred_consumer_depends_on_deferred_outputs() -> None:
    consumer = DEFERRED_CONSUMER.read_text()
    orch = DEFERRED_ORCH.read_text()
    test_py = DEFERRED_TEST_PY.read_text()

    assert "consumer_args.add_input(tget0_tmp)" in orch
    assert "consumer_args.add_input(tget1_tmp)" in orch
    assert "consumer_args.add_input(tget0_marker)" in orch
    assert "consumer_args.add_input(tget1_marker)" in orch
    assert "consumer_args.add_input(tput0_marker)" in orch
    assert "consumer_args.add_input(tput1_marker)" in orch
    assert "Status::kTgetMismatch" in consumer
    assert "Status::kTputMismatch" in consumer
    assert "TGET_ASYNC<pto::comm::DmaEngine::URMA>" in consumer
    assert "Status::kTputReadbackFailed" in consumer
    assert "comm_ctx->rankNum) * elem_count" in consumer
    assert "status[3] = tget_marker_sum" in consumer
    assert "status[4] = tput0_marker[0] + tput1_marker[0]" in consumer
    assert "status[5] = marker_sum" in consumer
    assert "status[6] = expected_marker_sum" in consumer
    assert "status[7] = static_cast<int32_t>(other_slot[0])" in consumer
    assert "CASES = (1, 16, 64, 256, 4096, 16384)" in test_py
    assert "def _run_case_on_worker" in test_py
    assert "chip_callable = build_chip_callable(platform)" in test_py
    assert "worker = Worker(" in test_py
    assert "worker.init()" in test_py
    assert "for elem_count in CASES:" in test_py
    assert "_run_case_on_worker(worker, chip_handle, elem_count, len(device_ids))" in test_py
    assert "def _run_iteration_on_worker" not in test_py
    assert 'parser.add_argument("--repeat"' not in test_py
    assert "tput_elems = (nranks + 1) * elem_count" in test_py
    assert "@pytest.mark.requires_hardware" in test_py
    assert "pytest.mark.platforms([\"a5\"])" in test_py
    assert "platform != \"a5\"" in test_py
    assert "ArgDirection.INOUT, ArgDirection.INOUT, ArgDirection.OUT, ArgDirection.IN, ArgDirection.IN" in test_py


def test_phase4_urma_backend_design_tracks_current_contract() -> None:
    source = URMA_BACKEND_DESIGN.read_text()
    status = STATUS_DOC.read_text()

    assert "PTO-ISA URMA async operations" in source
    assert "backend_cookie = workspace" in source
    assert "COMPLETION_TYPE_URMA_EVENT_HANDLE" in source
    assert "poll_urma_event_handle(event_handle, workspace)" in source
    assert "UrmaWorkspaceManager" in source
    assert "Transfer splitting" in source
    assert "Event coalescing" in source
    assert "Address helper" in source
    assert "peer_mr_base_addr" in source
    assert "peer_mr_ptr" in source
    assert "peer_mr_base_addr" in status
    assert "peer_mr_ptr" in status
