from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RDMA_BACKEND_KERNEL = (
    REPO_ROOT / "src/a5/runtime/tensormap_and_ringbuffer/runtime/backend/rdma/rdma_completion_kernel.h"
)
ASYNC_WAIT = REPO_ROOT / "src/a5/runtime/tensormap_and_ringbuffer/runtime/pto_async_wait.h"
MAILBOX_TYPES = REPO_ROOT / "src/a5/runtime/tensormap_and_ringbuffer/runtime/aicore_completion_mailbox_types.h"
HOST_CMAKE = REPO_ROOT / "src/a5/platform/onboard/host/CMakeLists.txt"
HOST_COMM = REPO_ROOT / "src/a5/platform/onboard/host/comm_hccl.cpp"
KERNEL_COMPILER = REPO_ROOT / "simpler_setup/kernel_compiler.py"
RDMA_DEMO = REPO_ROOT / "examples/a5/tensormap_and_ringbuffer/rdma_deferred_completion_demo"
RDMA_DEMO_COMMON = RDMA_DEMO / "kernels/aiv/rdma_deferred_completion_common.h"
RDMA_DEMO_TGET = RDMA_DEMO / "kernels/aiv/kernel_rdma_deferred_completion_tget.cpp"
RDMA_DEMO_TPUT = RDMA_DEMO / "kernels/aiv/kernel_rdma_deferred_completion_tput.cpp"
RDMA_DEMO_CONSUMER = RDMA_DEMO / "kernels/aiv/kernel_rdma_deferred_completion_consumer.cpp"
RDMA_DEMO_ORCH = RDMA_DEMO / "kernels/orchestration/rdma_deferred_completion_orch.cpp"
RDMA_DEMO_TEST = RDMA_DEMO / "test_rdma_deferred_completion_demo.py"
RDMA_SCHEDULER = (
    REPO_ROOT / "src/a5/runtime/tensormap_and_ringbuffer/runtime/backend/rdma/rdma_completion_scheduler.h"
)


def test_rdma_kernel_backend_exposes_deferred_adapter_contract() -> None:
    backend = RDMA_BACKEND_KERNEL.read_text()

    assert "enum class RdmaOp" in backend
    assert "struct RdmaRequestDescriptor" in backend
    assert "RdmaTget" in backend
    assert "RdmaTput" in backend
    assert "DmaEngine::RDMA" in backend
    assert "TGET_ASYNC<pto::comm::DmaEngine::RDMA>" in backend
    assert "TPUT_ASYNC<pto::comm::DmaEngine::RDMA>" in backend
    assert "register_rdma_async_event" in backend
    assert "COMPLETION_ENGINE_ROCE" in backend
    assert "COMPLETION_TYPE_RDMA_EVENT_HANDLE" in backend
    assert "backend_cookie" not in backend or "reinterpret_cast<uint64_t>(workspace)" in backend


def test_rdma_kernel_backend_uses_peer_mr_base_not_windows_in() -> None:
    backend = RDMA_BACKEND_KERNEL.read_text()

    assert "peer_mr_base_addr" in backend
    assert "peer_mr_ptr" in backend
    assert "PeerMrBaseAddr(workspace, peer)" in backend
    assert "windowsIn" not in backend


def test_rdma_completion_type_is_registered_in_async_wait() -> None:
    wait_source = ASYNC_WAIT.read_text()
    type_source = MAILBOX_TYPES.read_text()

    assert "#define COMPLETION_TYPE_RDMA_EVENT_HANDLE 3" in type_source
    assert 'backend/rdma/rdma_completion_scheduler.h' in wait_source
    assert "rdma_event_handle_poll_op" in wait_source
    assert "rdma_event_handle_retire_op" in wait_source
    assert "COMPLETION_TYPE_RDMA_EVENT_HANDLE = 3" in wait_source


def test_a5_host_cmake_gates_rdma_workspace_overlay() -> None:
    cmake = HOST_CMAKE.read_text()

    assert "option(SIMPLER_ENABLE_PTO_RDMA_WORKSPACE" in cmake
    assert "SIMPLER_ENABLE_PTO_URMA_WORKSPACE_DEFAULT OFF" in cmake
    assert "SIMPLER_ENABLE_PTO_URMA_WORKSPACE_DEFAULT ON" in cmake
    assert "Only one PTO async workspace overlay may be enabled" in cmake
    assert "PTO RDMA workspace overlay requires pto-isa RDMA headers" in cmake
    assert "PTO_RDMA_SUPPORTED" in cmake
    assert "PTO_RDMA_BACKEND_HNS_1825_SUPPORTED" in cmake


def test_a5_comm_hccl_keeps_urma_and_rdma_workspace_paths_macro_gated() -> None:
    source = HOST_COMM.read_text()

    assert "#ifdef SIMPLER_ENABLE_PTO_URMA_WORKSPACE" in source
    assert "#ifdef SIMPLER_ENABLE_PTO_RDMA_WORKSPACE" in source
    assert "#include \"pto/comm/async/rdma/rdma_workspace_manager.hpp\"" in source
    assert "#include \"pto/comm/async/rdma/backends/hns_1825/hns_1825_bootstrap.hpp\"" in source
    assert "std::unique_ptr<pto::comm::rdma::RdmaWorkspaceManager> rdma_workspace" in source


def test_a5_comm_hccl_uses_mr1374_rdma_host_api_shape() -> None:
    source = HOST_COMM.read_text()

    assert "#define __gm__" in source
    assert "ResolvePhyId(out.phy_id)" in source
    assert "ResolveLocalRdmaIp(" in source
    assert "PTO_ROCE_ROOTINFO" in source
    assert "ResolveLocalRdmaIp(out.phy_id, out.local_ip, rootinfo_path)" in source
    assert "rdma_resolve_local_ip_from_hccn_tool(device_id, out.local_ip)" in source
    assert "rdma_resolve_local_ip_from_rootinfo(out.phy_id, out.local_ip, rootinfo_path)" in source
    assert "hccn_tool -g -dev_info -i %d" in source
    assert "\\\"addr\\\"" in source
    assert "manager->Init(config)" in source
    assert "WorkspaceInitResult::READY" in source
    assert "RdmaBackend::HNS_1825" not in source


def test_kernel_compiler_forwards_rdma_workspace_macros_to_incore_builds() -> None:
    source = KERNEL_COMPILER.read_text()

    assert '"SIMPLER_ENABLE_PTO_RDMA_WORKSPACE"' in source
    assert '"-DPTO_RDMA_SUPPORTED"' in source
    assert '"-DPTO_RDMA_BACKEND_HNS_1825_SUPPORTED"' in source
    assert "cmd += self._incore_feature_defines()" in source


def test_kernel_compiler_adds_ascend_device_headers_for_rdma_backend_headers() -> None:
    source = KERNEL_COMPILER.read_text()

    assert "get_ascend_incore_include_dirs" in source
    assert 'root / "asc" / "include" / "interface"' in source
    assert 'root / "asc" / "include" / "basic_api" / "interface"' in source
    assert 'root / "ascendc" / "include" / "basic_api" / "interface"' in source
    assert '"kernel_operator_sys_var_intf.h"' in source
    assert '"kernel_operator_sys_var_intf_impl.h"' in source
    assert ".rglob(header)" in source
    assert "for inc_dir in self.get_ascend_incore_include_dirs():" in source


def test_rdma_deferred_completion_demo_uses_rdma_adapter_and_remote_mr_base() -> None:
    common = RDMA_DEMO_COMMON.read_text()
    tget = RDMA_DEMO_TGET.read_text()
    tput = RDMA_DEMO_TPUT.read_text()
    consumer = RDMA_DEMO_CONSUMER.read_text()
    orch = RDMA_DEMO_ORCH.read_text()
    test_py = RDMA_DEMO_TEST.read_text()

    assert "backend/rdma/rdma_completion_kernel.h" in common
    assert "PTO_RDMA_SUPPORTED" in common
    assert "pto2::rdma_backend::peer_mr_base_addr" in common
    assert "comm_ctx->windowsIn[peer]" not in common
    assert "RdmaScratchTile" in common
    assert "pto::comm::sdma::UB_ALIGN_SIZE" in common
    assert "rdma_scratch_tile" in common
    assert "RdmaTget(" in tget
    assert "RdmaTput(" in tput
    assert "rdma_scratch" in tget
    assert "rdma_scratch" in tput
    assert "comm_ctx->rankId" in tget
    assert "comm_ctx->rankId" in tput
    assert tget.count("send_request_entry") >= 2
    assert tput.count("send_request_entry") >= 2
    assert "event.Wait" not in tget
    assert "event.Wait" not in tput
    assert "BuildAsyncSession<pto::comm::DmaEngine::RDMA>" in consumer
    assert "rdma_scratch" in consumer
    assert "readback_session, 2" in consumer
    assert "TensorCreateInfo tget_output_info" not in orch
    assert "Tensor tget0_recv = tget_recv.view" in orch
    assert "Tensor tget1_recv = tget_recv.view" in orch
    assert "tget0_args.add_output(tget0_recv)" in orch
    assert "tget1_args.add_output(tget1_recv)" in orch
    assert "Tensor tget0_tmp = tget0_recv" in orch
    assert "Tensor tget1_tmp = tget1_recv" in orch
    assert "Tensor tget0_marker = tget0_outputs.get_ref(0)" in orch
    assert "Tensor tget1_marker = tget1_outputs.get_ref(0)" in orch
    assert "tget0_outputs.get_ref(1)" not in orch
    assert "tget1_outputs.get_ref(1)" not in orch
    assert "rt_submit_aiv_task(0" in orch
    assert "rt_submit_aiv_task(1" in orch
    assert "rt_submit_aiv_task(2" in orch
    assert "tget_elems = nranks * elem_count" in test_py
    assert "CASES = (16, 64, 256, 4096, 16384)" in test_py
    assert "CASES = (1," not in test_py
    assert 'shapes=(tget_elems,)' in test_py
    assert "rdma_workspace_enabled" in test_py
    assert "SIMPLER_ENABLE_PTO_RDMA_WORKSPACE" in test_py
    assert "pytest.skip" in test_py


def test_rdma_scheduler_abi_matches_mr1374_workspace_and_hns1825_contexts() -> None:
    source = RDMA_SCHEDULER.read_text()

    assert "uint32_t rank_count;" in source
    assert "uint32_t reserved;" in source
    assert "uint32_t local_token_id" not in source
    assert "static_assert(sizeof(RdmaWqCtx) == 96" in source
    assert "static_assert(sizeof(RdmaCqCtx) == 64" in source
    assert "struct Hns1825Cqe" in source
    assert "static_assert(sizeof(Hns1825Cqe) == 32" in source
    assert "uint32_t cqe_size;" in source
    assert "1u << cq_ctx.cqe" not in source
    assert "db_sw_addr" in source
    assert "kHns1825CqeMaxGenNum" in source
    assert "owner_id_qpn" in source
    assert "op_sr_wqebb" in source
