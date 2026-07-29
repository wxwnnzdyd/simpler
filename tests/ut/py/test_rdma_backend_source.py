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
    assert "h->rootinfo_path.c_str()" in source
    assert "manager->Init(config)" in source
    assert "WorkspaceInitResult::READY" in source
    assert "RdmaBackend::HNS_1825" not in source
