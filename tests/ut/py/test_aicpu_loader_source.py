from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
HOST_LOADER = REPO_ROOT / "src/common/aicpu_loader/host/load_aicpu_op.cpp"
DEVICE_DISPATCHER = REPO_ROOT / "src/common/aicpu_loader/device/aicpu_dispatcher.cpp"


def test_bootstrap_dispatcher_symbol_matches_exported_init() -> None:
    host_source = HOST_LOADER.read_text()
    dispatcher_source = DEVICE_DISPATCHER.read_text()

    assert '"DynTileFwkBackendKernelServerInit"' in host_source
    assert "DynTileFwkBackendKernelServerInit(void *args)" in dispatcher_source
    assert '"DynTileFwkKernelServerInit"' not in host_source
