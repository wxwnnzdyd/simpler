# A5 URMA Deferred Completion Status

## Phase 1: Sim Deferred Completion

Status: complete.

Validated:

- `a5sim` URMA deferred completion demo builds and runs.
- Fake TGET and TPUT paths both publish a pending CQE before scheduler polling.
- Scheduler observes pending before ready.
- Dependent consumer runs only after the fake URMA completion becomes ready.

Scope note:

- Phase 1 uses fake copy and fake CQE production. It does not validate HCCL
  channels, memory registration, huge pages, doorbells, or real URMA WQ/CQ
  hardware behavior.

## Phase 2: Real A5 Build And Workspace

Status: code complete; local build validation passed; hardware smoke blocked by
the current A5 account environment.

Validated:

- `pto_isa.pin` points at a PTO-ISA revision that contains URMA async support
  and `UrmaWorkspaceManager`.
- A5 onboard `tensormap_and_ringbuffer` runtime builds with CANN 9.1.
- A5 onboard host runtime can include PTO-ISA URMA host headers.
- A5 host link includes the HCOMM dependencies required by the URMA workspace
  manager.
- The selected `libhcomm.so` exposes:
  - `HcclCommMemReg`
  - `HcclChannelGetRemoteMems`
  - `HcclRankGraphGetLinks`
  - `HcclChannelAcquire`
- `comm_alloc_windows` initializes a base URMA workspace and mirrors
  `GetWorkspaceAddr()` into `CommContext::workSpace`.
- `CommContext::workSpaceSize` is populated with the expected workspace table
  size.
- Dense prefix derived contexts reuse the base URMA workspace.
- Non-dense derived contexts leave the URMA workspace disabled instead of
  reusing a mismatched base workspace.
- Dense dynamic-domain allocations initialize an independent workspace for the
  new symmetric buffer.
- Documentation records that the A5 URMA symmetric window currently keeps the
  HCCL Path-D allocation policy, `ACL_MEM_MALLOC_HUGE_FIRST`, and explains when
  to tighten it to `ACL_MEM_MALLOC_HUGE_ONLY`.

Not yet validated:

- The Phase 2 hardware smoke
  `tests/ut/py/test_worker/test_platform_comm.py --platform a5 --device 0-1`
  did not run to completion in the current account.
- The blocker is environment access, not a workspace assertion failure:
  `npu-smi` cannot load driver runtime libraries because
  `/usr/local/Ascend/driver/lib64/common/libsecurec.so` is readable only by
  `HwHiAiUser`.
- Because `npu-smi` cannot run, the A5 arch precheck cannot complete, and
  `aclrtSetDevice` reports `507899`.

Acceptance decision:

- Treat Phase 2 as complete for development planning so Phase 3 can proceed.
- Before declaring Phase 2 fully hardware-accepted, rerun the hardware smoke on
  an A5 environment where the test user can run `npu-smi info` successfully.

Required hardware acceptance command:

```bash
PYTHONPATH=$PWD:$PWD/python python -m pytest \
  tests/ut/py/test_worker/test_platform_comm.py \
  --platform a5 --device 0-1 -q
```

Expected result:

```text
1 passed
```

## Phase 3: Standalone Real URMA Correctness

Status: demo code added; local A5 compile validation passed; real hardware run
still pending.

Goal:

- Prove real PTO-ISA URMA submit and data movement correctness on A5, without
  connecting the operation to AICPU deferred completion.

Implemented:

- Added `examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/`.
- The demo allocates all URMA source, destination, and signal buffers inside a
  two-rank dynamic communication domain.
- Host staging uses `orch.copy_to` to initialize each rank's registered
  symmetric window before submitting the AIV kernel.
- The AIV kernel reads `CommContext::workSpace`, builds a PTO-ISA URMA
  `AsyncSession`, issues both `TGET_ASYNC<DmaEngine::URMA>` and
  `TPUT_ASYNC<DmaEngine::URMA>`, and waits directly with `event.Wait(session)`.
- URMA remote addresses are derived from `UrmaPeerMrBaseAddr(workspace, peer) +
  offset`.
- A notification barrier runs after the outgoing TPUT wait so each rank verifies
  its incoming TPUT only after the peer has completed the write.
- The demo covers a small case (`16` float32 elements) and a page-spanning case
  (`4096` float32 elements).

Validated:

- `test_urma_real_async_demo.py` passes Python syntax compilation.
- The new A5 incore kernel and orchestration shim compile with the local CANN
  9.0.1 toolchain after the local sparse PTO-ISA checkout is populated with A5
  instruction headers.
- Pytest collection includes the demo for `--platform a5`.
- Pytest collection deselects the demo for `--platform a5sim`, so sim batches do
  not accidentally run a real-URMA-only test.

Not yet validated:

- The demo has not been executed on real A5 hardware in the current account.
- Required hardware acceptance is blocked until the test user can run
  `npu-smi info` and `aclrtSetDevice` successfully.
- Device logs have not yet been inspected for URMA CQE status/substatus errors.

Required hardware acceptance command:

```bash
PYTHONPATH=$PWD:$PWD/python python \
  examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/test_urma_real_async_demo.py \
  -p a5 -d 0-1
```

Expected result:

```text
[urma_real_async_demo] count=16 rank=0 status=[0, 16, 1, ...]
[urma_real_async_demo] count=16 rank=1 status=[0, 16, 0, ...]
[urma_real_async_demo] count=4096 rank=0 status=[0, 4096, 1, ...]
[urma_real_async_demo] count=4096 rank=1 status=[0, 4096, 0, ...]
```
