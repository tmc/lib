# Test Inventory: gopccl

## Cheap Gates

- Plan schema validation.
- Generated artifact golden tests.
- Public API snapshot from simulated go doc output.
- ABI layout tests that do not require hardware.
- Fake backend `fake`: recorded-call assertions.
- Fake rendezvous store `Store`: in-memory key/value rendezvous.
- Provider metadata fields are part of generated capability snapshots.
- Health/readiness fields are part of generated capability snapshots.
- Fake transport state machine `deterministic-drop-retry` with deterministic drop/retry checks.
- Generated topology/state helper tests cover PCCL sync strategy and tensor ABI marshaling without running the provider.
- Backend adapter registry tests run without loading real providers.
- Build-tagged `pccl` adapter helper tests cover coordinator parsing, config, buffer, dtype, reduce-op, topology/state mapping, and capability mapping without hardware.
- Build-tagged `nccl` adapter helper tests cover config, store, buffer, dtype, reduce-op, stream mapping, and unsupported topology methods without hardware.
- Build-tagged `rccl` adapter helper tests cover config, store, buffer, dtype, reduce-op, stream mapping, and unsupported topology methods without hardware.

## Manual Gates

- `hardware`: set GOPCCL_TEST_HARDWARE=1 to run hardware/provider integration tests
- `multirank`: set GOPCCL_TEST_MULTIRANK=1 with an external accept-pump harness to run multi-rank datapath acceptance tests
