# Test Inventory: gonccl

## Cheap Gates

- Plan schema validation.
- Generated artifact golden tests.
- Public API snapshot from simulated go doc output.
- ABI layout tests that do not require hardware.
- Fake backend `fake`: recorded-call assertions.
- Fake rendezvous store `Store`: in-memory key/value rendezvous.
- Provider metadata fields are part of generated capability snapshots.
- Health/readiness fields are part of generated capability snapshots.
- Backend adapter registry tests run without loading real providers.
- Build-tagged `nccl` adapter helper tests cover config, store, buffer, dtype, reduce-op, and stream mapping without hardware.

## Manual Gates

- `hardware`: set GONCCL_TEST_HARDWARE=1 to run hardware/provider integration tests
