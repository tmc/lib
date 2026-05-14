# Test Inventory: gojaccl

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

## Manual Gates

- `rdma`: set JACCL_TEST_RDMA=1 to run hardware/provider integration tests
- `rdma_rtr`: set JACCL_TEST_RDMA_ALLOW_RTR=1 to run hardware/provider integration tests
