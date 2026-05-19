# Test Inventory: virtiofs

## Cheap Gates

- Plan schema validation.
- Generated file surface assertions.
- Config validation tests.
- Device round-trip tests over a MemoryTransport and fuse.MemoryFS.
- Response error decoding tests.
- Server request handling tests.
- Fake transport close behavior.
- Manual VM gate no-env skip and env-enabled fail-closed behavior.

## Manual Gates

- `vm`: set VIRTIOFS_TEST_VM=1 and provide a VM-backed RunVMGate hook.
