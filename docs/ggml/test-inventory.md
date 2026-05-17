# Test Inventory: ggml

## Cheap Gates

- Plan schema validation.
- Generated artifact golden tests.
- Public API snapshot from simulated go doc output.
- ABI layout tests that do not require hardware.
- Provider metadata fields are part of generated capability snapshots.
- Health/readiness fields are part of generated capability snapshots.
- Backend adapter registry tests run without loading real providers.

## Manual Gates

- `cuda`: set GGML_TEST_HARDWARE=1 and build with -tags=ggml_cuda to run ggml CUDA integration tests
- `metal`: set GGML_TEST_HARDWARE=1 and build with -tags=ggml_metal to run ggml Metal integration tests
