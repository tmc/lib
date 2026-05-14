# Manual Gates: gonccl

## hardware

Runs real provider or datapath checks. This gate is never run by default.

- Environment: `GONCCL_TEST_HARDWARE=1`
- Build tag: `gonccl_hardware`
- Command: `GONCCL_TEST_HARDWARE=1 NCCL_LIB_PATH=/path/to/libnccl.so go test -tags=gonccl_hardware -run '^TestIntegration' -count=1`
- Generated helper: `RequireHardwareGate(t *testing.T)`
- Harness hook: set `RunHardwareGate` from a handwritten external test before enabling the gate.
- Skip reason: set GONCCL_TEST_HARDWARE=1 to run hardware/provider integration tests
