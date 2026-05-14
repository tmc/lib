# Manual Gates: gorccl

## hardware

Runs real provider or datapath checks. This gate is never run by default.

- Environment: `GORCCL_TEST_HARDWARE=1`
- Build tag: `gorccl_hardware`
- Command: `GORCCL_TEST_HARDWARE=1 RCCL_LIB_PATH=/path/to/librccl.so go test -tags=gorccl_hardware -run '^TestIntegration' -count=1`
- Generated helper: `RequireHardwareGate(t *testing.T)`
- Harness hook: set `RunHardwareGate` from a handwritten external test before enabling the gate.
- Skip reason: set GORCCL_TEST_HARDWARE=1 to run hardware/provider integration tests
