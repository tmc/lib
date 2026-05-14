# Manual Gates: gopccl

## hardware

Runs real provider or datapath checks. This gate is never run by default.

- Environment: `GOPCCL_TEST_HARDWARE=1`
- Build tag: `gopccl_pccl`
- Command: `GOPCCL_TEST_HARDWARE=1 go test -tags=gopccl_pccl -run '^TestIntegration' -count=1`
- Generated helper: `RequireHardwareGate(t *testing.T)`
- Harness hook: set `RunHardwareGate` from a handwritten external test before enabling the gate.
- Skip reason: set GOPCCL_TEST_HARDWARE=1 to run hardware/provider integration tests

## multirank

Multi-rank datapath acceptance test. Verifies PeersPending, UpdateTopology, OptimizeTopology, and SynchronizeState sequence under simulated load. Expects an external accept-pump harness to prevent FSDP-style deadlocks.

- Environment: `GOPCCL_TEST_MULTIRANK=1`
- Build tag: `gopccl_pccl`
- Command: `GOPCCL_TEST_MULTIRANK=1 GOPCCL_TEST_HARDWARE=1 go test -tags=gopccl_pccl -run '^TestIntegrationMultirankGate$' -count=1`
- Generated helper: `RequireMultirankGate(t *testing.T)`
- Harness hook: set `RunMultirankGate` from a handwritten external test before enabling the gate.
- Skip reason: set GOPCCL_TEST_MULTIRANK=1 with an external accept-pump harness to run multi-rank datapath acceptance tests
