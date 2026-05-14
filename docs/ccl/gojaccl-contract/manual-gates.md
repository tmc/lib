# Manual Gates: gojaccl

## rdma

Runs real provider or datapath checks. This gate is never run by default.

- Environment: `JACCL_TEST_RDMA=1`
- Build tag: `gojaccl_rdma`
- Command: `JACCL_TEST_RDMA=1 go test -tags=gojaccl_rdma -run '^TestIntegration' -count=1`
- Generated helper: `RequireRdmaGate(t *testing.T)`
- Harness hook: set `RunRdmaGate` from a handwritten external test before enabling the gate.
- Skip reason: set JACCL_TEST_RDMA=1 to run hardware/provider integration tests

## rdma_rtr

Runs real provider or datapath checks. This gate is never run by default.

- Environment: `JACCL_TEST_RDMA_ALLOW_RTR=1`
- Command: `JACCL_TEST_RDMA=1 JACCL_TEST_RDMA_ALLOW_RTR=1 go test -run '^TestIntegration.*RTR' -count=1`
- Generated helper: `RequireRdmaRtrGate(t *testing.T)`
- Harness hook: set `RunRdmaRtrGate` from a handwritten external test before enabling the gate.
- Skip reason: set JACCL_TEST_RDMA_ALLOW_RTR=1 to run hardware/provider integration tests
