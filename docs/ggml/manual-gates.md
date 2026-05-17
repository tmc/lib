# Manual Gates: ggml

## cuda

Runs ggml CUDA provider tests on a host with CUDA-capable hardware.

- Environment: `GGML_TEST_HARDWARE=1`
- Build tag: `ggml_cuda`
- Command: `GGML_TEST_HARDWARE=1 go test -tags=ggml_cuda -run '^TestIntegration' -count=1 ./ggml`
- Generated helper: `RequireCudaGate(t *testing.T)`
- Harness hook: set `RunCudaGate` from a handwritten external test before enabling the gate.
- Skip reason: set GGML_TEST_HARDWARE=1 and build with -tags=ggml_cuda to run ggml CUDA integration tests

## metal

Runs ggml Metal provider tests on Apple GPU hardware.

- Environment: `GGML_TEST_HARDWARE=1`
- Build tag: `ggml_metal`
- Command: `GGML_TEST_HARDWARE=1 go test -tags=ggml_metal -run '^TestIntegration' -count=1 ./ggml`
- Generated helper: `RequireMetalGate(t *testing.T)`
- Harness hook: set `RunMetalGate` from a handwritten external test before enabling the gate.
- Skip reason: set GGML_TEST_HARDWARE=1 and build with -tags=ggml_metal to run ggml Metal integration tests
