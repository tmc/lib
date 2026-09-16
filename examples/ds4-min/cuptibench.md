# CUDA kernel benchmarks

`BenchmarkCUDAKernels` uses cuptibench to measure isolated synthetic projection
kernels. No model is loaded. Each leaf can be selected independently:

- `decode/q80/{q8_1,quantized,blockdiag}/{tensor}/tokens=1`
- `prefill/q80/{q8_1,quantized,blockdiag}/{tensor}/tokens=8`
- `{decode,prefill}/f16_pair/rows={256,512,1024}/tokens={1,8}`

The Q8_0 shapes come from `cudaQ80BenchShapes`, including its documented shape
provenance limits. The output head is decode-only: prefill does not project
every input token to logits. The eight-token cases exercise the batched kernel
entry points; they do not run a complete prefill. Attention, routing, expert
kernels, quantization, KV updates, and model throughput are not covered here.

Weights and activations are deterministic, device-resident fixtures. Setup
launches scalar kernels to establish a reference, warms the measured kernel,
and checks finite, bit-identical output. Output is poisoned before each timed
sequence and profiling pass, then checked again after collection. The decode
reference is a repeatability check, not an independent correctness oracle;
use the existing CUDA qualification tests for numerical qualification.

## Running

Build from this directory on the Mac control plane:

```sh
GOOS=linux GOARCH=arm64 GOWORK=off go test -mod=readonly -c -o "$HOME/tmp/ds4-cuptibench.test" .
```

Copy the binary to an authorized Linux/ARM64 GPU host. Coordinate with the
current GPU owner first; do not overlap the Pi provider's model or another
GPU workload. Run GPU commands remotely through `ssh dgx` under the host's
`gpu-lock` wrapper. The benchmark requires `-ds4-cuptibench` and a nonempty
`GPU_LOCK_HELD`; that variable is a guard, not proof of exclusive ownership.
Do not set it manually to bypass the wrapper.

Illustrative argument fragment for the locked binary invocation (not yet
GPU-qualified; supply the installed binary and lock wrapper on the remote host):

```text
-test.run '^$' -test.bench '^BenchmarkCUDAKernels$/^prefill$/^f16_pair$/^rows=256$/^tokens=8$' -test.benchtime=1x -ds4-cuptibench
```

Use `-test.bench '^BenchmarkCUDAKernels$'` to select all leaves. Start with one
leaf and one iteration before longer runs. CUDA/NVRTC libraries and CUPTI
profiling permissions must be available on the target host. Failure to create
the CUPTI plan is a benchmark failure, not a timing-only fallback.

Optional cuptibench flags are `-cuptimetrics=launches` (the default metric set)
and `-cuptiprofile=<fresh-path>.pb.gz`. The latter writes a pprof profile and
companion JSON manifest. Destinations must not already exist. A failed test,
benchmark, output check, or plan cleanup prevents successful export.

## Reading results

One operation is one kernel launch, processing the leaf's `tokens/op` inputs.
`ns/op` includes host submission and completion; `gpu-ns/op` is separately
measured CUDA-event time. Native CUPTI metrics come from separate replayed
executions, not the event-timed execution. The plan uses `Kernels` mode with
one operation per profiling sequence, bounded capture capacity, and
`CachePreserve`. This requests no cache flush; it does not promise cold memory
or a fixed cache state. Warm device-resident fixtures differ from model weights
read through host memory.

Do not sum leaves into a model latency or interpret `launches` as a host API
call count: its native counter is `sm__ctas_launched.sum`. See the
[cuptibench documentation](../../nvidia/x/cuptibench/README.md) for metric,
replay, export, and hardware-admission contracts.

## Validation status

Linux/ARM64 cross-compilation and vet passed. CPU-only execution on DGX passed
the float-bit comparator controls and default benchmark skip; explicit opt-in
without the lock marker failed before device creation, as intended. No GPU
timings, counter captures, or profile round trips have been collected for this
benchmark. The standalone macOS package test remains blocked by the existing
module-tidiness requirement under `-mod=readonly`; module files were not changed.
