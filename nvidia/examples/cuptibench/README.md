# CUPTI metrics in Go benchmarks

This example runs a small single-token inference pipeline:

1. RMS normalization of a 1024-element float32 activation vector.
2. A dense 1024→256 projection using 1 MiB of float32 weights.
3. SiLU activation of the 256 outputs.

The inputs and weights are deterministic. Every executed stage, including an
isolated stage's prepared prefix, is checked against a CPU reference using
double-precision accumulation and an explicit absolute/relative tolerance.
These are readable reference kernels, not tuned kernels or a complete DS4 layer.
There is no model download, attention, KV cache, or quantization.

It demonstrates:

- `TestMain` registration and profile publication after successful cleanup.
- One `full` subbenchmark with named `rmsnorm`, `project`, and `silu` regions.
- Independently selected isolated stages. Isolated `silu` prepares its
  normalization and projection prefix outside timing and counter collection.
- Bounded replay, explicit stream completion, and output validation.

Read [benchmark_test.go](benchmark_test.go) for the lifecycle. The kernels are
embedded from [kernels.cu](kernels.cu) and compiled with NVRTC.

## Run

From the `nvidia` module directory, on an exclusively owned CUDA device:

```sh
CUPTIBENCH_EXAMPLE=1 GOWORK=off go test ./examples/cuptibench \
  -run '^$' -bench '^BenchmarkPipeline$' -benchtime=1x -count=1
```

Without `CUPTIBENCH_EXAMPLE=1`, benchmarks skip. With it set, unavailable CUDA,
CUPTI, NVRTC, or metrics fail. The NVRTC binding currently loads
`libnvrtc.so.13` on Linux. The NVRTC compiler must emit PTX supported by your
driver. On the qualified DGX setup, use:

```sh
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/targets/sbsa-linux/lib
```

Select only the final stage, with no sibling benchmark prerequisite:

```sh
CUPTIBENCH_EXAMPLE=1 GOWORK=off go test ./examples/cuptibench \
  -run '^$' -bench '^BenchmarkPipeline$/^isolated$/^silu$' -benchtime=1x -count=1
```

Create a new output directory and export a pprof profile plus a JSON manifest:

```sh
mkdir -p "$HOME/tmp"
profile_dir=$(mktemp -d "$HOME/tmp/cuptibench-example.XXXXXX")
CUPTIBENCH_EXAMPLE=1 GOWORK=off go test ./examples/cuptibench \
  -run '^$' -bench '^BenchmarkPipeline$' -benchtime=1x -count=1 \
  -cuptiprofile="$profile_dir/pipeline.pprof"
go tool pprof -top -tagfocus="view=full" "$profile_dir/pipeline.pprof"
```

The default `launches` metric set counts **thread blocks**, not kernel calls:
full is 258 CTAs: normalization 1, projection 256, and activation 1.
`profile-ops` is 1. `ns/op` measures
host submission and completion; `gpu-ns/op` uses CUDA events. Counters come from
separate replay executions, not the timed iteration. A `1x` run demonstrates
operation and export; it does not establish a stable performance estimate.

The full benchmark is the aggregate measurement. Go does not sum isolated
subbenchmark times into a parent result, and isolated runs may have different
cache state. The profile includes full and isolated experiments with distinct
capture/view labels: filter to one experiment before interpreting its total.
Regions export terminal regions; inclusive operation counters remain in JSON.

`-cuptimetrics=instructions` selects another named metric set. Unsupported
metrics fail rather than silently substituting another counter. Existing
profile destinations are never overwritten.

For CUDA/PTX/SASS source capture and disassembly, see
[ptx-sass-trace](../ptx-sass-trace/README.md).

Compilation, weight upload, and one warmup sequence happen once per leaf
benchmark, outside timing.
Preparation clears scratch buffers before each sequence; it does not flush the
GPU cache. The 1 MiB weight matrix may remain cached, so this example is not a
DRAM-bandwidth benchmark. Compare full and isolated views as separate experiments.
