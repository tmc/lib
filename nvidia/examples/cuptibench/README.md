# CUPTI metrics in Go benchmarks

This example runs a two-kernel pipeline: `mid = input + 3`, then
`output = 2 * mid`. It demonstrates:

- `TestMain` registration and profile publication after successful cleanup.
- One `full` subbenchmark with named `project` and `scale` regions.
- Independently selected isolated stages. Isolated `scale` prepares its
  `project` prefix outside timing and counter collection.
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

Select only the second stage, with no sibling benchmark prerequisite:

```sh
CUPTIBENCH_EXAMPLE=1 GOWORK=off go test ./examples/cuptibench \
  -run '^$' -bench '^BenchmarkPipeline$/^isolated$/^scale$' -benchtime=1x -count=1
```

Create a new output directory and export a pprof profile plus a JSON manifest:

```sh
mkdir -p "$HOME/tmp"
profile_dir=$(mktemp -d "$HOME/tmp/cuptibench-example.XXXXXX")
CUPTIBENCH_EXAMPLE=1 GOWORK=off go test ./examples/cuptibench \
  -run '^$' -bench '^BenchmarkPipeline$' -benchtime=1x -count=1 \
  -cuptiprofile="$profile_dir/pipeline.pprof"
go tool pprof -top "$profile_dir/pipeline.pprof"
```

The default `launches` metric set counts **thread blocks**, not kernel calls:
full is 8 CTAs and each isolated stage is 4. `profile-ops` is 1. `ns/op` measures
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
