# cudadoctor

Check the tools and shared libraries needed by one CUDA workflow:

```
go run ./x/cmd/cudadoctor -workflow=jit
go run ./x/cmd/cudadoctor -workflow=profile -json
```

| Workflow | Required checks |
| --- | --- |
| `jit` (default) | Load NVRTC and query its version |
| `aot` | Run `nvcc --version` and `ptxas --version` |
| `profile` | Load driver and CUPTI, query CUPTI version, run `nvdisasm --version` |
| `sanitize` | Load driver, run `compute-sanitizer --version` |
| `tileir` | Load NVRTC, query version, check TileIR extraction symbols |

These are prerequisite checks. Profile checks do not establish permissions,
supported counters or capture completeness. TileIR checks do not establish Tile
C++ header availability, successful compilation or device support. AOT checks do
not compile a translation unit. No default check creates a CUDA context.

The report includes the Go version used to build cudadoctor, its module build
information, relevant environment overrides, tool search candidates and library
loader paths. A library path that is only a soname is labeled as a loader request;
it is not claimed to be an absolute resolved filename. Tool symlinks are resolved.
Tool discovery uses the same `cudatool` resolver as the other CUDA tools: toolkit
candidates precede PATH. Library discovery uses the bindings' own loaders.
`CUDA_TOOLKIT_PATH` is recorded but is not a `cudatool` search override.

Profiling reports also check cuda-gdb as an optional tool.
JIT and TileIR reports optionally observe nvcc. A different nvcc/NVRTC version
produces a warning, not a failure: the versions need not match for every workflow.
The command does not change paths, download software, clear caches or repair an
installation. `cudavet` continues to analyze Go source for CUDA hazards.

## Smoke tests

```
cudadoctor -workflow=jit -smoke=compile -arch=compute_75
cudadoctor -workflow=jit -smoke=load -arch=compute_75
cudadoctor -workflow=jit -smoke=run -arch=compute_75 -json
```

`compile` produces fresh PTX without a CUDA context or artifact-cache reuse.
`load` also initializes device 0 and loads the module. `run` additionally allocates
an integer, initializes it to zero, launches a one-thread kernel, synchronizes,
and checks that the copied-back result is 42. Load/run require exclusive access
when used on a shared benchmark GPU. Choose a virtual target supported by the
selected NVRTC and, for load/run, the device. These tests validate this small JIT
path, not the application's kernels or performance.

Each library/smoke probe runs in a separate subprocess, bounded by `-timeout`
(default 15 seconds). A crash or timeout is a failed check. Tool probes have the
same timeout. Exit status is 0 if required checks pass (warnings may remain), 1
for a failed check, and 2 for invalid arguments. JSON has schema version 1 and
preserves individual check status, paths, versions, diagnostics and remedies.
