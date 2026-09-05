# nvidia

Pure-Go bindings for the NVIDIA CUDA libraries, generated from the vendor
headers and loaded at runtime with [purego]. No cgo, no C toolchain, and
no NVIDIA SDK at build time: programs cross-compile anywhere Go does, and
the libraries are located and loaded on first use.

[purego]: https://github.com/ebitengine/purego

## Packages

| Package | Binds | Use it for |
|---|---|---|
| driver | CUDA driver API (cuda.h) | contexts, memory, module loading, kernel launch, streams, events, graphs |
| nvrtc | NVRTC | compiling CUDA C++ to PTX at runtime |
| nvvm | NVVM | compiling NVVM IR and libdevice to PTX or LTOIR |
| nvjitlink | nvJitLink | linking device code to cubin or PTX |
| nvml | NVML | device enumeration, telemetry, and management |
| cublas | cuBLAS | dense BLAS on device buffers |
| cublaslt | cuBLASLt | lightweight matmul with algorithm selection |
| cudnn | cuDNN | deep-learning primitives |
| cufft | cuFFT | fast Fourier transforms |
| curand | cuRAND | random number generation (device and host) |
| cusolver | cuSOLVER (dense) | factorizations and eigensolvers |
| cusparse | cuSPARSE (generic API) | sparse matrices and SpMV/SpMM |
| cutensor | cuTENSOR | tensor contractions, reductions, permutations |
| cupti | CUPTI | profiling and activity tracing |
| nccl | NCCL | multi-GPU collectives |
| ncclep | NCCL EP (`libnccl_ep`) | expert-parallel dispatch and combine |
| ncclm2n | NCCL M2N (`libnccl_m2n`) | mesh-to-mesh tensor resharding |
| cufile | cuFile (GPUDirect Storage) | direct file↔GPU I/O; Linux only |
| fabricmanager | NVIDIA Fabric Manager | NVSwitch fabric partitions |
| x/... | — | idiomatic layer over the raw bindings; see x/DESIGN.md |

Every package follows the same conventions, so knowing one is knowing
them all.

## Load model

Packages load their library lazily on the first call, searching the
standard soname list for the platform. Nothing happens at import time.

  - `Load(path)` loads explicitly (empty path = default search list) and
    is idempotent; loading a second, different path fails rather than
    replacing the library.
  - `Loaded()`, `Path()` report state; `Close()` exists for test teardown.
  - Symbols absent from an older installed library are left unbound;
    calling one returns a `*SymbolError` instead of panicking. `Has()`
    and per-function `HasX()` probes let you check capability up front.

## Error model

Functions return `error` (nil on success). The concrete type is the
package's `Result` (or a small struct where the C API reports more than a
status), so `errors.Is`/`errors.As` work against exported codes:

	if errors.Is(err, nvml.ErrNotSupported) { ... }

Where the library exports an error-string function, messages are rich;
otherwise they fall back to the Go constant name and numeric code.

## The pure-Go kernel path

CUDA kernels can be compiled, linked, and launched entirely from Go at
runtime — no nvcc, no cgo:

 1. nvrtc: compile CUDA C++ source to PTX (`CreateProgram`, `Compile`,
    `GetPTX`).
 2. driver: `Init`, `NewDeviceGet`, `NewContext`, then either load the
    PTX directly (`NewModuleLoadData`) or JIT-link several inputs with
    `LinkCreate`/`LinkState` first.
 3. driver: `Module.GetFunction`, allocate with `MemAlloc`, copy with
    `DevicePtr.MemcpyHtoD`, and `Function.LaunchKernel`.

An nvjitlink package (LTO-capable linking) is planned; until it lands,
driver's LinkState covers JIT linking.

## Graphs

Capture a stream of work once, then replay it with near-zero launch
overhead:

	s.BeginCapture(driver.StreamCaptureModeGlobal)
	// ... enqueue kernels, copies ...
	graph, _ := s.EndCapture()
	exec, _ := graph.Instantiate(0)
	// exec.Launch(stream) each iteration

## Building an inference or training engine

The raw bindings are deliberately 1:1 with the C APIs: handles are
uintptrs, device memory is explicit, and nothing allocates behind your
back — polling loops (e.g. `Device.GetComputeRunningProcesses(buf)`) can
run allocation-free. Engine-level ergonomics — typed device slices,
context plumbing, kernel caching, graph fusion, test/bench integration —
are the charter of the `x/` packages; see x/DESIGN.md for the layer map.

## Platform notes

  - Linux and Windows are the primary targets; darwin loads nothing (no
    current NVIDIA driver) but everything compiles, so cross-platform
    builds and tests stay green.
  - cufile is Linux-only at the package level: its files carry a linux
    build constraint and the package disappears from other platforms.
  - nvptxcompiler (static PTX compiler library) does not fit the dynamic
    loading model and is planned as a separate cgo-opt-in package.

## Examples

Each package has a runnable example in its `example_test.go` showing the
primary workflow behind a load-check guard. Larger programs live in the
repository's `examples/` directory.
