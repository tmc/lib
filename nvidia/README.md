# nvidia

Pure-Go bindings for the NVIDIA CUDA libraries, generated from the vendor
headers and loaded at runtime with [purego]. No cgo, no C toolchain, and
no NVIDIA SDK at build time: programs cross-compile anywhere Go does, and
the libraries are located and loaded on first use. Two exceptions:
`x/nvlshm` and `x/moonep` are `//go:build !windows`, because they exchange
file descriptors over a Unix domain socket.

[purego]: https://github.com/ebitengine/purego

## Packages

| Package | Binds | Use it for |
|---|---|---|
| driver | CUDA driver API (cuda.h) | contexts, memory, module loading, kernel launch, streams, events, graphs |
| nvrtc | NVRTC | compiling CUDA C++ to PTX, cubin, LTOIR, or TileIR at runtime |
| nvvm | NVVM | compiling NVVM IR and libdevice to PTX or LTOIR |
| nvjitlink | nvJitLink | linking device code to cubin or PTX |
| nvfatbin | nvFatbin | packaging alternative CUDA device images into fatbinaries |
| nvml | NVML | device enumeration, telemetry, and management |
| cublas | cuBLAS | dense BLAS on device buffers |
| cublaslt | cuBLASLt | lightweight matmul with algorithm selection |
| cudnn | cuDNN | deep-learning primitives |
| cufft | cuFFT | fast Fourier transforms |
| curand | cuRAND | random number generation (device and host) |
| cusolver | cuSOLVER (dense) | factorizations and eigensolvers |
| cusparse | cuSPARSE (generic API) | sparse matrices and SpMV/SpMM |
| cutensor | cuTENSOR | tensor contractions, reductions, permutations |
| cudss | cuDSS | direct sparse linear solves: analysis, factorization, solve |
| cublasmp | cuBLASMp | multi-process, multi-GPU dense linear algebra over an NCCL process grid |
| cusolvermp | cuSOLVERMp | multi-process, multi-GPU dense factorization and solve |
| custatevec | cuStateVec (cuQuantum) | state-vector quantum circuit simulation |
| cutensornet | cuTensorNet (cuQuantum) | tensor-network path finding and contraction |
| cudensitymat | cuDensityMat (cuQuantum) | many-body operator action and expectation values |
| vpi | VPI (Vision Programming Interface) | context and stream lifecycle, image wrapping, rescale |
| cupti | CUPTI | profiling and activity tracing |
| sanitizer | Compute Sanitizer public API | host callbacks, instrumentation patch management, and callback-safe memory operations |
| dcgm | DCGM client library (libdcgm) | host-engine telemetry, entity and field discovery, health and diagnostics |
| nccl | NCCL | multi-GPU collectives |
| ncclep | NCCL EP (`libnccl_ep`) | expert-parallel dispatch and combine |
| ncclm2n | NCCL M2N (`libnccl_m2n`) | mesh-to-mesh tensor resharding |
| nvshmem | NVSHMEM host API (`libnvshmem_host`) | symmetric heap allocation and PE-to-PE communication; host calls only |
| gdrcopy | GDRCopy (`libgdrapi`) | mapping GPU memory into a process's address space |
| cufile | cuFile (GPUDirect Storage) | direct file↔GPU I/O; Linux only |
| nvcomp | nvCOMP (batched LZ4) | compressing and decompressing independent chunks on device |
| nvjpeg2k | nvJPEG2000 | JPEG 2000 decoding on device |
| nvdec | NVDEC (nv-codec-headers) | hardware video decoding: parser, decoder, surface mapping |
| nvenc | NVENC (nv-codec-headers) | hardware video encoding through the driver's function table |
| fabricmanager | NVIDIA Fabric Manager | NVSwitch fabric partitions |
| x/... | — | idiomatic layer over the raw bindings; see the [generated package inventory](x/README.md) |

Every package follows the same conventions, so knowing one is knowing
them all.

## Load model

Packages load their library lazily on the first call, searching the
standard soname list for the platform. Nothing happens at import time.
Every package builds on every platform Go supports; what varies is
whether the vendor ships a library there. Where NVIDIA publishes no
build for an operating system, the search list for it is empty and the
first call reports that, rather than searching for a name that cannot
exist. NCCL, Fabric Manager, cuFile, DCGM, cuBLASMp, cuSOLVERMp, nvCOMP
and GDRCopy are Linux-only for that reason; CUPTI, cuTENSOR, the
cuQuantum packages and nvJPEG2000 ship on Windows but no Windows name is
configured for them yet, so they too report an empty search list there.

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

## Regenerating

`cufft/init.gen.go` carries hand-written declarations -- `LibraryError`,
`LibraryAttempt` and their helpers -- that no generator commit produces. The
generator's API-removal check refuses to regenerate the package for that
reason, and the refusal is correct: regenerating cufft today would delete that
error type and the loader code that returns it. Do not pass
`-allow-api-removal` to get past it. The package can be regenerated once
`contrib/cuda` emits the loader that `internal/ffigen` already emits, or once
those declarations are moved into a maintained file beside the generated ones,
which the check does not read.

`-allow-api-removal` is for a removal or rename you intend and will review in
the diff, such as regenerating nccl to pick up the `HascommInitAll` ->
`HasCommInitAll` fix. It is not a way to quiet a check that is telling you an
input is wrong: a header pinned to the wrong revision reports the same way, as
hundreds of unrelated declarations going missing at once.

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

The `nvjitlink` package provides LTO-capable device-code linking;
`x/nvjitlinkx` owns linker inputs, diagnostics, and cleanup. The driver's
`LinkState` also supports JIT linking.

## Graphs

Capture a stream of work once, then replay it with near-zero launch
overhead:

	s.BeginCapture(driver.StreamCaptureModeGlobal)
	// ... enqueue kernels, copies ...
	graph, _ := s.EndCapture()
	exec, _ := graph.Instantiate(0)
	// exec.Launch(stream) each iteration

## Building an inference or training engine

The raw bindings closely follow the C APIs. Callers manage device memory
explicitly; sized-output helpers return allocated Go slices.
Engine-level ergonomics — typed device slices,
context plumbing, kernel caching, graph fusion, test/bench integration —
are the charter of the `x/` packages; see the
[generated package inventory](x/README.md).

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
