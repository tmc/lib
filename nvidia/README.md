# NVIDIA bindings for Go

This module contains Go bindings for NVIDIA libraries, loaded dynamically with
[purego](https://github.com/ebitengine/purego), and higher-level packages under
[x/](x/README.md). The module requires Go 1.27 or later.

Building the bindings does not require NVIDIA headers. Running a native
operation requires the library and any devices, drivers or services that the
operation uses. Platform availability and build constraints vary by package;
cuFile, for example, is a Linux-only package.

## Packages

| Package | Binds | Use it for |
|---|---|---|
| [driver](driver/) | CUDA driver API (cuda.h) | contexts, memory, module loading, kernel launch, streams, events, graphs |
| [nvrtc](nvrtc/) | NVRTC | compiling CUDA C++ to PTX, cubin, LTOIR, or TileIR at runtime |
| [nvvm](nvvm/) | NVVM | compiling NVVM IR and libdevice to PTX or LTOIR |
| [nvjitlink](nvjitlink/) | nvJitLink | linking device code to cubin or PTX |
| [nvfatbin](nvfatbin/) | nvFatbin | packaging alternative CUDA device images into fatbinaries |
| [nvml](nvml/) | NVML | device enumeration, telemetry, and management |
| [cublas](cublas/) | cuBLAS | dense BLAS on device buffers |
| [cublaslt](cublaslt/) | cuBLASLt | lightweight matmul with algorithm selection |
| [cudnn](cudnn/) | cuDNN | deep-learning primitives |
| [cufft](cufft/) | cuFFT | fast Fourier transforms |
| [curand](curand/) | cuRAND | host APIs for random number generation |
| [cusolver](cusolver/) | cuSOLVER (dense) | factorizations and eigensolvers |
| [cusparse](cusparse/) | cuSPARSE (generic API) | sparse matrices and SpMV/SpMM |
| [cutensor](cutensor/) | cuTENSOR | tensor contractions, reductions, permutations |
| [cudss](cudss/) | cuDSS | direct sparse linear solves: analysis, factorization, solve |
| [cublasmp](cublasmp/) | cuBLASMp | multi-process, multi-GPU dense linear algebra over an NCCL process grid |
| [cusolvermp](cusolvermp/) | cuSOLVERMp | multi-process, multi-GPU dense factorization and solve |
| [custatevec](custatevec/) | cuStateVec (cuQuantum) | state-vector quantum circuit simulation |
| [cutensornet](cutensornet/) | cuTensorNet (cuQuantum) | tensor-network path finding and contraction |
| [cudensitymat](cudensitymat/) | cuDensityMat (cuQuantum) | many-body operator action and expectation values |
| [vpi](vpi/) | VPI (Vision Programming Interface) | context and stream lifecycle, image wrapping, rescale |
| [cupti](cupti/) | CUPTI | profiling and activity tracing |
| [sanitizer](sanitizer/) | Compute Sanitizer public API | host callbacks, instrumentation patch management, and callback-safe memory operations |
| [dcgm](dcgm/) | DCGM client library (libdcgm) | host-engine telemetry, entity and field discovery, health and diagnostics |
| [nccl](nccl/) | NCCL | multi-GPU collectives |
| [ncclep](ncclep/) | NCCL EP (`libnccl_ep`) | expert-parallel dispatch and combine |
| [ncclm2n](ncclm2n/) | NCCL M2N (`libnccl_m2n`) | mesh-to-mesh tensor resharding |
| [nvshmem](nvshmem/) | NVSHMEM host API (`libnvshmem_host`) | symmetric heap allocation and PE-to-PE communication; host calls only |
| [gdrcopy](gdrcopy/) | GDRCopy (`libgdrapi`) | mapping GPU memory into a process's address space |
| [cufile](cufile/) | cuFile (GPUDirect Storage) | direct file↔GPU I/O; Linux only |
| [nvcomp](nvcomp/) | nvCOMP (batched LZ4) | compressing and decompressing independent chunks on device |
| [nvjpeg2k](nvjpeg2k/) | nvJPEG2000 | JPEG 2000 decoding on device |
| [nvdec](nvdec/) | NVDEC (nv-codec-headers) | hardware video decoding: parser, decoder, surface mapping |
| [nvenc](nvenc/) | NVENC (nv-codec-headers) | hardware video encoding through the driver's function table |
| [fabricmanager](fabricmanager/) | NVIDIA Fabric Manager | NVSwitch fabric partitions |
| x/... | — | idiomatic layer over the raw bindings; see the [generated package inventory](x/README.md) |

## Loading libraries

The direct binding packages expose `Load(path)`, `Loaded()`, `Path()` and
`Close()`. A native call can trigger lazy loading. Use an explicit load when
you want to handle availability before starting a workflow. Consult each
package's loader documentation for its environment variable, search list and
supported platforms. Loading a library does not establish that every symbol,
device or operation is available.

`Has` and generated capability predicates check for individual symbols.
Functions with an error result report loading, missing-symbol or native-status
errors through that result. Wrappers without an error result can panic when a
library or symbol is unavailable. Inspect the function's signature and comment;
do not assume every operation follows one error convention.

`Close` is for controlled teardown. Release native resources first, and do not
close a library while other goroutines or callbacks may still use it.

## Native resources and errors

The raw bindings closely follow the C APIs. A returned handle usually needs an
explicit release operation; Go garbage collection does not release native
resources. Buffers, offsets, pointer lifetimes and synchronization remain the
caller's responsibility unless a particular helper documents otherwise.

Status-returning APIs translate failures to Go errors. Many packages expose a
`Result` type and named error constants; cuFile instead reports its structured
native status and also has byte-count-returning I/O methods. Use the package's
actual error types and codes with `errors.Is` or `errors.As` as appropriate.

The CUDA driver associates a current context with an OS thread. Raw context
workflows must keep related calls on that thread with `runtime.LockOSThread`.
The [cuda wrapper](x/cuda/) provides `Device.WithContext` for scoped operations.

## Examples and validation

Direct binding packages include `example_test.go` files. Examples requiring
NVIDIA libraries or devices illustrate native workflows; those without an
`Output:` directive are compiled but are not executed by `go test`. They do
not establish that the workflow has run successfully on a GPU. Read each
example's prerequisites before adapting it.

Examples with an `Output:` directive are executed and checked by:

```sh
go test ./... -run '^Example' -count=1
```

This command covers packages available on the current platform. It does not
exercise Linux-only cuFile on macOS, or opt-in hardware tests behind build
tags. Compilation, an unavailable-library return path and a successful native
workflow are different checks.

The [PTX/SASS trace example](examples/ptx-sass-trace/) demonstrates a larger
compile-and-launch workflow. The [x package inventory](x/README.md) lists
higher-level memory, compilation, graph and tracing helpers.

## Regeneration

[upstream.json](upstream.json) records package inputs and generation commands.
The generator repository is `tmc/libgen` (formerly `tmc/codegen`). Read the
package's recorded command and header pins rather than choosing a generator
by name: several packages use dedicated survey commands.

Generate into a separate output directory, then review the diff and validate
the affected package. Generated API fixes belong in libgen; editing generated
output alone loses the fix on the next run. Preserve historical provenance and
record which new outputs were actually regenerated.

Use `-allow-api-removal` only for an intended, reviewed removal. An unexpected
removal can indicate a wrong header pin or generator command.
