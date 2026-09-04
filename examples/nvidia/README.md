# NVIDIA examples

Run a demo from its directory with `GOWORK=off go run .`. The raw-library
demos are intentionally small and each owns only the lifecycle and resources
needed for its package.

| Package | Demo |
| --- | --- |
| CUDA Driver | `driver-info`, `simple-addition` |
| NVRTC | `nvrtc-compile` |
| cuRAND | `curand-host` |
| cuBLAS | `cublas-sgemm` |
| cuBLASLt | `cublaslt-layout` |
| CUPTI | `cupti-version` |
| cuSOLVER | `cusolver-version` |
| cuSPARSE | `cusparse-version` |
| cuTENSOR | `cutensor-version` |
| NVML | `nvml-info` |
| NCCL | `nccl-allreduce` |
| cuDNN | `cudnn-tensor` |
| cuFile | `cufile-status` (Linux only) |
| Fabric Manager | `fabricmanager-status` |
| NVLink shared memory | `nvlshm-capabilities`, `tcpboot-roundtrip` |
| MoonEP host layer | `moonep-layout` |

`features` probes every package in one process and reports unavailable host
services explicitly. The compositions are `jit-vector-add` (CUDA context,
NVRTC, typed memory, and typed launch) and `gemm-allreduce` (cuBLAS followed by
NCCL without an intermediate host copy).

The higher-level `x` packages have focused workflows too:

| Package | Demo |
| --- | --- |
| `x/cudaattn` | `cuda-attention-compile` |
| `x/cudakv` | `cuda-kv-cache` |
| `x/cudasafetensors` | `cuda-safetensors` |
| `x/cudasample` | `cuda-token-sampling` |
| `x/cudatensor` and `x/cudafuse` | `cuda-tensor-dsl` |
| `x/cudatrace` and `x/perfetto` | `cuda-trace` |
| `x/cudaattn`, `x/cudakv`, and `x/cudasample` | `llama-infer` |

`cuda-trace` writes inspectable `.pftrace` artifacts beneath `~/tmp`; open
them with <https://ui.perfetto.dev>.

The DGX used for validation has no cuTENSOR or Fabric Manager library and a
cuDNN sublibrary unavailable to the installed package. Their demos are kept so
those runtime requirements are visible and reproducible.
