# NVIDIA examples

Run a demo from its directory with `GOWORK=off go run .`. The raw-library
demos are intentionally small and each owns only the lifecycle and resources
needed for its package.

| Package | Demo |
| --- | --- |
| CUDA Driver | `simple-addition` |
| cuBLAS | `cublas-sgemm` |
| NVLink shared memory | `nvlshm-capabilities`, `tcpboot-roundtrip` |
| MoonEP host layer | `moonep-layout` |

The single-package demos for the CUDA Driver, NVRTC, cuRAND, cuBLASLt, cuFFT,
CUPTI, cuSOLVER, cuSPARSE, cuTENSOR, NVML, NCCL, cuDNN, cuFile and Fabric
Manager are the `Example` functions in those packages under
`github.com/tmc/lib/nvidia`.

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
| `x/nvrtcx`, `x/cudasafetensors`, and `cublas` | `laya-infer` |

`cuda-trace` writes inspectable `.pftrace` artifacts beneath `~/tmp`; open
them with <https://ui.perfetto.dev>.
