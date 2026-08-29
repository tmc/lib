# lib

This repository contains independent, multi-module Go libraries and bindings.

## Repository Layout

There is no root module. Each subdirectory is an independent Go module with its own versioning and release tags (`<dir>/vX.Y.Z`).

| Module | Import Path | Description |
|---|---|---|
| `fuse` | `github.com/tmc/lib/fuse` | Pure Go FUSE interface |
| `ggml` | `github.com/tmc/lib/ggml` | GGML backend integration |
| `gonccl` | `github.com/tmc/lib/gonccl` | High-level NCCL interface |
| `gopccl` | `github.com/tmc/lib/gopccl` | Cross-accelerator collective communication interface |
| `gorccl` | `github.com/tmc/lib/gorccl` | High-level RCCL interface |
| `kvm` | `github.com/tmc/lib/kvm` | Linux KVM API wrapper |
| `libp2p` | `github.com/tmc/lib/libp2p` | Pinned libp2p host/node wrapper |
| `lldb` | `github.com/tmc/lib/lldb` | Purego LLDB C API bindings |
| `mgccl` | `github.com/tmc/lib/mgccl` | Multi-GPU collective communication pump |
| `nbd` | `github.com/tmc/lib/nbd` | Pure Go NBD (Network Block Device) server and client |
| `nvidia` | `github.com/tmc/lib/nvidia` | Purego NVIDIA CUDA/driver bindings |
| `pccl` | `github.com/tmc/lib/pccl` | Purego PCCL bindings |
| `rocm` | `github.com/tmc/lib/rocm` | Purego ROCm/HIP/RCCL bindings |
| `virtiofs` | `github.com/tmc/lib/virtiofs` | VirtioFS server and device framework |
| `vulkan` | `github.com/tmc/lib/vulkan` | Purego Vulkan API bindings |

## Usage

To use a package, `go get` the specific submodule:

```bash
go get github.com/tmc/lib/nvidia@latest
```

Note: `go get github.com/tmc/lib` fails by design because there is no root module.

## Development

Local development uses `go.work` at the root of the repository.

To build and test all modules:

```bash
make
```
