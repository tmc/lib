# Package Spec: ggml

Package ggml provides low-level Go bindings for ggml tensors and compute graphs.

- Import path: `github.com/tmc/lib/ggml`
- ABI package: `internal/abi`
- ABI library base name: `ggml`
- Public API note: The generated package is a low-level tensor and graph binding. Model loading, sampling loops, KV-cache policy, and serving orchestration belong in handwritten packages above it.

## Parse Inputs

- Headers: `ggml.h`

## Semantic Sources

- `llama.h` (reference): higher-level model/session API shape and tokenizer boundary
- `llama.cpp` (implementation): semantic reference for model serving loops; not an ABI input
- `docs/ggml-backends.md` (docs): backend and hardware gate semantics
- ggml.h is the only ABI parse input for the first target.
- llama.h and llama.cpp are semantic references only; they must not be parsed for ABI symbols.
- CPU tensor metadata and graph construction are cheap default gates.
- CUDA, Metal, Vulkan, and other provider execution are manual-gated adapters.

## Capability Metadata

- Provider metadata: generated `ProviderInfo` fields on `Capabilities`.
- Health/readiness: generated `HealthSnapshot` fields on `Capabilities`.

## Resources

- `Context` wraps `abi.Context` and is owned.
- `Tensor` wraps `abi.Tensor` and is borrowed.
- `Graph` wraps `abi.CGraph` and is borrowed.
- `BackendBuffer` wraps `abi.BackendBuffer` and is owned.

## Operations

- `NewTensor1D`: NewTensor1D allocates a one-dimensional tensor in the context. wraps `ggml_new_tensor_1d` calls backend `newTensor1D`.
- `TensorData`: TensorData returns the backend-owned data pointer. wraps `ggml_get_data` calls backend `tensorData`.
- `Add`: Add appends an elementwise add node to the graph. wraps `ggml_add` calls backend `add`.
- `Mul`: Mul appends an elementwise multiply node to the graph. wraps `ggml_mul` calls backend `mul`.
- `BuildForward`: BuildForward builds a forward compute graph rooted at t. wraps `ggml_build_forward` calls backend `buildForward`.
- `GraphCompute`: GraphCompute computes g on the CPU backend and stops waiting when ctx is canceled. wraps `ggml_graph_compute` calls backend `graphCompute`; blocking, context policy `required`.

## Adapters

- `cuda` build tag `ggml_cuda`; package `github.com/tmc/lib/ggml/ggmlcuda`; runtime `cuda`; device kind `DeviceCUDA`; manual gate `cuda`.
- `metal` build tag `ggml_metal`; package `github.com/tmc/lib/ggml/ggmlmetal`; runtime `metal`; device kind `DeviceApple`; manual gate `metal`.

## Unsafe Policy

- `TensorData` may expose `unsafe.Pointer`: ggml tensor data may be owned by ggml contexts, mmap-backed model files, or provider buffers and cannot be represented as a Go slice by default Safer layer: A higher-level model package may expose typed views only after it owns the storage and dtype policy..
- `BackendBuffer` may expose `unsafe.Pointer`: provider buffers are runtime-owned memory with backend-specific lifetime and placement rules Safer layer: Provider adapters should wrap backend buffers in typed Go resources when ownership is known..

## Element Sets

- `Element` constraint `~float32 | ~int32` maps to `DType`:
  - `float32` -> `DTypeF32` (enabled)
  - `Float16` -> `DTypeF16` (disabled)
  - `int32` -> `DTypeI32` (enabled)
