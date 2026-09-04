# Metal backend and e-graph integration

This document replaces the earlier proposal to create a `metal/x` backend. The
implemented package tree is `apple/x/metal`, and `examples/llama-infer` already
has a Metal execution path.

Status: superseded historical port plan. The source snapshot used for this
revision was commit `501e487`; the package and gap tables below are not a
current-tree audit.

## Current package map

| Concern | Current package | Status relevant to inference |
| --- | --- | --- |
| Metal bridge | `apple/x/metal` | Device, queue, library, pipeline, buffer, and command APIs exist |
| Memory | `apple/x/metal/metalmem`, `mempool` | Shared and pooled storage helpers exist; ownership and lifetime still need plan-level contracts |
| Model tensors | `apple/x/metal/metalsafetensors` | Safetensor support exists |
| Decoder kernels | `apple/x/metal/metalmegakernel` | Standalone and fused Llama-oriented kernels exist |
| Attention | `apple/x/metal/metalattn` | Decode-oriented attention kernels exist |
| KV state | `apple/x/metal/metalkv` and `SequenceKVCache` in the example | Several cache experiments exist; the example uses per-sequence state |
| Sampling | `apple/x/metal/metalsample` | GPU argmax exists |
| Graphs | `apple/x/metal/egraph`, `metalir`, `metalgraphexec` | Graph, extraction, IR, and execution experiments exist |
| Profiling | `apple/x/metal/metalprof` | Bounded timing records and profiling helpers exist |

Package presence is not integration evidence. The unified engine still executes
a mostly handwritten dispatch loop. Its `FuseEGraph` flag selects fused branches
directly; an extracted graph plan does not control the backend.

## Corrected design boundaries

### Metal is the execution target

Metal Performance Shaders is a GPU API. Calling MPS does not establish execution
on the Apple Neural Engine or AMX. Any ANE placement in graph data is a
suggestion until an ANE executor reports the operation as executed.

Metal SIMD-group and CUDA warp concepts can inform similar reductions, but
backend capability, resource, memory, and synchronization contracts remain
distinct. Do not infer performance parity from a structural mapping.

### Unified memory is not zero work

Apple Silicon's shared memory can avoid a discrete PCIe transfer, but loading
still maps or reads files, validates tensors, may convert dtype, allocates Metal
resources, and establishes their lifetime. The active inference path broadly
converts FP16 and BF16 checkpoint data into `float32`; therefore it is not an
instantaneous zero-copy checkpoint path. Half-weight kernels and pipeline states
now compile, but the forward loop still binds the float32 weights and pipelines.

No Metal buffer may retain a pointer into Go memory unless the implementation
proves the required pinning, alignment, lifetime, and deallocation contract.
Prefer allocator-owned storage with explicit release.

### Fusion must preserve the full operation

The obsolete document included an RMSNorm/QKV sample that did not compute the
RMS normalization and wrote only Q. It has been removed. A fusion is eligible
only when an independent reference verifies every output, tail dimension,
dtype, cache update, and alias behavior.

Current fused Metal kernels are useful candidates, not proof that the e-graph
selected them or that they outperform the unfused plan. The fused QKV shape also
repeats the RMS reduction for output-row threadgroups; profile this resource
tradeoff before promotion.

## Target integration

```text
model tensors and semantic graph
          |
          v
legal Metal alternatives from graph rewrites
          |
          v
resource and correctness validation
          |
          v
selected prefill/decode plan with stable buffer slots
          |
          v
Metal executor dispatches the selected operations
          |
          v
trace records the same plan and operation identities
```

The graph layer owns pattern matching and fusion alternatives. The extractor
produces explicit Metal operations. The runtime owns validation, profiling,
selection, memory planning, loading, invalidation, and execution. The executor
must not search the extracted graph for new patterns or choose a fused path from
a separate configuration flag.

## Required plan data

Each Metal operation records:

- function or library identity;
- input and output dtype, shape, stride, and layout;
- alignment and alias constraints;
- buffer slot and lifetime;
- grid and threadgroup geometry;
- static and dynamic threadgroup memory;
- supported device features;
- synchronization and readback boundaries; and
- semantic-reference identity and admitted tolerance.

Compilation rejects an unsupported plan before timing. Unknown resource limits
or missing tensors fail closed.

## Prefill and decode

Use separate phase plans. The current prompt path repeats the one-token decoder
schedule; one command buffer can reduce host overhead but does not turn those
operations into matrix-oriented prefill.

A prefill plan carries a token dimension through projection and causal attention
and writes all prompt KV positions. A decode plan carries a sequence batch plus
per-sequence positions and cache metadata. Shape buckets allow specialization
without changing semantics.

## Precision

Preserve FP16 or BF16 weight storage only after typed Metal kernels and loaders
pass shape, allocation-size, and numerical checks. FP32 accumulation may remain
part of the plan. Quantized weights require a separate format and accuracy
contract; they are not an implicit fallback.

Record actual allocation bytes and the executed dtype in benchmark provenance.

## Validation

For each candidate plan:

1. compare prefill and several cached decode positions with a semantic
   reference;
2. prove fused and unfused paths execute distinct operations;
3. mutate and restore one uploaded weight while holding shape and input fixed;
4. exercise aligned and tail dimensions;
5. interleave two sequence KV states and compare with serial runs;
6. inject compilation, allocation, command, and readback errors; and
7. profile end to end through the same buffer and command path used in serving.

A successful compile, a rendered e-graph, or a faster microkernel is not enough
to admit an inference plan.

## Work order

1. Finish fail-closed tensor, allocation, and command error handling.
2. Complete native 16-bit loading and verify compact device allocation.
3. Add a matrix-oriented prefill API and implementation.
4. Add real multi-sequence decode over isolated KV state.
5. Define authoritative Metal plan operations and stable buffer arenas.
6. Lower a small set of proven graph alternatives into those operations.
7. Profile finalists through the deployed executor and retain evidence.
8. Evaluate command reuse, ICBs, MPS, or specialized MSL per plan and workload.

No fixed cold-start or token-throughput target belongs in this architecture
document. Record such results in revision- and device-specific benchmark
artifacts after the correctness gates pass.
