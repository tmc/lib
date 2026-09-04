# Metal Mixture-of-Experts execution design

This document proposes a GPU execution path for routed Mixture-of-Experts models
on Metal. It replaces the earlier monolithic-kernel design, whose intermediate
storage and cross-threadgroup assumptions were not implementable as written.

Status: proposal. Package `lm/moe` provides the current CPU routing and expert
file-cache reference; no `apple/x/metal/metalmoe` package or Metal execution
path exists.

## Semantic contract

The plan must describe the model's exact routing rules:

- gate computation and optional score correction;
- grouping constraints, tie-breaking, and top-K selection;
- whether selection uses biased scores while weights use unbiased logits;
- normalization and routing scale;
- routed and shared experts;
- expert weight dtype, quantization format, group size, scales, and biases; and
- output accumulation dtype and order.

These choices are model semantics, not tuning parameters. The existing
`GroupedTopKRouter` is a useful CPU reference for its declared policy, but every
target model needs schema validation against its checkpoint and configuration.

## Recommended pipeline

Use a small pipeline of specialized stages rather than one kernel:

1. **Gate and route.** Compute expert scores and produce expert IDs and routing
   weights for every token.
2. **Group tokens by expert.** Build a stable token-to-expert work list, counts,
   and offsets. Padding may be added for matrix tile boundaries.
3. **Grouped gate/up projection.** Execute expert-specific gate and up GEMMs and
   fuse the SwiGLU epilogue where the chosen primitive supports it.
4. **Grouped down projection and weighted reduction.** Execute the down GEMM,
   multiply by routing weights, and scatter or reduce contributions to the token
   output.

The physical plan may use two, three, four, or more dispatches depending on
batch size, dtype, available matrix primitives, and whether routing metadata can
be consumed directly. Dispatch count is an output of plan selection, not a goal
by itself.

```text
tokens
  |
  v
gate + top-K ----> expert IDs and weights
  |
  v
stable expert grouping ----> expert offsets and token map
  |
  v
grouped gate/up GEMM + SwiGLU
  |
  v
grouped down GEMM + weighted scatter/reduce
  |
  v
token outputs
```

## Why not one monolithic expert kernel

Threadgroup memory is private to one threadgroup. It cannot hold a large hidden
activation once and share it with other output threadgroups. A grid with several
threadgroups per token must therefore materialize the hidden activation, repeat
the gate/up computation, or introduce another synchronization boundary.

A single threadgroup per token avoids cross-threadgroup sharing but provides
insufficient parallelism for large expert projections and cannot keep a typical
intermediate dimension in registers or bounded threadgroup memory. Tiling the
hidden dimension while accumulating every output dimension also creates a large
accumulator problem.

For those reasons, the design treats materialization and dispatch boundaries as
explicit tradeoffs. Grouped matrix operations with fused epilogues are the
starting point. A later custom kernel may fuse adjacent stages only when its
resource model and end-to-end profile justify it.

## Routing and grouping

For decode batch one, grouping has little weight reuse and its metadata cost may
dominate. Keep a direct small-batch path that executes the selected experts
without a sort when measurement favors it.

For larger decode batches and prefill, group token-expert pairs by expert so
that matrix tiles reuse each expert's weights. The grouping operation produces:

- stable expert offsets;
- original token indices;
- routing-slot indices;
- per-expert padded counts; and
- a validity count so padded work cannot affect output.

Tie-breaking and grouping order must be deterministic where the reference
requires it. Routing remains on device in the production path to avoid a host
readback boundary.

## Weight and activation layout

The plan records rather than assumes:

- expert-major or packed expert weight order;
- matrix orientation and tile packing;
- quantization packing and scale granularity;
- activation dtype and alignment;
- token/expert work-list layout; and
- output scatter or reduction layout.

An expert-major logical checkpoint layout may still require repacking for the
selected Metal matrix primitive. Repacking is a versioned load-time plan with a
memory cost and cache key. “Contiguous” alone does not establish coalescing or
cache behavior.

Use FP32 accumulation for routing weights and output reduction unless a narrower
policy passes its own accuracy gate. Gate/up and down intermediates may use a
compact dtype when the kernel contract and reference comparison admit it.

## Integration choices

### One C or Go call around existing operations

Combining several API calls behind one Go-to-native boundary can reduce host FFI
overhead. It does not by itself reduce Metal dispatches, command-buffer barriers,
or intermediate materialization. Measure it as an API batching optimization and
retain the operation trace.

### Existing matrix primitives

Prefer an existing validated grouped or gather-matrix primitive when it matches
the model layout. Fuse activation and weighted-reduction epilogues only through
documented primitive support or explicit kernels. MPS and MLX operations execute
on the GPU; they do not imply ANE execution.

### Custom Metal kernels

Custom kernels are appropriate for routing, grouping, quantized unpacking, and
epilogues when profiles show those stages matter. A custom expert GEMM requires
shape buckets, resource checks, tail handling, and comparison with the best
available matrix primitive.

## Memory and performance model

For one routed token-expert pair, a dense three-matrix expert reads approximately

```text
bytes_per_weight * (2 * hidden_dim * intermediate_dim
                    + intermediate_dim * hidden_dim)
```

before quantization metadata, cache reuse, shared experts, activations, routing,
and output traffic. This is a modeled upper-level accounting identity, not
observed DRAM traffic. Actual transactions depend on batching, expert reuse,
packing, caches, and the matrix implementation.

Intermediate bytes can be modeled separately from weight bytes. Saving a small
activation materialization may not matter when expert weights dominate, while
grouping several tokens for one expert can materially improve weight reuse.
Profile the complete phase with the real expert distribution.

Do not publish fixed percentage gains or launch-overhead estimates without raw
measurements tied to the source revision, model, batch and sequence shapes,
device, dtype, exact path, and correctness result.

## Expert residency

File-backed expert caching is not GPU residency management. A production
resident-expert cache needs explicit Metal-buffer ownership, load completion,
in-flight references, eviction safety, memory-pressure behavior, and latency
accounting.

An expert required by an admitted step must be resident before dispatch. If it
is not, the scheduler waits within a declared deadline or rejects the step.
Dropping the expert or substituting another expert changes model semantics and
is not an eviction policy.

## Correctness gates

Compare the complete MoE output with an independent unfused reference over:

- decode batch one and multi-sequence decode;
- several prefill lengths;
- repeated and highly skewed expert assignments;
- ties and grouping boundaries;
- zero, one, and maximum valid routed experts where the model permits them;
- shared-expert combinations;
- aligned and tail matrix dimensions;
- every supported dtype and quantization format; and
- expert-cache miss, cancellation, and error paths.

Add mutation controls for gate weights, each expert projection, routing weights,
and token grouping. Verify that permuting the grouped work list while preserving
metadata does not change the semantic result beyond the declared tolerance.

## Benchmark protocol

Measure stages and the end-to-end MoE block:

- gate and route;
- grouping;
- gate/up projection and activation;
- down projection and reduction;
- residency misses and repacking; and
- total wall and device time.

Use the same token/expert distribution for the candidate and reference. Report
batch, prompt or context length, expert counts, top-K, selected-expert histogram,
dtype, quantization, resident set, and device. A microbenchmark win is promoted
only if the real model phase also improves.

## Implementation order

1. Freeze and test the model-specific routing and tensor-layout contract.
2. Add a GPU gate/router and compare its IDs and weights with the CPU reference.
3. Add stable on-device grouping with mutation and padding controls.
4. Build a grouped path from existing validated matrix primitives.
5. Fuse SwiGLU and weighted-reduction epilogues where the trace shows a useful
   boundary.
6. Add quantized formats one at a time with independent references.
7. Add device-resident expert lifetime and eviction rules.
8. Profile custom GEMM candidates only for buckets where existing primitives are
   inadequate.

This sequence earns a correct grouped implementation before investing in broad
kernel fusion.
