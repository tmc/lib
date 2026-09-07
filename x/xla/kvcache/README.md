# Device KV cache

`kvcache.New` initializes fixed-capacity key/value history on the engine's
default compilation device. Use `Cache.Device` when uploading inputs. `Append`
accepts device Q/K/V tensors and returns causal attention for the appended
chunk. Batch members share a common length. Supported elements are f32, f16,
and bf16; reduced-precision attention computes in f32.

Set QueryHeads and KVHeads for grouped-query attention. QueryHeads must be a
positive multiple of KVHeads. Inputs then use [batch..., heads, chunk, features],
and BatchShape excludes the head dimension. Query head h uses KV head
h/(QueryHeads/KVHeads). Persistent history retains only KVHeads; the graph groups
queries for dot products without explicitly replicating K/V tensors. Setting
both counts to zero retains unheaded inputs.

History remains on the device. Updates produce replacement device buffers;
this API does not promise in-place mutation or buffer donation. Each append
uploads an eight-byte position scalar. The cache retains a decode executable
and at most one prefill executable, keyed by chunk length. Different prefill
lengths replace that second executable. `Reset` resets logical length and keeps
the buffers and executables; masking excludes the stale suffix.

Close returned outputs and the cache before closing the borrowed engine.
Inputs remain caller-owned. Synchronize all calls externally. Validation,
compilation and execution errors preserve logical history. Failure to release
replaced history, the position buffer or an evicted program closes the cache.
Native destruction failures can still leave resources allocated by the plugin.

There is no eviction, per-batch variable-length state, sharded cache, or
hardware-specific donation optimization. Full capacity
returns an error before dispatch. Multi-device clients are accepted, but one
cache and its inputs reside on the device selected for its initialization.

Run host tests with `go test ./...`. Set `XLA_PJRT_CPU_PLUGIN` to an installed
CPU plugin to run native tests, including f32/f16/bf16 full-prefix references,
grouped/multi-query head mapping, history shapes, chunked prefill, decode reuse,
cancellation, reset, and capacity rejection.
