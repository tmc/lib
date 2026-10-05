# Native search memory gate

`TestNativeSearchMemoryProducerConsumer` is a bounded Linux/native gate over a
finite FP32 affine fixture. It compiles scalar, float4 and wrong-output kernels,
including nonmultiple-of-four tails, and exercises CUPTI collection, executable
ABBA evaluation, both search strategies, trajectory admission, cache publication,
catalog updates and a second workload's replay. A category is a heuristic;
counter collection does not establish the causal value of its recommended rule.
No speedup threshold or promotion is asserted.

Run only after independent source/packet review on an authorized DGX. From the
sealed worktree's `nvidia` directory, the coordinator sets absolute persistent
receipt paths and the exact source manifest SHA256. The receipt directory must
not exist. The test verifies every manifested deployed source file before opening
the device. Submit this payload through the coordinator's independently reviewed
bounded watchdog, which must contain the whole nested gpu-lock process group:

```
/home/tmc/ml-explore/bin/gpu-lock env GOWORK=off \
  CUDASEARCH_MEMORY_NATIVE=1 \
  CUDASEARCH_MEMORY_RECEIPTS="$receipt_dir" \
  CUDASEARCH_MEMORY_SOURCE_MANIFEST="$source_manifest" \
  CUDASEARCH_MEMORY_MANIFEST_SHA256="$manifest_sha256" \
  go test -json -count=1 -tags=native -timeout=120s \
  -run '^TestNativeSearchMemoryProducerConsumer$' ./x/cudasearch
```

Retain the full command, watchdog status and JSON stdout/stderr outside the new
receipt directory. A timeout applied only to the outer command does not establish
descendant containment. Dispatch is held until the watchdog and packet are reviewed.
A skip, absent counter, unsuccessful collection or partial
packet leaves native qualification pending. Never replace unavailable metrics
with theoretical occupancy, a host duration or a zero value.

The packet retains CUDA source, explicitly selected PTX loader input, compiler flags/logs, driver/NVRTC/CUPTI
library hashes and versions, source manifest, exact input/reference/readback
bytes, raw collection quality and metrics, execution/trial receipts, catalogs,
policy identities and consulted trajectory IDs. Inputs are finite exactly
representable FP32 values; the oracle requires exact output and signed-zero bits
with `--fmad=false`. Buffers are poisoned before each correctness or timed run.
Event timings cover one kernel on a private stream and include its event boundary;
they are not host time or uninstrumented throughput claims. CUPTI warm cache
collection is separate from event comparisons but initializes the profiler for
the process. Timing and profiling overhead have no benefit claim in this gate.

Controls include same-implementation ABBA, a numerically wrong native kernel
refused before timing, dropped/mixed-device capture refusal, and quarantining an
actually produced persisted trajectory whose contents are corrupted. The second
task also runs an explicit prior-category replay experiment; that category stays
pending and does not relabel its independently collected profile. Replay from
another category is excluded from the profile-guided epoch. Search decisions are
frozen before evaluation, and real rewards can favor or penalize either path.

Cancellation does not release in-flight storage. Every timed launch joins stream
completion and destruction drains before freeing buffers. The external watchdog
is necessary for a stuck native call. Cleanup errors fail qualification; a
collector cleanup error retains its owner via `io.Closer` for retry.
The fixture has one ordered teardown: pending collectors, evaluators, executables,
module, then device. The first failure stops dependent destruction and retains
the owner for process lifetime. Host controls exercise repeated failure and retry
at every stage; they do not establish a native drain.

Metric names follow [NVIDIA's metric mappings](https://docs.nvidia.com/cuda/archive/11.0/cupti/Cupti/r_main.html)
and [SM throughput definition](https://docs.nvidia.com/nsight-compute/CustomizationGuide/index.html).
Actual availability and dimensions must be checked on the target. The adapter
currently accepts the native `percent` dimensional token only; an unexpected
token requires reviewing the retained native properties before extending it.


The virtual WithArch compilation options are unchanged. The producer selects
OutputPTX explicitly with LoadOutput and retains the exact selected bytes in
fixture.ptx; it never falls back to another output. This binds the loader input,
not the driver's resulting JIT CUBIN or SASS. compiler output type/extent and
artifact digest are retained in toolchain.json. Partial-load cleanup ownership
remains in the existing io.Closer retry chain.
