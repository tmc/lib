# Qualified controlled GEMV counters

`counter-gemv workload` runs a deterministic float32 matrix/vector operation.
`simple` and `unroll` preserve dimensions, input, launch geometry and output.
Every completed output is checked against an exact CPU result. Operation names
include both matrix dimensions; metadata contains explicit phase, token, layer,
dtype, cache-length zero and optional false fields. Native NVTX markers live on
the submitting thread. No integrated Gemma slowdown is implied by this control.

Run every GPU command inside one existing `gpu-lock` hold. Commands refuse an
unset `GPU_LOCK_HELD`; they do not acquire a nested lock. With a freshly built
cudaprof executable and activity injection agent:

```sh
C="$HOME/tmp/cudaprof"
A="$HOME/tmp/libcudaprof_agent.so"
D="$HOME/tmp/gemv-counters"
mkdir -p "$D"
"$C" capture --inject "$A" --finalize exit-group --capture-range none \
  -o "$D/base.pprof" -- "$C" counter-gemv workload -implementation simple
"$C" capture --inject "$A" --finalize exit-group --capture-range none \
  -o "$D/target.pprof" -- "$C" counter-gemv workload -implementation unroll
"$C" shape-compare -domain driver -iteration-marker cudaprof_gemv \
  "$D/base.pprof" "$D/target.pprof" > "$D/shapes.json"
```

Select the observed exact identity, including optional-field presence, from the
shape receipt rather than inventing it or shortening a kernel symbol:

```sh
jq '.shape_qualification.matched[] | select(.identity.token == 0) | .identity' \
  "$D/shapes.json" > "$D/identity.json"
"$C" counter-receipt select -shape "$D/shapes.json" \
  -identity "$D/identity.json" > "$D/selection.json"
```

If shape qualification fails, selection and collection refuse. Original capture
paths must remain available: collect verifies their SHA-256 bytes, decodes their
complete journals and repeats the shape selection before touching the GPU.
Original stationarity receipts are retained separately. Defaults are rows 128,
columns 256; pass identical explicit dimensions to workload and collect when
changing them. A selected nonzero token requires `-token N` during collection.

Metric names must be supported by the native query on this device. The following
name is a request, not an availability guarantee:

```sh
"$C" counter-gemv collect -selection "$D/selection.json" -arm base \
  -implementation simple -metrics sm__cycles_elapsed.avg -max-passes 16 \
  > "$D/base-counters.json"
"$C" counter-gemv collect -selection "$D/selection.json" -arm target \
  -implementation unroll -metrics sm__cycles_elapsed.avg -max-passes 16 \
  > "$D/target-counters.json"
jq '.collection.metrics[0]' "$D/base-counters.json" > "$D/base-metric.json"
jq '.collection.metrics[0]' "$D/target-counters.json" > "$D/target-metric.json"
"$C" counter-receipt compare "$D/base-metric.json" "$D/target-metric.json"
```

Collection is a separate user replay with a bounded 1–64 passes and at most 32
metrics. Matrix, vector and output state are restored before every profiling
pass. Every pass returns an application identity and completed output hash.
Automatic native range identity checks the raw symbol, direct launch grid,
block and available shared memory; its trace must remain identical across
passes. Missing/extended/cooperative/graph launch geometry is refused, rather
than replaced with the historical launch parameters.

Receipts retain native metric descriptions, dimensional units, hardware units,
type/scope, CUPTI and driver versions, chip name, physical device UUID, executable
SHA, native completion/drop progress and collection failures. Source/input/output
hashes identify the controlled replay. Each metric has an unavailable value when
query, replay, identity, output, decode or cleanup fails. Compatible comparison
requires matching definitions and collection context. Source implementations may
differ while operation/input/output remain equivalent.

Ordinary controls run before and after collection and verify output. Collection
wall time includes metric query, configuration, replay and cleanup; it does not
measure isolated kernel overhead. Counter overhead remains explicitly unmeasured.
Counter values or elapsed controls alone do not establish bandwidth/cache/
occupancy/contention causes.

# Application integration

`counters.Collect` is the same process-local collector used by this command.
A Go application can call it in a separate profiling invocation with an exact
qualified `Selection`, capture SHA, metric list, lock declaration and complete
`Replay` callbacks. `Prepare` restores state outside profiling. `Run` submits
one selected direct kernel, synchronizes and returns the actual application
identity plus a hash of completed output. Native callback geometry and raw symbol
must agree independently. This permits a real application replay without
substituting the controlled GEMV for its launch.

A parent process cannot apply its Range Profiler object to a child process's
CUDA context. Arbitrary shell-command profiling therefore needs in-process
application integration. Graph replay is not admitted by this direct-launch
workflow; native graph/extended geometry and replay support must be qualified
before collection is broadened. The collector does not claim application
semantics from a callback symbol or copied marker alone.

CPU validation and CUDA-header ABI checking are separate from GPU validation:

```sh
cd nvidia
go test ./x/cudaprof/counters ./x/cuptirange ./x/cmd/cudaprof \
  -run 'TestCollect|TestReplayWitness|TestNativeReplayRange|TestCounterGEMV|TestDirectLaunchGeometry|TestKernelReplayGeometryWitness|ExampleCollect'
```
