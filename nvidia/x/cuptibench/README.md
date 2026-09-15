# cuptibench

`cuptibench` collects CUPTI metrics for replayable CUDA workloads and reports
measurements through Go benchmarks. It supports whole operations, nested named
regions, and individual kernel ranges. `Collect` also works in ordinary tests.

The implementation has unit, race, real Go-runner, and profile round-trip tests.
Physical GPU tests are opt-in. Cross-compilation does not qualify GPU behavior.

## Workloads and sub-benchmarks

A workload supplies its streams, a preparation callback, the operation to execute,
and a completion callback. One call to `Run` is one logical operation.
Preparation restores the beginning of the complete configured operation sequence
before each timing repetition or profiling pass.

```go
work := cuptibench.Workload{
    Streams: []*cuda.Stream{stream},
    View: "full",
    Preparation: "restore input and KV state for the selected token position",
    Prepare: restore,
    Synchronize: func() error { return stream.Synchronize(nil) },
    Run: func(s *cuptibench.Scope) error {
        if err := s.Region("attention", attention); err != nil {
            return err
        }
        return s.Region("ffn", ffn)
    },
}
```

The application owns the callbacks and CUDA resources in this sketch. Use the
actual forward implementation, including its dependencies and graph behavior.
Do not replace it with a list of unrelated microbenchmarks and label their sum
as a decode pass.

Create an explicitly bounded plan and close it before releasing the device:

```go
metrics, err := cuptibench.Resolve("launches")
if err != nil {
    b.Fatal(err)
}
plan, err := cuptibench.New(dev, cuptibench.Config{
    Metrics: metrics,
    Mode: cuptibench.Regions,
    CachePolicy: cuptirange.CachePreserve,
    MaxDepth: 3,
    Operations: 1,
    Limits: cuptirange.Limits{
        MaxRanges: 64, MaxRangeTreeNodes: 128,
        MaxCounterBytes: 16 << 20, MaxPasses: 32,
    },
    MaxConfigBytes: 4 << 20,
})
if plan != nil {
    b.Cleanup(func() {
        if err := plan.Close(); err != nil {
            b.Error(err)
        }
    })
}
if err != nil {
    b.Fatal(err)
}
b.Run("full", func(b *testing.B) {
    cuptibench.Benchmark(b, plan, work)
})
```

`Resolve` expands named presets without loading CUDA. `New` checks the selected
chip's native metric definitions and counter availability. Arbitrary exact CUPTI
metric names are also accepted in `Config.Metrics`.

Register isolated operations as separate leaves, using independently prepared
fixtures or an unmeasured execution prefix. They must work when selected alone:

```text
BenchmarkDecode/full
BenchmarkDecode/isolated/layer00/attention
BenchmarkDecode/isolated/layer00/ffn
```

Set `View: "isolated"` on isolated workloads. `b.Run` selects independent
measurements; it must not carry state from one child to the next. The runnable
physical pipeline fixture demonstrates a dependent full workload and isolated
stages with unmeasured prefix preparation. It is synthetic, not DS4.

`Benchmark` owns `b.Loop()`. Call it once per leaf, outside any other iteration
loop and outside `RunParallel`. It excludes restoration and counter collection
from Go's timed loop. Ordinary `ns/op` describes host time per configured sequence,
including submission and completion. `gpu-ns/op` is CUDA-event time per logical
operation. `profile-ops` is the logical operation count in each profiling pass.
Profiling passes never multiply the denominator.

## Collection modes

| Mode | Result | Profile samples |
|---|---|---|
| `Whole` | One operation range | Operation counters |
| `Regions` | Inclusive root and named region tree | Disjoint terminal regions |
| `Kernels` | Actual kernel range identities | Individual kernel ranges |

In `Kernels` mode, `Benchmark` also collects a separate `Whole` capture for
aggregate columns. It is returned as `Result.Aggregate`, with a distinct capture
identity and completion record. It remains in JSON rather than being added to
kernel samples. `Plan.Collect` performs only the selected mode.

Named regions use stable occurrence identities and bounded nesting. Their trace
must match across replay passes. Identical named regions alone do not prove
identical data-dependent kernel launches or routing; the workload owns that
invariant. GPU work submitted by undeclared streams or unrelated goroutines is
outside the supported contract.

The event timer uses start and completion dependencies across the declared streams.
It does not order those streams against each other. Preparation and its GPU work
finish before native profiling starts.

### Graph kernels and replay mode

Whole and Regions can measure actual graph execution with user replay. For
per-kernel graph attribution, use explicit `ReplayMode: cuptirange.KernelReplay`
in a Kernels plan, and supply the graph executables in `Workload.Graphs`.
Instantiate those executables with `cudagraph.InstantiateOptions{
RetainKernelOrder: true}` before destroying their source graphs. This retains a
bounded snapshot of native kernel symbols and completion dependencies; it does
not rewrite the graph.

The current proof requires a unique kernel order. Independent allocation/free or
other admitted nonkernel nodes need no relative order. It rejects parallel kernel
ambiguity, unsupported node types, device-updatable nodes and invalidated snapshots. Graph
updates and destruction invalidate retained evidence. Raw mutations and concurrent
mutation are prohibited. On GB20B with CUPTI 130202, synthetic forward and
reversed graph controls passed symbol association with unequal CTA counts. This
does not qualify arbitrary graphs or the full DS4 workload. A separate native
allocation/kernel/free graph control also passed exact range-count and output
checks.

CUPTI returns numeric automatic-range names. `Range.NativeName` preserves that
ordinal; `Range.Name` is associated through the observed launch sequence and,
for registered graphs, the retained native order. A decoded range-count mismatch
is an error, regardless of the native dropped counter. Saturated automatic
inventories without an independent completeness proof are rejected.

Kernel replay is distinct from application-controlled user replay. The host
workload runs once while CUPTI performs internal replays; those internal passes
are not directly observed. `Quality.PassesStarted/Stopped` count host API calls.
Results retain `replay_mode`, `configuration_passes`,
`configured_nesting_levels`, and the pass-accounting description. Never divide
counter values by either host calls or an inferred internal pass count.
Kernel replay can change counter-collection scheduling and cache behavior; event
timing remains a separate uninstrumented execution. The extra Whole capture used
for benchmark aggregate columns always uses user replay and has its own metadata.

### Throughput percentages

Raw throughput metrics retain their exact CUPTI names. They are not additive
profile samples and are not translated into memory utilization or remaining
headroom. There is no SoL preset, hierarchy-maximum calculation, or inferred DRAM
percentage. An explicitly requested raw metric remains a native observation,
not a validated interpretation of the hardware's limiting resource.

Comparisons or derived analysis must match normalization bases, such as
`pct_of_peak_sustained_active` versus `pct_of_peak_sustained_elapsed`, as well as
chip, scope, filters, and preparation. A shared percent unit is insufficient.
Do not transfer a throughput rollup formula or L2-to-DRAM calibration across
chips or workload regimes without independent evidence.

## Cache policy

Set `Config.CachePolicy` explicitly for retained measurements. `CacheDefault`
and `CachePreserve` use CUPTI's documented no-reset behavior. This does not freeze
cache contents: preparation, prior passes, and concurrent work can change them.
The result records requested policy, documented effective behavior, and its basis.
Do not transfer Nsight Compute's cache-control default to this direct CUPTI path.

CUPTI 13.4 documents per-pass cache reset controls. The generated binding used
here contains the older enable-parameter prefix; its extended reset ABI has not
been qualified. Requests for `CacheFlushL2` or `CacheFlushAll` therefore fail
before workload execution. No flush measurement or universal cold/warm bound is
claimed. [Native cache-control documentation](https://docs.nvidia.com/cupti/13.4.0/main/main.html#cache-state-reset).

## Reading the numbers

Raw CUPTI columns retain their native names. Their values describe the profiling
sequence; percentages and throughputs returned by CUPTI remain profiled-execution
metrics. They are not silently divided by `b.N` or summed across kernels.

Timing and counters come from separate executions. There is no automatic combined
bandwidth or roofline percentage. L2 sectors are not DRAM bytes, and restored
input bytes do not imply restored cache state. Preserve preparation and cache
policy when comparing experiments.

The JSON result retains native units, range identity, view, operation count,
collection quality, device/version information, and capture provenance. Missing
or incomplete collection is an error. Only verified additive metric definitions
are eligible for integer pprof samples; other finite metrics remain in JSON and
benchmark results where applicable.

## Profile output

Register flags and finish output explicitly:

```go
func TestMain(m *testing.M) {
    cuptibench.Init()
    code := m.Run()
    if err := cuptibench.Finish(); err != nil {
        fmt.Fprintln(os.Stderr, err)
        code = 1
    }
    os.Exit(code)
}
```

```sh
go test ./path/to/package -run '^$' -bench '^BenchmarkDecode$/^full$' \
    -cuptiprofile "$HOME/tmp/decode.pb.gz"
go tool pprof -sample_index=sm__ctas_launched.sum "$HOME/tmp/decode.pb.gz"
```

Close all contributing plans before calling `Finish`, normally through
`b.Cleanup`. `Finish` checks completion without closing caller resources. If a
native cleanup attempt fails, that plan's queued measurements remain invalid
for export even after cleanup succeeds on retry. An unclosed contributing plan
also makes `Finish` fail. In either case no profile or manifest is published.

This requires a benchmark adopting the workload API. The flag cannot add replay
or semantic regions to an arbitrary test binary. Use a fresh destination per run;
existing output is never overwritten. The companion `.json` manifest retains all
results, including inclusive parents and unsupported profile value types.
Regions profiles project only terminal regions. Work outside those leaves may
be omitted, so the profile total is not the whole-operation total. The measured
inclusive root remains in JSON; export does not invent a residual sample by
subtracting leaf values from it. Each Regions capture carries this coverage
limit in the profile comments.
Incompatible metric schemas produce separate profile partitions named in the
manifest instead of inserting unavailable values as zeros. If no selected range
has a verified additive metric, `Finish` returns an error and writes neither a
profile nor a JSON manifest. Ratio-only results remain available in the returned
`Result`; this profile flag requires at least one exportable counter.

For multiple test packages, the runner must supply `{package}` in the
destination. It expands from the verified test-binary build identity. A test
binary cannot determine how many sibling packages the runner launched; literal
paths remain available for single-package commands, with exclusive collision
refusal if another writer claims the same output. Metadata distinguishes benchmark,
view, run, range, and actual kernel identity. A profile containing several
experiments is not one decode pass whose samples should all be summed.

`-cuptimetrics=launches` replaces the plan's metric selection for `Benchmark`.
It does not affect direct `Collect`. Metric presets are validated before workload
execution. Metric override plans retain retryable cleanup ownership.

## Comparing C and Go

The [benchfmt adapter](benchfmt/README.md) converts a documented JSON measurement
format into Go benchmark rows. The `cuptibench-import` command works without a
CUDA library or GPU. It preserves one statistical sample per source record and
rejects implicit normalization or timing-kind changes.

```sh
go run ./cmd/cuptibench-import c-samples.json > c.txt
benchstat c.txt go.txt
```

Use the same benchmark names and matching measurement scopes in both files. A C
host duration cannot be relabeled as Go device-event time. A future oracle or
Nsight reader can feed the same normalized format once a retained source format
has been selected.

## Verification

From the `nvidia` module:

```sh
GOWORK=off GOFLAGS=-mod=readonly go test -race \
    ./x/cuptibench/... ./x/cuptirange ./x/cudaprof ./x/cuda \
    ./internal/cuptiowner ./cmd/cuptibench-import
```

On an admitted Linux GPU host with an exclusive device lease and profiling
permissions, inspect the opt-in controls in `physical_linux_test.go` before
running them. They test exact CTA counts, workload changes, reset contamination,
isolated preparation, range capacity, stream timing, and profile round trips.
The tests preserve failed receipts and do not establish support on another host.

Native cleanup failures invalidate the measurement and retain ownership for retry.
Always check `Close` and `Finish` errors. Hardware tests skipped on this host are
not successful hardware qualification.

## Portability

CUPTI layout admission is based on observed header and runtime evidence, not
only the CUDA toolkit release name. CUPTI 27 is admitted using the CUDA 12.9.41
L4 evidence described in [QUALIFICATION.md](QUALIFICATION.md). That admission
does not qualify KernelReplay, which remains restricted to 130200 and 130202.
The reported L4 run passed Whole and Regions controls; Kernels returned zero
ranges and remains unqualified on that configuration.

The physical test fixtures compile CUDA source through `nvrtcx`. Its generated
Linux NVRTC loader currently selects `libnvrtc.so.13`, so these fixtures need
that library and a compiler whose emitted PTX the installed driver accepts.
`cuptibench` itself does not import NVRTC or compile kernels; a workload using
precompiled kernels does not acquire this compiler dependency from the package.

## Design and qualification

See the [design contract](design/cuptibench-spec-final.md),
[C adapter alternatives](design/c-workload-adapter-spec.md), and
[qualification status](QUALIFICATION.md). Hardware receipts describe the
recorded source snapshots, not a new hardware run of the main integration.
