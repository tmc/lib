# cuptibench: final design specification

Status: design contract updated 2026-09-15. Implementation exists in isolated
worktrees; synthetic GB20B controls and real DS4 Regions benchmarks/profiles pass.
Real DS4 per-kernel capture fails at native Stop and remains unqualified; an
earlier non-finite-logit failure remains unexplained despite a passing rerun.
Package: `github.com/tmc/lib/nvidia/x/cuptibench`.

This specification supersedes the three drafts named `cuptibench-spec.md`,
`cuptibench-spec-D304CF8D.md`, and `cuptibench-spec-DF6906F6.md`. It incorporates
the requested DS4 decode-pass and kernel sub-benchmark workflow. API declarations
below describe the implementation contract; qualification is tracked separately.

## 1. Purpose

Integrate CUPTI counter collection with Go tests, benchmarks, and profiles.
A caller must be able to measure a complete decode pass, inspect work within that
pass, and benchmark selected operations independently using the same workload
implementation.

Existing `cuptirange` code already configures, collects, and evaluates hardware
metrics. This package adds workload preparation, measurement boundaries,
benchmark reporting, and profile ownership. It does not implement another CUPTI
collector.

Counter acquisition remains separate from CUDA-event timing. The sibling
`cudabench` measures event duration and derives rates from caller-declared work.
`cuptibench` obtains requested counters from CUPTI. It has no `SetBytes` API.
Caller-declared algorithmic work may be an input to a separate analysis, but
must never be represented as a hardware measurement.

## 2. Three measurement views

### Full pass

Execute one real forward decode operation, with the application's actual kernel
sequence, dependencies, and stream scheduling. Measure its event duration in an
uninstrumented execution and its whole-operation counters in separate profiling
executions. One logical operation is one invocation of the workload's Run.

### Attributed full pass

Execute that same complete operation with named regions. Report per-region or
per-kernel counters from within the operation. Preserve the distinction between
an algorithmic region and a CUDA kernel launch: one attention region may contain
several launches; one fused launch may implement several algorithmic operations.

Region and kernel attribution are separate collection modes. If the backend
requires separate captures for these views, replay the workload explicitly and
record separate capture identities. Do not present them as simultaneous data.

### Isolated operation

Prepare the input state for a selected operation, then measure that operation
alone. Preparation may restore a retained fixture or execute an unmeasured
prefix of the full pass. The result records which preparation method was used.
An isolated operation can contain multiple kernels. A benchmark claiming to
measure a single kernel must verify that exactly one kernel range was captured.

Isolated results and attributed full-pass results must have distinct names and
metadata. Restoring bytes does not restore cache state. Their values must not be
presented as interchangeable.

## 3. Workload model and Go sub-benchmarks

`testing.B.Run` selects independent measurements. It is not an execution
scheduler for dependent stages. A selected child must work even if none of its
siblings ran, and each child must tolerate independent repetition.

The application owns a reusable decode workload and its operation hierarchy.
Keep DS4-specific dependency graphs, tensors, KV layouts, routing, and fixture
management out of cuptibench. The package supplies named region instrumentation
and replay boundaries rather than a generic graph scheduler.

A representative benchmark tree is:

    BenchmarkDecode/full
    BenchmarkDecode/isolated/layer00/attention/q_proj
    BenchmarkDecode/isolated/layer00/attention/kv_update
    BenchmarkDecode/isolated/layer00/attention/attention
    BenchmarkDecode/isolated/layer00/attention/out_proj
    BenchmarkDecode/isolated/layer00/ffn/gate
    BenchmarkDecode/isolated/layer00/ffn/up
    BenchmarkDecode/isolated/layer00/ffn/down

These names illustrate an application-defined decomposition, not a verified DS4
kernel inventory. The application must expose actual fused operations and
conditional work faithfully.

`full` produces a whole-pass result and the selected attributed profile. It is a
measured leaf benchmark, not a parent whose printed time is assumed to equal its
children. The other leaves execute isolated workloads.

Selection uses ordinary Go benchmark patterns, with the package argument before
custom flags:

    go test ./path/to/package -run '^$' -bench '^BenchmarkDecode$/^full$'
    go test ./path/to/package -run '^$' -bench '^BenchmarkDecode$/^isolated$/^layer00$/^attention$/^q_proj$'
    go test ./path/to/package -run '^$' -bench '^BenchmarkDecode$'

The registration shape is ordinary Go. In this illustrative sketch, `decode`
is application-owned; Full and Isolated construct Workload values around the
same forward implementation and manage their own fixtures/resources:

    func BenchmarkDecode(b *testing.B) {
        // Construct device, decode fixtures, and an explicit profiling plan.
        // Register cleanup for each owner, including errors from plan.Close.
        b.Run("full", func(b *testing.B) {
            cuptibench.Benchmark(b, plan, decode.Full())
        })
        b.Run("isolated", func(b *testing.B) {
            for _, stage := range decode.Stages() {
                b.Run(stage.Path, func(b *testing.B) {
                    cuptibench.Benchmark(b, plan, decode.Isolated(stage))
                })
            }
        })
    }

`Stages` is an ordered inventory, not a map traversal or an execution of the
forward pass. Workload execution remains in Full/Isolated. Fixture setup must
not depend on whether `full` was selected. Hierarchical path components can also
be registered through nested b.Run calls when parent-level setup is useful.

Use the full `b.Name()` as the benchmark identity. Go disambiguates duplicate
sibling names; nevertheless, application stage names should be stable and
meaningful. Repeated benchmark runs require a separate run identifier.

## 4. API

The collection API is independent of testing. The benchmark adapter adds timing,
Go iteration accounting, and ReportMetric calls.

    type Mode int

    const (
        Whole Mode = iota // one user range enclosing the logical operation
        Regions           // root plus caller-defined nested regions
        Kernels           // automatic kernel ranges for the full workload
    )

    type Config struct {
        Metrics        []string
        Mode           Mode
        Operations     int              // logical operations per profiling pass
        Limits         cuptirange.Limits
        MaxConfigBytes int
        MaxDepth       uint16 // Regions nesting including root; zero defaults to 16
        CachePolicy    cuptirange.CachePolicy
        ReplayMode     cuptirange.ReplayMode // UserReplay default
    }

    type Workload struct {
        Graphs      []*cudagraph.Exec // retained native kernel order, when needed
        View        string // full, isolated, or empty (unspecified)
        Streams     []*cuda.Stream // complete set of participating streams
        Prepare     func() error
        Run         func(*Scope) error
        Synchronize func() error
        Preparation string // fixture/prefix/state and cache policy description
    }

    type Scope struct { /* unexported */ }

    func (s *Scope) Region(name string, run func() error) error

    type Plan struct { /* unexported */ }

    func New(dev *cuda.Device, config Config) (*Plan, error)
    func (p *Plan) Close() error
    func (p *Plan) Collect(w Workload) (*Result, error)
    func Benchmark(b *testing.B, p *Plan, w Workload) *Result

`New` rejects invalid modes, empty/duplicate metric names, nonpositive operation
counts, invalid limits, and unsupported configurations before invoking workload
code. Config is copied. The device must remain open until Plan.Close succeeds.
A nonnil plan returned with an error must be closed to retry cleanup.

All plans require Close, including after successful collection. A plan is
reusable sequentially across sub-benchmarks. Concurrent use fails rather than
silently queueing measured work. Plan.Close must not race an active operation.
The implementation must reject that conflict and retain ownership.

A Scope is valid only during Run, on its calling goroutine. Region executes its
callback immediately. It is a no-op wrapper for uninstrumented timing and kernel
mode, and a balanced push/pop in region mode. Region names are nonempty path
components, with no slash; nested calls build the path. Repeated occurrences
receive explicit occurrence identities. Whole mode ignores nested annotations.
Region balance must survive errors and panic unwinding; panic propagation is
preserved after cleanup.

Run and Synchronize are required. Prepare may be nil only when the caller's
workload can repeat without state restoration; Preparation must document that
choice. Run, Prepare, and Synchronize return errors instead of calling Fatal or
Skip. The benchmark adapter decides how to fail on the calling test goroutine.

The package executes callbacks on a locked OS thread with the plan's retained
CUDA context current. Callers create compatible resources on that context and
must not replace the current context. Work on additional streams must join the
completion boundary. Version one supports one device and context per plan.
Multi-device decode needs separately scoped plans and is outside this contract.

Closures use the application's real streams, listed explicitly in Streams. The
list must be nonempty, unique, and fixed for one measurement. Undeclared streams
and work submitted by unrelated host goroutines are outside the supported
contract. An internal timing stream may coordinate start/end events, but must
not become the stream on which the application workload executes.

### Native kernel identity and graph replay

CUPTI automatic range names are numeric ordinals. Preserve them as NativeName.
Associate symbols through an independently observed same-context launch sequence;
require decoded range count to equal the observed launch count. A named prefix
is never acceptable. Zero reported drops cannot establish completeness. Enforce
this contract in cuptirange, including its legacy Capture entry point.

For per-kernel graph attribution, require explicit KernelReplay and live graph
executables instantiated with RetainKernelOrder. Retain a bounded snapshot of
native symbols and completion dependencies. Require a unique kernel order; independent admitted nonkernel nodes need no
relative order. Allocation/free nodes retain their dependency edges. Reject
parallel kernel ambiguity, unsupported nodes and invalidated evidence. Graph
updates and destruction invalidate snapshots. Raw or concurrent mutation is
outside the supported contract. Ordinary graph instantiation remains unchanged.

KernelReplay is version-gated to the qualified CUPTI layouts (130200 and130202),
Kernels mode and depth1. Run the host workload once and require native completion;
do not repeat Run to simulate internal kernel replay. PassesStarted/Stopped count
host start/stop calls, not unobserved native internal passes. Retain replay_mode,
configuration_passes, configured_nesting_levels and explicit accounting metadata.
Regions nesting can require additional host passes even for a one-pass metric
configuration. Never divide counters by these pass counts.

The separate Whole capture used for aggregate benchmark columns uses UserReplay,
with a distinct identity and provenance. Whole/Regions support graph execution
without a retained per-kernel identity snapshot.

### Results

    type Metric struct {
        Name       string
        Unit       string
        Value      float64
        Additive   bool // verified counter semantics, not inferred from spelling
    }

    type Range struct {
        ID         string
        ParentID   string
        Name       string
        NativeName string // native automatic range ordinal, not kernel symbol
        Kind       string // operation, region, or kernel
        Inclusive  bool
        Metrics    []Metric
    }

    type Timing struct {
        Operations int64
        Total      time.Duration
        Method     string
    }

    type Result struct {
        Aggregate  *Result // separate Whole capture for Benchmark in Kernels mode
        Operations int
        Ranges     []Range
        Timing     *Timing // nil for counter-only Collect
        Quality    cuptirange.CollectionQuality
        Metadata   map[string]string
    }

Metric order follows the requested configuration in Result. Benchmark output
column ordering belongs to testing, not this package.

Metadata has documented reserved keys for device UUID, chip, CUDA/CUPTI versions,
resolved metric names, measurement mode, preparation policy, capture identity,
benchmark identity/run identity when present, and profile conversion policies.
Metric units come from a verified mapping/native definition. Unknown units remain
explicitly unknown in Result and are not guessed for conversion or export.

Kernel range records retain actual kernel identity when available. A region path
must not be substituted into a field claiming to identify the CUDA kernel.

Collect returns diagnostic quality alongside an error where possible. No values
from a failed capture are admissible measurements. Cleanup uses the underlying
retryable ownership contract; failure leaves the plan responsible for cleanup.
A subsequent collection cannot start until pending cleanup succeeds.

Returning Result earns its place because ordinary TestXxx functions can call
Collect, check errors, and assert on counters, ranges, or collection quality.

## 5. Replay and state boundaries

The required profiling sequence for every pass is:

1. Drain previously submitted work on the participating context/streams.
2. Prepare the workload state and synchronize preparation.
3. Start counter collection and the selected ranges.
4. Execute the configured number of logical operations.
5. Join all participating streams and finish the collected work.
6. Close ranges, stop collection, and inspect native pass completion.
7. Repeat from preparation when CUPTI requests another pass.
8. Decode and evaluate only after complete, loss-free collection.

For Operations greater than one, Prepare restores the start of the entire
sequence once per pass. Consecutive Run calls may advance state as the real
application does. All passes must replay the same sequence, including conditional
kernel launches and routing decisions. A one-token experiment uses Operations=1;
a sequence of evolving decode tokens must label its token positions and workload
semantics explicitly.

Timing repetitions restore the same start state before each sequence. Thus a
sequence benchmark measures the same workload distribution as counter replay.
Normalize both timing and counters by logical operations, never profiling passes.

The current cuptirange callback runs after collection starts. Its lifecycle must
be extended so Prepare and its GPU work finish before collection starts. The
new wrapper must not disguise preparation kernels as measured workload kernels.
User ranges and range identity retrieval also belong in cuptirange.

No equality of restored inputs implies equality of cache state, clocks, or
interference. Record preparation, warmup, and cache policy. Do not change clocks,
permissions, or device configuration implicitly.

## 6. Benchmark execution and timing

Benchmark owns its `b.Loop()` loop. Call it once in a leaf benchmark; callers must
not put it inside another b.Loop, a b.N loop, or RunParallel.

One Go benchmark iteration executes one sequence of Config.Operations logical
operations. Before each iteration, stop the host timer, restore and synchronize
state, then start timing. Run the actual workload sequence and join completion.
Call the next b.Loop with the timer running; it stops timing when iteration ends.
Do not leave the timer stopped at that boundary. Setup, restoration, profiling,
decoding, and export are excluded
from the ordinary timed loop. Document the resulting ns/op as host duration per
sequence, including submission and completion waits.

An uninstrumented CUDA-event measurement encloses the complete device sequence.
For multiple streams, record a start event on an internal timing stream and
make each declared stream wait for it. Record completion events after work on
each declared stream, then have the timing stream wait for those events before
recording the end event. Event placement must neither
omit work nor introduce dependencies between otherwise independent kernels.
This timing support belongs in a reusable CUDA timing helper, not a second copy
of collector code. Unsupported stream topologies return an error.

After b.Loop ends, perform the selected bounded counter capture once for that
benchmark invocation. In Kernels mode, also perform a separate Whole capture for
aggregate benchmark columns. Store it in Result.Aggregate with its own capture
identity and completion quality, linked to the same benchmark run and timing
identity. Never sum native per-kernel ratios to manufacture a Whole result.
Collect performs only its selected mode. Capture counts are independent of b.N.
The Result retains timing and profiling operation counts separately.

Report `gpu-ns/op` as total event duration divided by timed logical operations.
Do not derive a rate when the device clock does not resolve a positive duration.
The fixed profiling sample size and the timed iteration count are both retained.

This design does not ignore Go's calibration. Counter-only tests use Collect;
there is no separate counter-only benchmark mode that silently ignores b.N.
The added cost is one bounded profiling capture after each timed leaf invocation,
or two in Kernels mode. Each capture may require multiple replay passes.

Benchmark returns the Result after successful reporting. It calls b.Helper and
fails on collection, workload, or cleanup errors. Optional environment skipping
is explicit caller policy before invoking Benchmark. Requested metrics that are
unavailable, invalid, or incomplete must not become zero-valued output.

## 7. Reporting and aggregation

Report raw whole-operation metric values under their exact CUPTI names. These
are values for a sequence of Result.Operations, whose count is retained and
reported as `profile-ops`. The exact metric name is the ReportMetric unit token;
this does not imply it is a physical unit. Throughput and percentage values
returned by CUPTI are permitted and remain profiled-execution metrics.

Do not relabel a raw SoL metric as memory utilization, remaining headroom, or a
cache-hierarchy maximum without qualification for the exact chip and regime.
There is no automatic SoL preset or interpretation in this implementation.
Any derived throughput comparison must establish matching normalization bases
(active versus elapsed), scope, filters, and preparation, not merely percent units.

Additional normalized columns require verified metric semantics. Additive
counters may divide by logical operations. Ratios, percentages, throughput, and
per-instance averages must not be summed or divided by operation count merely
because their names look familiar. A .sum suffix alone is not a conversion proof.

Whole-pass event duration is measured directly. Never report the sum of isolated
kernel durations as decode latency. Likewise, isolated counter values are not an
in-context whole-pass measurement.

Nested user ranges are inclusive. Results retain the range tree and its scope.
Do not sum a parent and its descendants. Summation over disjoint leaves is valid
only for compatible additive metrics with stated coverage. Unattributed work
stays visible; it is not silently distributed among named children.

There is no default composite `GB/s` or roofline percentage. An optional analysis
consumer may divide explicitly identified counter-derived traffic by separately
measured time, provided it records both capture identities and the comparability
assumptions. This is a derived cross-execution estimate, not a rate measured in
one execution.

## 8. Metric sets, traffic, and roofline analysis

Named sets resolve to explicit metric names for the observed chip and supported
backend configuration. Expose discovery and resolution through cuptirange where
possible. Unknown set names, unresolved sets, and unavailable requested metrics
are errors. Persist the resolved names; set names alone do not establish
cross-chip equivalence.

L2 read-miss sectors converted using a verified sector size remain L2 read-miss
bytes. In particular, a TEX-source read-miss metric is not total DRAM traffic.
Do not store it in DRAMReadBytes or silently broaden its meaning in a generic
profile slot. Preserve source-unit and operation filters in metadata.

The drafts attribute GB10 DRAM-counter absence, a 24 MiB L2, and historical
232-239 GB/s throughput to earlier work. They are not independently established
by this review and are not universal package invariants. Query current hardware
and metric availability; qualify conversion assumptions with retained evidence.

Residency is an explicit input to optional roofline analysis, not a required
setter for raw counter collection. A string parameter cannot establish valid
residency at compile time. Unknown or mixed residency must prevent analysis
against an incompatible single-memory-path ceiling, while leaving raw data usable.

The analysis must identify the ceiling, units, memory path, provenance, and
qualification regime. An unverified ceiling may be retained as a labeled model
assumption; it must not produce a verified efficiency percentage or an unqualified
bottleneck classification. Algorithmic FLOPs remain caller-declared model input.

### Cache control implementation boundary

Config.CachePolicy is independent of Workload.Preparation. Default and Preserve
use CUPTI's documented no-reset behavior on admitted versions; they do not promise
identical cache contents. Retain requested/effective policy and the evidence basis.
FlushL2 and FlushAll currently return an error before workload execution because
the extended native reset ABI is not qualified by the installed binding. This is
a binding limitation, not a claim that CUPTI universally lacks cache control.
Flushed and unflushed observations are sensitivity comparisons, not universal
mathematical bounds. Hardware tests must qualify the selected preparation regime.

## 9. Profiles and flags

Use cudaprof as the pprof sink. Profile writing does not belong in cuptirange.

Proposed process helper:

    func Init()
    func Finish() error

    func TestMain(m *testing.M) {
        cuptibench.Init()
        code := m.Run()
        if err := cuptibench.Finish(); err != nil {
            fmt.Fprintln(os.Stderr, err)
            code = 1
        }
        os.Exit(code)
    }

Init registers flags explicitly and idempotently. It has no import-time flag
side effects. A conflicting registration must produce a clear diagnostic rather
than take ownership of another package's flag. Finish finalizes export and
releases process-owned writer resources; do not rely on deferred cleanup after
os.Exit. Measurement plans still have explicit Close ownership. Finish requires all plans
associated with queued results to have closed successfully. A native Close
failure permanently invalidates that plan's queued results even if retry later
releases its resources; Finish never closes caller-owned plans itself.

    -cuptiprofile <path>    write a process profile plus result metadata
    -cuptimetrics <sets>    override benchmark metric selection with named sets

Absent a metric override, the plan's explicit metrics apply. A supplied override
replaces that selection, is resolved before workload execution, and is recorded.
It applies consistently to timing-associated counter results and profile output;
it must not silently create a different profile-only metric set. Benchmark builds
and owns a temporary resolved plan when an override differs from the supplied
plan, preserving the supplied limits and mode; it never mutates the caller's
plan. Collect used by ordinary tests does not consult process benchmark flags.

Absent -cuptiprofile, Benchmark still collects and reports metrics. The flag
cannot retrofit replay or stage boundaries into an arbitrary benchmark that has
not adopted the workload contract.

The output is one process profile, accompanied by `<path>.json` containing the
structured results and metadata. Write through sibling temporary files and
rename on successful completion. Refuse existing destination files by default.
For multi-package commands, require a `{package}` placeholder in the output path
and expand it to a collision-free encoding of the test package import path.
Resolve package identity from verified build information; fail if unavailable.
The invoking runner must supply this placeholder for multi-package commands;
a test binary cannot infer how many sibling packages the runner selected.
Cross-process exclusive creation prevents two writers from claiming one path.
Do not silently overwrite or append to an existing profile.

All sub-benchmark invocations contribute with their full benchmark name, view,
run identity, operation count, region path, kernel identity when known, and
capture identity. Metadata that varies by sample is attached per sample/result,
not only in a process-wide comment. Keep benchmark name separate from kernel name.

### Inclusive totals and attributed samples

A profile contains one non-overlapping measurement level per capture. In Regions
mode export disjoint terminal regions; in Kernels mode export individual kernel
ranges; in Whole mode export the operation range. Inclusive ancestors remain in
the JSON results, not additional additive samples. Record excluded/unattributed
coverage explicitly. Scope-labelled samples from different views remain distinct;
do not imply their global sum is one decode pass.

Select the default sample type explicitly. A multi-view profile is a collection
of experiments, and comparison must select the desired view and workload regime.

### Numeric representation

pprof sample values are int64, while evaluated CUPTI values are float64.
Prefer additive counters in pprof. Preserve the evaluated float64 values in the
result sidecar. Integer encoding requires finite values, explicit scale and
rounding rules, overflow checks, and documented precision limitations above
2^53. Reject conversions that cannot meet the declared policy.

Fixed-point encoding does not make ratios additive. Do not export percentages
or rates as additive samples merely by scaling them. Preserve these in structured
results; derive ratios from correctly scoped numerator/denominator samples when
their native definitions permit it.

Reuse builtin cudaprof sample types only when their semantics match. If dynamic
sample types are needed, add an explicit declaration with name and unit, reject
duplicates/collisions, freeze the schema before recording, and make recording
return errors. Missing values must not masquerade as measured zeros. Partition
incompatible sample schemas into separate profile outputs recorded by the JSON
manifest rather than filling unavailable columns with zero.

## 10. Ownership, resource bounds, and support

Coordinate known incompatible collectors through a shared internal ownership
mechanism, backed by native errors for external conflicts. A cuptibench-local
mutex cannot establish ownership against another package or process. Device
isolation across test binaries remains an explicit runner responsibility.

Reject concurrent use of a plan and unsupported concurrent collectors. Ordinary
sequential sub-benchmarks reuse an explicitly owned plan. Do not hold a profiling
lease while registering/running child benchmarks or executing arbitrary parent
setup. No automatic global plan cache is required.

If caching is added later, the key includes device/counter availability,
CUPTI/backend identity, ordered metrics, configuration mode, and relevant limits.
Chip plus set name is insufficient. Cached native resources require explicit
shutdown ownership.

Preserve MaxPasses, range/tree capacity, configuration bytes, and counter bytes
bounds. Nested region depth and operation counts must fit the selected backend.
Reject excess limits before execution where predictable; native completion and
loss checks remain authoritative afterward.

CUPTI replay does not universally serialize kernels. State the behavior of the
selected profiling mode, and qualify preservation of concurrency where needed.
Supported versions and ABI layouts are those admitted by the underlying binding;
cuptibench must not weaken those checks or duplicate a stale version allowlist.

No library installation, privilege escalation, clock tuning, or persistent
configuration changes occur implicitly. Cleanup failures retain explicit ownership
for retry and invalidate the measurement; zero surviving side effects cannot be
promised after a native cleanup failure.

## 11. Implementation order and acceptance

1. Extend cuptirange with preparation boundaries, user ranges, nested range
   identity, and bounded lifecycle control. Test callback ordering, failed
   preparation, incomplete passes, dropped ranges, overflow, and cleanup retry.
2. Implement testing-independent Plan/Collect and structured Result. Test that
   replay pass count never changes logical-operation normalization.
3. Add the Benchmark adapter and event timing for supported stream topologies.
   Exercise b.Loop, nested b.Run, filtered children, repeat counts, errors, and
   cleanup through real test binaries. Verify one selected post-loop collection
   per leaf invocation, the extra Whole capture in Kernels mode, and correct host
   timer exclusions.
4. Integrate a DS4-owned workload hierarchy using actual forward execution.
   Validate full-pass output, retained intermediate fixtures, KV restoration,
   isolated-stage output, and stable replay traces before interpreting counters.
5. Add process profile export, JSON results, and optional metric-set overrides.
   Round-trip profiles and test collisions, missing metrics, conversion errors,
   multiple runs, filtering, and no parent/child double counting.
6. Run physical qualification for admitted hardware/backend combinations. Retain
   hardware/software identity, workload outputs, requested/resolved metrics,
   losses, replay completion, and measurement boundaries.

Physical controls must independently check counter accounting: a known workload,
a changed workload with a predicted counter response, a deliberately wrong
normalization, reset-work contamination, and incomplete/lost capture rejection.
Use cold and warm working sets with a stated preparation regime to test traffic
interpretation. A historical bandwidth band is a separate performance comparison,
not the correctness oracle for the counter pipeline.

For multi-stream/full-pass support, test dependent and overlapping kernels.
Verify whole-pass timing against an independent completion measurement within a
predeclared tolerance. Demonstrate that the profile preserves overlap semantics
and does not count replay or parent ranges as additional logical work.

Hardware-dependent tests use explicit opt-in and report unavailable environments
as unqualified. Mock/unit success does not establish physical counter accuracy.
No performance or support claim is made by this specification itself.

## 12. Source basis and boundaries

Current-source review used:

- Main: nvidia/x/cuptirange/{capture,collect,replay,evaluate,properties}.go.
- Main: nvidia/x/cuptiactivity/{doc,session}.go.
- Main: nvidia/x/cudaprof/{proto,profile}.go.
- Main: generated RangeProfilerPushRange/PopRange bindings in nvidia/cupti.
- Read-only sibling worktree ~/tmp/ds4-lib-workflow/worktree:
  nvidia/x/cudabench/bench.go, nvidia/x/cudacensus/census.go, and
  nvidia/x/cudaroofline/roofline.go.

The sibling worktree has APIs absent from the reviewed main tree. Integration
must reconcile those dependencies without assuming branch-local code is merged.
Neither tree was edited or executed for this design review.

Authoritative API references:

- Go testing: https://pkg.go.dev/testing
- Go sub-benchmarks: https://go.dev/blog/subtests
- Go B.Loop: https://go.dev/blog/testing-b-loop
- CUPTI usage and range profiling: https://docs.nvidia.com/cupti/main/main.html

The original drafts' host measurements remain attributed historical evidence.
Implementation must verify native layouts, available counters, metric semantics,
and physical behavior on the selected execution host.
