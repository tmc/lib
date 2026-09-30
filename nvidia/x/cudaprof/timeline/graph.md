# Graph provenance

`Correlations` accepts a graph replay only when the activity's exact nonzero
correlation identifies a unique retained `cuGraphLaunch` or `cudaGraphLaunch`
API record in the requested domain and process generation. A construction API
is reported as `graph_construction`, without replay range context. Missing
calls and duplicate identities remain missing and ambiguous. Neither graph
membership nor a containing envelope supplies a missing launch identity.

The native collector retains graph membership for CUDA 12 Kernel9 and CUDA 13
Kernel10. The kernel correlation is the single driver-or-runtime launch ID,
not a separately retained runtime ID. Native node IDs remain uint64 throughout
journal decoding. See NVIDIA's [activity API reference](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html).

`GraphLifecycle` consumes explicit native handle events. Collection is opt-in
with `CUDAPROF_GRAPH_LIFECYCLE=1`; it enables driver and graph resource
callbacks and records `graph_lifecycle=collected` with coverage
`driver_partial`. Defaults and older journals have an empty
`Records.GraphLifecycle`. A requested unavailable callback domain is recorded
as a failure; collection state is not a completeness gate. Handle identity is distinct from CUPTI membership
identity. Source graph destruction does not destroy an executable's snapshot;
executable destruction followed by reuse starts a new generation. An update
with absent status or source poisons subsequent replay qualification. Child
relationships remain unsupported. Unhandled successful graph APIs are retained
and invalidate known lifecycle state in that journal. Read-only query, debug
and upload observations are retained without supplying identity. Resource
callbacks preserve raw graph/executable/node and clone-source handles; they
do not invent a replay link. Complete mutation coverage is required by
the caller; an omitted mutation cannot be detected from these events alone.

For a marked native control, compile `x/cmd/cudaprof/testdata/spin.cu`:

```sh
nvcc -O2 -arch=sm_121 x/cmd/cudaprof/testdata/spin.cu -ldl -o ~/tmp/spin
CUDAPROF_SPIN_MARKERS=1 ~/tmp/spin graph 100000 3 5
```

The opt-in path refuses a missing NVTX provider. Run this only inside the
established GPU hold, with capture `--graph-trace=node --nvtx --memsets=false`.
The CUDA workload prints independent expected node and submission counts;
three nodes replayed five times require 15 measured kernel records and five
replay operation ranges. The separate graph-construction range precedes the
profiler range. Disable markers by omitting the environment variable, and
collect ordinary timing controls before estimating instrumentation overhead.
The same executable can be launched with Go `exec.Command` or Python
`subprocess.run`; its ranges belong to the C++ submitting thread. This control
establishes collector integration, not equivalence of Go and Python engines
or an explanation for GEMV duration.


Native collection parses CUDA driver `cuGraphCreate`, stream capture end
(including ptsz), instantiate (original, v2, flags, params and ptsz params),
launch (including ptsz), executable/graph destruction, kernel-node addition,
executable update (original/v2 with explicit update result), and executable
kernel-node parameter updates (original/v2). Successful explicit node updates
advance the executable version and make its source graph snapshot unknown.
Other mutations retain unsupported observations and invalidate known state.
Runtime-domain lifecycle callbacks and recursive child graph reconstruction
are unsupported. Driver-only observations cannot qualify mutations that bypass
this callback domain.

The producer compiled against real CUDA 13.4 headers. CPU stub controls cover
CUDA 12/13 layouts, opt-in/refusal behavior, executable handle reuse and explicit
updates. Real CUDA 12.6 compilation remains blocked by the pre-existing
`CUpti_ActivityMemcpy6` dependency; the stub controls do not establish a shipping
CUDA 12 build. No lifecycle collection overhead estimate follows from these
CPU tests. Compare marker-enabled ordinary runs with profiled lifecycle-off and
lifecycle-on runs under the same hold before quoting overhead.

`CUDAPROF_SPIN_STABLE_OUTPUT=1` checks device-record cycle/timer invariants and
emits deterministic workload parameters and independent expected counts. Use
this for campaign output equality; the default output includes measured timers
and wall time and cannot serve as an equality reference.
