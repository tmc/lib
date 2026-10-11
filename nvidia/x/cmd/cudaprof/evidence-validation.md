# Controlled evidence validation

The 2026-09-30 follow-up used an NVIDIA GB10 on Linux arm64, driver
580.173.02 and CUPTI 13.4. CPU tests covered the command package, all
cudaprof analysis packages, Perfetto, activity export and native producer
controls. The native producer compiled against CUDA 13.4. CUDA 12 stub
controls do not establish a working CUDA 12.6 production build: the
existing Memcpy6 dependency prevents that build.

## Native controls

GPU work ran under the shared `gpu-lock`. Independent expectations were
written before capture: ten measured spin kernels and ten submission ranges
for ten stream launches or one graph node replayed ten times. The workload
checks its device result invariants. Collection retained every expected
kernel. Runtime-domain submission and explicit operation metadata matched
all ten kernels. Driver-domain identity was not substituted when absent.

The separate delivery variant synchronizes, copies each result to the host,
checks it, and records explicit materialization scopes and delivered events.
Both stream and graph variants retained ten kernels, ten materialization
scopes, ten delivered events and one token-limit event. Graph kernels also
matched the observed executable lifecycle through exact runtime replay
correlations. This variant serializes the workload; it is not an overhead
control for asynchronous execution.

Graph collection remains opt-in and partial. Unknown mutations, child graphs,
missing events and unsupported callback paths cannot become qualified
lifecycle provenance. A replay link identifies submission context, not a
cause of kernel duration. Delivery markers do not complete dependency coverage.

## Capability probes

The managed allocation control returned the expected value. That result and
managed/pageable-access device attributes do not observe page residency.
Separate native UM counter configuration probes returned result 28,
`CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED`, for both transfer directions,
CPU faults and GPU faults. Migration and fault attribution remain unavailable
on this backend.

The controlled `sm__cycles_elapsed.avg` probe queried its native definition,
completed one user replay pass, decoded without dropped ranges, and checked
the expected result before, during and after collection. Its receipt retains
chip, driver, CUPTI and tool identity. This diagnoses counter collection for
the control kernel; it is not matched-GEMV counter collection. Counter
profiling overhead remains unmeasured. `counter-receipt select` separately
validated an exact historical launch pair from the qualified wrapper shapes.

## Repeated wrapper campaign

A Go subprocess wrapper and a Python subprocess wrapper ran the same native
asynchronous workload. The Go wrapper needed external linking so injection
could observe the parent handshake. The static-wrapper campaign failed that
gate and remains excluded. A second campaign used an incorrect deterministic
stdout expectation and excluded every block. Neither produced an interval.

A fresh campaign with the source-derived expected stdout qualified three
ABBA blocks. Every arm had profiled and ordinary controls, independent marker
expectations, output equality, artifact stamps, complete journals, steady
iteration gates and explicit runtime-domain shape matches. The bootstrap
resamples whole blocks, conditional on unverified exchangeability. Its effect
measures steady device-window elapsed duration, not host delivery latency,
individual kernel samples, an inference-engine comparison or a slowdown cause.

## Existing captures and presentations

The existing Gemma4 captures retain their original failed stationarity and
shape gates. Their standalone comparison summary contains no equivalent-work
numerical rows; unmatched work stays visible. The token overview has no
observed application tokens or delivered events. Its optional arg_reduce
intervals are explicitly inferred and keep crossing and unmatched work.

Captures, JSON receipts, standalone HTML, Perfetto traces, import queries
and failed runs are machine-local artifacts, not portable committed fixtures. The CPU fixtures and examples exercise
qualified and refused cases without a GPU.

Perfetto v58.2 imported the controlled raw-plus-summary trace with ten
observed delivery instants, ten host materialization scopes, nine separately
inferred intervals and bounded NVML query receipts. Raw record sequence and
journal annotations support source lookup. The dependency trace remained
partial with no fabricated flows; a synthetic explicit-edge fixture imported
one flow. The large existing Gemma trace exposed reference duplication and
same-name sibling track merging. References are now emitted once per group,
and explicit track UUIDs remain distinct. The corrected trace imported without
positive error statistics, kept all summary evidence events and retained 32
inferred intervals with zero observed delivery events.
