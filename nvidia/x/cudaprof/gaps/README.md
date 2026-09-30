# Gap evidence

`cudaprof gaps --json capture.pprof` reports the union of recorded device
activity across every stream on each device. Kernel, memcpy, and memset
records contribute occupancy. Work on a second stream fills a gap in the
first stream. The window starts at the earliest device activity and ends at
the latest activity end; time outside those bounds is not measured.

The collector already supports memset and synchronization records. Memset
collection is opt-in (`--memsets`); its capability receipt must be checked
before interpreting a capture that contains no memset records. Synchronization
records contain a context rather than a device. The decoder resolves the
context through device activity, and reports unresolved synchronization
counts and durations separately when no device record names it. CUPTI
overhead and graph replay envelopes are also decoded and exported.

Synchronization, overhead, and non-opaque graph envelopes overlap gaps as
context. They do not contribute ordinary device occupancy and do not prove
why work was absent. Opaque graph envelopes count as occupancy when no graph
node records were retained; internal gaps then remain unobservable.

The JSON `scope` and `window` fields state the bounds. `unknown_cause_ns`
retains every gap as causally unexplained even when an API timing class or
synchronization span overlaps it. Historical `idle_ns`, `hardware_queue`,
`driver_submission`, and `host_behind` fields remain for existing consumers.
They describe absence of recorded work and its position relative to API
entry and return, not proven hardware idle, queue latency, driver delay, or
host starvation. Correlation IDs alone cannot prove those mechanisms.

Missing, truncated, omitted, or lossy journals are refused. A lossless journal
still cannot account for activity kinds that were disabled or unsupported.
The report does not infer unseen work or extend its window to capture edges.

CPU-only regression example:

```
cd nvidia
go test ./x/cudaprof/gaps -run 'TestCountsMemsetsAsBusy|TestGapEvidenceLimits'
```

The first test reconstructs a journal with memset between kernels. The second
merges overlapping streams and a memset, clips synchronization to the gap,
and retains the gap's unknown cause and activity-bounded window.

`correlation_match_pct` uses the same strict API linkage rules as `correlate`.
`api_timing_association_pct` separately counts unique API IDs in the selected
process-generation journal and domain, including graph timing associations.
A graph timing association does not identify graph construction or replay
provenance. Reused IDs within a journal and domain remain unmatched. API record
counts include both domains and repeated calls; they are not unique ID counts.
Use `--api-domain=driver` (the default) or `--api-domain=runtime` explicitly when
comparing host timing. Neither percentage establishes a cause for a gap.
