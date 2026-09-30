# Retained submission links

`cudaprof correlate [-domain driver|runtime] capture.pprof` writes a versioned
JSON report with device activities, retained host calls, NVTX ranges, and links
indexing those arrays. The driver domain is the default. Links match exact
correlation IDs within the process-generation journal; they do not match by
kernel name, timestamp proximity, thread scheduling, or stream order.

Each link carries separate API and range resolution states:

- `matched`: one retained API call, or one innermost enclosing NVTX range.
- `missing`: the capture does not retain a usable identity or matching record.
- `ambiguous`: more than one candidate shares the identity, or enclosing ranges
  cross or have identical boundaries.
- `unsupported`: the retained fields cannot resolve the relationship, including
  graph nodes, or a range whose API link is unresolved and ambiguous.

An NVTX range must enclose the whole host API interval on the same process and
thread. GPU execution may finish after that range closes. Cross-thread work
needs instrumentation on the submitting thread. Multiple streams may link to
the same API; summing API time once per linked kernel double counts that call.
Repeated IDs remain ambiguous even when timestamps suggest a likely candidate.
Runtime links for transfers are unsupported because the timeline projection
retains only one transfer correlation field.

Graph provenance and host stacks are unsupported. Existing graph IDs identify
membership, but cannot distinguish graph construction from replay submission.
The command refuses incomplete or omitted journals through the same journal
qualification checks used by timeline analysis. Unpaired NVTX markers are
reported separately. A submission link establishes retained host provenance;
it alone does not explain a slow kernel or qualify a performance comparison.

CPU regression tests cover asynchronous execution, process generations,
threads, nesting, overlapping ranges, repeated IDs and graph limitations. No
GPU capture or marker-overhead measurement accompanies this change.
