# Shape comparison

```
cudaprof shape-compare -iteration-marker arg_reduce base.pprof target.pprof > shapes.json
```

Both captures must retain complete journals, submission API calls, and explicit
NVTX application ranges. Use the schema in `x/cudaprof/marker`. Required matching
fields are observed phase, token, operation, dtype, cache length, grid, and block.
Optional layer, shared memory, lookahead, and materialization also participate;
absence does not match an explicit zero or false. Capture hashes, raw symbols
and template parameters remain in the report.

The command refuses incomplete captures through the same completeness checks as
iteration analysis. It reports the existing kernel-marker stationarity gate on
each full capture separately from shape qualification. A failed stationarity
gate stays failed even when shapes match. Kernel `arg_reduce` boundaries are
inferred; application labels are explicitly observed.

Matching requires one launch per complete identity on each side. Repeated
identities are ambiguous, differing counts are unmatched, and missing fields
remain unavailable. This conservative first version does not pair launches by
occurrence or merge fused and split operations. Distributions summarize matched
launch durations; they are not independent-run estimates of uncertainty.

No real Gemma4 marker-instrumented Go/Python capture has been collected for this
implementation. Existing captures without markers produce unmatched work;
the tool cannot recover missing token, dtype, or cache-length observations from
raw kernel names. Shape qualification alone supplies no causal explanation for
a GEMV residual.

## Explicit request pairing

Version 2 markers carry capture-local request IDs. They never automatically
match across arms, even if their strings agree. Supply `-request-pairs pairs.json`
to declare the intended equivalence:

```json
[{"base_journal":"go","base_process":1,"base_id":"request-1",
  "target_journal":"python","target_process":2,"target_id":"run-a",
  "role":"measured"}]
```

The receipt retains this mapping; bound trace validation recomputes it from the
original captures. Kernel shape identity excludes cumulative successful-forward
counts: Go full-prompt prefill and Python split prefill can have different counts
at the same semantic token. `work_scope` and `diagnostics` report that distinction
separately. A shape pass is not an equal-work, lifecycle-completeness or correctness
pass. Missing boundaries remain unavailable; missing counters never imply zero.
