# Token and operation overview

```
cudaprof overview -html overview.html marked.pprof
cudaprof overview -iteration-marker arg_reduce -html legacy.html legacy.pprof
```

JSON and standalone HTML use the same versioned summary. The HTML hierarchy is
expandable: phase, token, layer, operation, then recorded activity. Missing or
ambiguous submission context stays unattributed. A source link opens its raw
record row, retaining journal identity, decoded index and journal sequence for
lookup in the raw Perfetto annotations. Paired ranges are derived records and
have no single journal sequence.

Host scope unions, recorded device activity unions, single-device/journal GPU
span and summed kernel work are separate quantities. Parents deduplicate their
source records; adding parent and child totals would count the same work again.
Unions remain separate by journal and device. Multiple-device captures have no
single GPU span, and kernel-marker inference is unavailable for those journals.

An explicit scoped `delivered` event gives the interval from the earliest
observed host scope for that declared token identity to delivery. It is not
request latency. GPU work may cross the host scope; those records stay visible.
EOS and token-limit events are separate termination observations. Lookahead and
materialization fields remain in original marker records and do not inherit.
Reused phase/token numbers in one process journal represent one declared identity;
the current marker schema has no request identifier.

`-iteration-marker` identifies raw kernel symbols and emits separately labeled
inferred intervals. It supplies no observed token identity, delivery, or
materialization evidence. Existing unmarked Gemma4 captures therefore retain
unmatched work and unavailable observations.

Collector receipts, target outcome, correctness and stationarity remain separate
from summary generation. Missing or incomplete journals refuse timeline
reconstruction, write an empty diagnostic summary with the refusal and retained
collector receipt, and return failure. No surviving records are presented as a
complete timeline. A generated report is not workload or causal qualification.

Synthetic analysis and CLI tests cover overlapping tokens, asynchronous completion,
explicit delivery, EOS/token-limit conflicts, missing scope, ambiguity, concurrent
streams/devices, host-only hierarchy, inferred intervals, raw source links, HTML
escaping and incomplete collector receipts. Native tracing and profiler overhead
must be validated independently.
