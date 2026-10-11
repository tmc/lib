# Existing Gemma4 capture demonstration

This CPU-only check uses existing machine-local Gemma4 captures from
2026-09-30. No new GPU work was collected.
The preexisting `plan.json` specifies the fixed-token workload's independent
arg_reduce expectations and explains their engine-specific arithmetic.
These expectations were not chosen from the observed capture counts.

Reproduce the workload checks with:

```sh
captures="$HOME/tmp/mlx-go-gemma4-deltas-20260930/pipeline"
cudaprof iterations -iteration-marker arg_reduce -expect-markers 33 \
  -skip-iterations 0 -json "$captures/go-fixed32.pprof" > go-gate.json
cudaprof iterations -iteration-marker arg_reduce -expect-markers 35 \
  -skip-iterations 0 -json "$captures/python-fixed32.pprof" > python-gate.json
```

Each command exited 2: its independent marker-count expectation was met, but
its existing iteration stationarity gate failed. Collector qualification
found no contradictory loss evidence. Summed timing comparisons remain
unqualified; no threshold was relaxed and no causal speedup is claimed.

Diagnostic commands:

```sh
cudaprof correlate "$captures/go-fixed32.pprof" > go-links.json
cudaprof correlate "$captures/python-fixed32.pprof" > python-links.json
cudaprof shape-compare "$captures/go-fixed32.pprof" \
  "$captures/python-fixed32.pprof" > shapes.json
cudaprof memory "$captures/go-fixed32.pprof" > memory.json
```

The shape diagnostic exited 0, meaning the JSON was written, not that shape
qualification passed. Shape qualification failed and both stationarity
failures remained visible. These captures lack explicit application ranges;
most recorded device work has graph provenance that this implementation
cannot resolve into construction versus replay submissions. The diagnostic
retains missing and unsupported identities instead of reconstructing token,
dtype or cache length from names. No numerical timing comparison is justified.

The memory diagnostic exited 0 and reported residency, managed-memory
migration and page-fault evidence unsupported by the current collector.
Device/backend support remains unprobed. Retained memcpy and allocation-call
records are separate evidence and do not change those dispositions.

Local output artifacts for this check are `gemma4-shapes.json`, `gemma4-memory.json`,
`go-correlations.json`, `python-correlations.json`, `go-gate.json`, and
`python-gate.json`. These files and the source captures are machine-local,
are not committed, and are not portable fixtures. The focused synthetic tests
provide portable coverage of matched identities, alternating dispatch,
missing metadata, ambiguity, graph limits, and failed completeness receipts.
