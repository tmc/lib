# Compare Perfetto timelines

`export-compare` opens two complete cudaprof captures on one Perfetto timeline:

```sh
cudaprof export-compare -iteration-marker arg_reduce \
  -skip-iterations 1 -iteration-count 7 \
  -base-label Python -target-label Go \
  -base-expect-markers 10 -target-expect-markers 11 \
  -demangle -o comparison.tar python.pprof go.pprof
```

Open the TAR in a current [Perfetto UI](https://ui.perfetto.dev). It contains two
Perfetto traces, a version 1 `perfetto_manifest.json`, and a hidden
`.alignment.json` receipt. Perfetto ignores the receipt as trace input. Extract
it with `tar -xOf comparison.tar .alignment.json`.

The window uses the same marker, skip, and count rules as `compare`.
`-base-skip-iterations` and `-target-skip-iterations` override the shared skip
when the captures contain different warm-up work. Without an explicit count,
the remaining interval counts must agree. Marker matching uses both the raw
symbol and its demangled name with template arguments retained.

Only the selected starting boundary aligns. Later boundary markers preserve
timing drift, and every recorded activity retains its duration. Before-window
and after-window work is retained and labeled; the exporter does not guess
whether it represents prefill, warm-up, EOS, or lookahead. Both traces have
separate arm labels, machines, and track UUID namespaces.

The receipt records capture SHA256 hashes, original marker timestamps, origins,
selected windows, offsets, retained record counts, collector statuses, and
full/selected iteration gates. Expected marker counts are independent workload
invariants supplied by the caller. Their default is **unchecked**, not passed.
A completeness failure or expected-count mismatch refuses export. A stationarity
failure remains visible in the alignment track and manifest metadata; the
bundle remains useful diagnostically and does not certify summed performance.

The exporter rebases the traces, shifts both by the base arm's nonnegative
padding, and applies the remaining relative offset through the manifest. This
keeps every prefix event nonnegative. It uses explicit BOOTTIME packet IDs as
private display coordinates for the two synthetic comparison clocks. These
are not real host BOOTTIME measurements: the manifest states that independent
runs were aligned by a marker. Do not combine these clocks with unrelated host
traces or interpret overlapping arms as simultaneous execution.

`-demangle` changes kernel display labels only, caches each unique symbol, and
retains the raw symbol in each slice's `raw_symbol` argument. Invalid symbols
remain unchanged. The existing `export` command accepts the same opt-in flag;
its default labels and timestamps are unchanged.

The manifest follows the official [format reference](https://perfetto.dev/docs/reference/perfetto-manifest)
and [merge model](https://perfetto.dev/docs/analysis/merging-traces). Older
Perfetto versions may not support manifest merging. A current trace processor
can validate the bundle without running a GPU workload:

```sh
trace_processor comparison.tar -Q \
  "SELECT name, value FROM stats WHERE severity = 'error' AND value > 0"
```

An empty result establishes successful import. It does not establish capture
completeness or stationarity; those are separate receipts.
