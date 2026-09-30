# Offline counter receipt qualification

```
cudaprof shape-compare base.pprof target.pprof > shapes.json
cudaprof counter-receipt select -shape shapes.json -identity identity.json
cudaprof counter-receipt compare base-counter.json target-counter.json
```

`select` consumes the full `shape-compare` envelope and an exact `shape.Identity`
JSON object. Phase, token, operation, dtype, cache length, grid and block must be
explicit, including zero token/cache values. Optional layer/shared-memory/
lookahead/materialization fields match exactly, including absence. A shortened
kernel name is not a selector. Capture hashes and completeness, qualified shape
identity, unique matched groups and unique historical launches are checked.
Original stationarity receipts remain visible; selection is not an independent
timing qualification.

The result identifies historical raw symbols, journals and activity indexes. It
does not replay application code, collect counters, or prove equivalent future
launches. Input file SHA-256 digests preserve the exact supplied receipts. The
CLI structurally checks supplied JSON; it does not reopen the original captures
to attest the receipt's claims. Inputs are bounded to 128 MiB each.

`compare` accepts two JSON `counters.Receipt` objects, retaining their values,
failures, native metric definitions/units, capture identity, device/tool versions,
replay behavior and overhead disposition. A numerical delta is emitted only for
compatible complete receipts. Unavailable or incompatible receipts preserve the
reason, omit the delta, and return failure. Unknown fields and trailing JSON are
rejected. Receipt compatibility is not workload correctness, an independent-run
statistical result, a timing comparison, or a bottleneck explanation.

Native `probe -metrics ...` runs a separate deterministic diagnostic kernel. It
is not a collection implementation for the historical GEMVs selected here and
its ordinary controls do not measure profiler overhead. Collecting counters for
qualified integrated application launches still requires an instrumented replay
workflow, independent output checks and externally arranged GPU isolation.
