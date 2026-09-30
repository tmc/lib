# Delivery dependencies

```
cudaprof dependency -html dependency.html capture.pprof
```

The command links nodes to journal activity indexes and retains submission
context and application ranges separately. Current journal fields do not retain
an explicit stream submission sequence, event wait identities, graph dependency
edges, or verified delivery/materialization endpoints. The command therefore
returns a partial inventory and no critical path. Adjacent timestamps, enclosing
API ranges, correlated launches and shared graph IDs do not supply those edges.

The `x/cudaprof/dependency` API accepts a caller-verified graph with explicit
`stream_order`, `event_wait`, `graph_edge` and `host_materialization` edges. Every
node and edge carries its raw-record provenance. A caller-provided `Complete`
flag is a claim about supplied evidence, not a native collector qualification;
the API validates its structure but cannot verify provenance against a capture.
The CLI deliberately does not accept external claims as collector evidence.

Analysis rejects cycles, mixed-clock edges, missing records and dependency edges
whose source completion overlaps destination execution. A missing endpoint,
disconnected delivery node, incomplete inventory or any missing evidence yields
a partial view. Independent branches remain separate; the longest path sums
only work on one serial dependency path. Its work time excludes gaps and must
not be presented as observed delivery latency or as a cause of an engine delta.

Controlled tests cover parallel streams represented by explicit order edges,
event waits, graph dependencies, host materialization, missing records and
endpoints, rejected timestamp edges, overlapping intervals and cycles. They are
analysis tests. A native marked-workload dependency qualification still needs
retained event/graph edge identities and delivery endpoints, plus independent
collector and overhead validation.

`-html FILE` writes a standalone, escaped HTML inventory of indexed nodes,
explicit edges, coverage reasons, source records and scoped delivery-event
observations. JSON remains on stdout. `delivered`, `eos` and `token_limit`
instantaneous markers are inventoried only as endpoint observations; missing
process/thread identity and range-marker events remain unsupported. A delivery
marker does not fill missing event/graph edges or establish a critical path.
