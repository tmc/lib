# egraph

`egraph` implements equality graphs, rewrite rules, bounded saturation,
cost-based extraction, and Metal Shading Language generation for Metal
inference work.

Create a graph with `New`, add nodes with `Add`, and define rewrites with
`NewRewrite` or the pattern constructors. Use `SaturateWithReport` or
`ExecuteSchedule` to run bounded rewrite schedules, then `ExtractBest` or
`ExtractTopK` to select expressions.

## Features

- Pattern, relation, and typed function-table rewrites, with analysis updates,
  subsumption, and exact constructor deletion.
- Reported saturation limits and composable run, sequence, repeat, and
  fixed-point schedules.
- Cost-based extraction, including top-k alternatives and checked additive
  extraction for negative local costs.
- Candidate snapshots and bounded candidate search with caller evaluation,
  ranking, restart, and final validation hooks.
- Typed operator and datatype registries, strict sort checking, and registered
  MSL generators and rewrite factories.
- Symbolic non-negative dimensions, interval specialization, and independently
  saturated search spaces for interval buckets.
- Typed relations and function tables, including fresh-ID defaults, row
  subsumption, and checked scalar action expressions.
- Rebuild and rewrite reports for merges, analyses, relations, function rows,
  node deletions, and subsumptions.

An `EGraph` is mutable and is not safe for concurrent use. Stored node payloads
and analysis values must remain immutable. Cost models must be deterministic
and must not mutate node payloads.
