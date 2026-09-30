# Targeted counter receipts

This CPU-only package provides selection and comparison gates. It does not
claim that a metric is supported on any device and performs no counter
collection. The existing `cuptirange` package provides bounded native metric
configuration, metric-property queries, counter availability, capture and
replay. Inspect its returned errors and native metric catalog on the actual
device before choosing metrics. Similar metric names do not establish identical
units or definitions.

For a controlled workload:

1. Capture equivalent instrumented launches with ordinary activity timing.
   Preserve capture hashes, completeness, correctness and stationarity receipts.
2. Produce a passing shape comparison, then call `Select` with the full observed
   identity. Repeated or missing identities must be repaired in instrumentation.
3. In a separate process, acquire the established GPU lock for the whole counter
   experiment. Use the same selected workload state and verify replay restores
   it on every pass. Do not acquire a nested lock inside replay.
4. Query counter availability and native metric properties before configuring
   the plan. Availability does not prove permission to collect. Preserve device
   UUID, chip, driver/tool versions, replay mode, native definitions and failures
   in each receipt. A missing observation has a nil value, never an invented zero.
5. Compare the receipts. The same device and exact definitions, versions and
   replay contracts are required. Retain each incompatible or unavailable result.

Counter availability initializes CUPTI profiling for the process; that state
outlives the query. Counter collection may replay or otherwise change execution.
An ordinary timing baseline must therefore remain separate. Set `Overhead` to
`unmeasured` until a controlled profiled/unprofiled experiment measures it.
A compatible metric delta is an observation, not a bandwidth, cache, occupancy
or contention explanation by itself. No GPU collection, device support probe,
overhead measurement or Gemma4 counter demonstration accompanied this package.
