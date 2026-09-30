# Workload evidence implementation status

This batch implements the first independently testable slices of the eight
improvement requests. It extends existing NVTX, CUPTI activity, comparison,
qualification and range-profiler facilities. It does not certify the Gemma4
GEMV residual's cause or complete the hardware collection acceptance criteria.

| Request | Implemented | Remaining evidence or integration |
| --- | --- | --- |
| Shape comparison | Explicit observed identity, raw symbols, exact geometry, matched/unmatched JSON, ambiguity and shape failures separate from stationarity | Instrumented real Go/Python workload; fused/split operation mappings; graph replay identity |
| Application markers | Versioned strict NVTX message schema, Go/Python examples, missing/malformed/conflicting/crossing range handling, metadata in correlations and Perfetto | Application instrumentation and overhead measurement |
| CPU/GPU correlation | Process-generation journal scoping, driver/runtime selection, API/range/metadata resolution receipts, Perfetto provenance annotations | Graph construction/replay resolution, host stacks, graphical flow arrows |
| Memory evidence | Transfer and allocation-call reports, collector support matrix, explicit residency/migration/fault limits, controlled investigation procedure | Device capability probe, native managed-memory collector, controlled demonstration and event tracks |
| Hardware counters | Selection from qualified unique shapes, native metric definition and context checks, unavailable values, replay and overhead receipts | Controlled native collection using existing cuptirange, device metric inventory, replay verification and overhead measurement |
| Gap accounting | Device-wide activity union, explicit window and unknown causes, concurrent memset/synchronization regression, audit of existing collection | More activity kinds if supported; capture-edge window analysis and process-scoped legacy gap API attribution |
| Repeated comparisons | Retained block/run sample counts, descriptive block mean spread, exclusions retained, existing GPU-lock procedure documented | End-to-end correctness/identity manifests, profiled/unprofiled campaigns and run-level uncertainty justified by the design |
| Clock telemetry | Timestamp domain, requested poll interval, observed nearby poll gap, query lag/duration, sensor observation/refresh limits | Continuous sensor sampling, supported alignment and comparison tracks; observation-time uncertainty bounds |

## Validation

Affected CPU package tests pass:

```sh
cd nvidia
go test ./x/cudaprof/marker ./x/cudaprof/shape ./x/cudaprof/counters \
  ./x/cudaprof/timeline ./x/cudaprof/gaps ./x/cuptiactivity \
  ./x/perfetto ./x/cmd/cudaprof
```

Focused regressions cover alternating dispatch, absent and malformed metadata,
case-alias JSON fields, process/thread identity, asynchronous launches, repeated
correlation IDs, crossing and nested ranges, graph work, invalid metric receipts,
unsupported memory evidence, overlapping device activity and failed capture
completeness. Existing CLI help tests cover the new commands.

The [existing-capture demonstration](evidence-demo.md) ran the new diagnostic
commands on machine-local Gemma4 captures. Their independent marker expectations
were met, while existing stationarity failed and shape qualification remained
failed. There is no timing comparison or causal speedup claim.

No new GPU workload, marker-overhead measurement, native counter collection,
managed-memory support probe or profiler-overhead campaign was performed.
Hardware support is unprobed where the collector cannot provide an observation;
that is distinct from a hardware refusal established by an API result.

## Next slices

Instrument a small controlled Go/Python operation with the marker convention
and verify output equality before collecting it under one shared GPU-lock
hold. Establish matching and overhead on that workload before instrumenting
Gemma4. Investigate managed-memory and counter capabilities in separate
processes, preserve native API failures, and extend the collector only for
qualified layouts and supported evidence. Keep all receipts separate from
ordinary timing runs and preserve existing failed gates.
