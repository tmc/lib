# Managed interchange

`ReadExposition` accepts bounded Prometheus 0.0.4 and OpenMetrics 1.0.0 snapshots.
`ReadManaged` applies caller-selected mappings to canonical readings. Serialize
the result with the existing `Writer`; replay it with `ReadStreamStrict`.
Histogram components remain separate samples for a consumer to validate;
summaries, exemplars and unsupported metric types are refused. Metadata must
precede samples. Unknown fields are ignored by the managed mapping.

The additive `source` object uses `nvtelemetry.managed.v1`. It preserves exact
numeric text, producer version, field name, selected labels, native lineage and
the original publication timestamp spelling/unit. `SourceID` identifies the
source incarnation; `NativeOrigin` identifies the shared upstream observation
path. Calling two importers does not establish independent sensor evidence.

Each source also stores the full `ManagedField` conversion and its SHA-256
digest (Go JSON encoding of that field). Strict replay checks the digest,
canonical name/unit and exact scaled integer or float value against raw text.
Integer copies retain the canonical rounded float alongside exact `IntValue`.
Readers must explicitly select one source before joining duplicate lineage.

An exposition timestamp is not a sensor timestamp. The importer never sets
`HasSourceTimestamp`. Caller-provided query bounds refer to the collector's
clock; no wall-to-monotonic calibration is inferred. Freshness policies produce
unavailable readings when timestamps are absent or outside their configured
limits. Missing metrics remain absent. Counter reset detection belongs to the
existing canonical window consumer; the stateless importer does not infer a
counter epoch from its first snapshot.

Default retained labels are GPU identity fields only. Additional labels require
an explicit bounded allowlist. Do not select secrets, request IDs or environment
values. Exporters copy only the device UUID, never arbitrary source labels.

`DCGMExporterFields` pins the default counter mappings at NVIDIA exporter
[fafd151148052628061a80450b4ee037a5fa0c3c](https://github.com/NVIDIA/dcgm-exporter/blob/fafd151148052628061a80450b4ee037a5fa0c3c/etc/default-counters.csv).
It converts watts to milliwatts and MiB to bytes, retaining exact integer energy
counters. Record the actual deployed producer version separately. This mapping
does not prove that an exporter or DCGM session ran. DCGM blank sentinel ranges
are unavailable rather than valid numeric observations.

`WriteExposition` emits private canonical-unit names and separate availability
gauges. `WriteOTLPMetrics` emits OTLP/HTTP JSON private `nvtelemetry.*` names,
UCUM units and integer strings. Cumulative counters require an explicit known
counter start epoch. Unsigned integers above the signed OTLP range are refused;
bitmasks, events and unavailable readings require the diagnostic stream instead.
No sensor time or hardware semantic-convention stability is asserted.

Protocol references:
[Prometheus text](https://prometheus.io/docs/instrumenting/exposition_formats/),
[OpenMetrics 1.0.0](https://github.com/OpenObservability/OpenMetrics/blob/v1.0.0/specification/OpenMetrics.md),
[OTLP 1.11.0](https://opentelemetry.io/docs/specs/otlp/),
[GPU conventions (Development)](https://opentelemetry.io/docs/specs/semconv/hardware/gpu/).

## Qualification matrix

`DCGMIMEXFields` selects raw domain and daemon status without discovery or
administrative actions. The field definitions are pinned to NVIDIA/DCGM commit
[`64df9f894541e426e416131a9820cae97aa4dd81`](https://github.com/NVIDIA/DCGM/blob/64df9f894541e426e416131a9820cae97aa4dd81/dcgmlib/dcgm_fields.h).
The daemon codes are 0 initializing, 1 starting authentication server,
2 waiting for peers, 3 waiting for recovery, 4 initializing GPU, 5 ready,
6 shutting down, and 7 unavailable. These remain raw codes, not health judgments.
The domain statuses UP/DOWN/DEGRADED have no numeric mapping in that header;
string results retain `Native.RawText` and explicitly remain unavailable in the
numeric reading model. Discrete status fields cannot be averaged. Real IMEX
deployment, permissions, and native status/time comparisons remain separate gates.

`OpenDCGMClient` explicitly attaches to an existing engine after caller-owned
library initialization. `NewDCGMSampler` takes a fresh exclusive client, selects
physical devices and fields, creates private groups and a bounded watch, and
implements the existing `Sampler`. It records native field layout version,
numeric field ID/name, status, scalar bits and microsecond field timestamp.
The timestamp is DCGM's cache sampling time, not independent sensor-time proof.
The ID-to-UUID association and actual producer version are caller-verified
inventory inputs; model, architecture and dynamic MIG discovery remain unknown.

Configuration validation retains caller ownership. After ownership transfer,
setup failure closes the client and returns cleanup errors. Successful callers
must stop the collector and close the sampler. Close waits for a native query;
context cancellation is checked between calls and cannot interrupt an arbitrary
DCGM FFI call. The adapter creates no cancellation goroutine. It never calls
global Init/Shutdown or stops an engine owned by another process. Native calls
and own-watch cleanup remain unqualified until an admitted native receipt.

### Pinned convention migration reference

The reference is OpenTelemetry semantic conventions **v1.44.0**, commit
[`e10a930844c6951757a43b849d364f7d056ac32b`](https://github.com/open-telemetry/semantic-conventions/blob/e10a930844c6951757a43b849d364f7d056ac32b/docs/hardware/gpu.md).
Its GPU conventions have **Development** status. Private wire names remain
unchanged. This is a migration ledger, not an implemented standard-name adapter.
All standard rows require verified `hw.id`; changing instruments or units also
requires an explicit consumer migration and independent receiver controls.

| Private canonical input | Pinned standard candidate | Migration state |
| --- | --- | --- |
| gpu_utilization_pct | hw.gpu.utilization, Gauge, 1 | Adapter absent; divide percent by 100, preserve task/window semantics |
| memory_used_bytes | hw.gpu.memory.usage, UpDownCounter, By | Adapter absent; do not rename a gauge into a different instrument |
| memory_total_bytes | hw.gpu.memory.limit, UpDownCounter, By | Adapter absent; explicit instrument migration required |
| memory_utilization_pct | hw.gpu.memory.utilization, Gauge, 1 | Not equivalent: native memory activity is not allocated-memory fraction |
| pcie_rx_bytes / pcie_tx_bytes | hw.gpu.io, Counter, By | Adapter absent; explicit receive/transmit direction and counter epoch |
| xid_event / health statuses | hw.errors / hw.status | No automatic equivalence; event/count/state mappings remain absent |
| power, energy, temperature, clocks | Not listed in the pinned GPU document | Retain private names; broader hardware conventions need separate review |

`WriteOTLPLogs` and `NewLogsDelivery` encode and deliver bounded diagnostic
readings, including events, bitmasks and unavailable status. The numeric writer
continues to refuse those fields. Generic GPU histogram/exponential-histogram
export and the standard-name adapter remain portable source gaps. Aggregate
Triton/vLLM delta windows use `WriteWorkloadOTLP`; that does not establish precise
request joins or generic GPU histogram support. Endpoint serving/configuration
and reproducible backend acceptance are documented in
[ENDPOINTS.md](../cmd/nvtelemetry/ENDPOINTS.md).

| Target | Implemented path | Remaining acceptance |
| --- | --- | --- |
| Direct NVML | Existing native sampler, unchanged | Independent exact-head physical comparison, dynamic MIG, overhead |
| Direct DCGM | Existing helpers plus owned canonical sampler/watch adapter | Admitted native ownership/cleanup, version/UUID inventory, MIG and field/status fixture comparison |
| DCGM exporter | Explicit pinned mappings and bounded offline import | Authentic deployed-version capture and direct/import comparison |
| Prometheus/OpenMetrics | Bounded reader/writer and cached HTTP handler/CLI | Controlled real collector/Prometheus harness; native/deployment qualification remains separate |
| OTLP HTTP JSON | Metrics/diagnostic logs encoding and bounded independent delivery | Controlled actual collector/Prometheus harness; authenticated cloud/backend storage and production rollout remain separate |
| Datadog | Explicit configured OTLP endpoint, subject to provider support | Authenticated versioned receiver and query receipt |
| Grafana Cloud | Explicit configured OTLP endpoint | Authenticated receiver and query receipt |
| SigNoz | Explicit configured OTLP endpoint | Versioned receiver and query receipt |
| Elastic | Explicit configured OTLP endpoint, subject to provider support | Authenticated receiver and query receipt |
| New Relic | Explicit configured OTLP endpoint, subject to provider support | Authenticated receiver and query receipt |

Local byte fixtures and HTTP wire tests qualify their protocol behavior only.
Provider SDKs, credentials, native watch ownership and production deployment
stay with callers. `FetchManaged` performs bounded caller-configured HTTP
acquisition; no dependencies or duplicate collector are added.
