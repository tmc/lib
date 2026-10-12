# Offline workload metrics

The workload readers normalize bounded exports into `WorkloadExport`. They
retain the complete original bytes and SHA-256, label identity, numeric spelling,
timestamp spelling and units, and the presence of timestamps and steps. Generic
workload records use no framework, tokenizer, k6 or experiment SDK Go dependencies.

`WorkloadSource` supplies the export's provider/version and run context. Run,
phase and rank are explicit caller context; a run name is not an exclusive GPU
allocation. User-defined training metric units default to `unknown`. The readers
do not infer convergence, checkpoint correctness, phase transitions or successful
work from a metric's name.

## Pinned contracts and qualification

| Provider | Accepted version | Public reader | Supported interpretation | Qualification |
| --- | --- | --- | --- | --- |
| Triton | `25.06` | `ReadServingExposition` | Requests, failed requests, uncached inference items, uncached batch executions, response-cache lookups, cumulative durations, pending requests, first-response histogram | Source-derived protocol controls pass; pinned server capture pending |
| vLLM | `0.8.5` | `ReadServingExposition` | Legacy `engine/metrics.py` prefill/generation counters, finish-reason labels, scheduler gauges, KV-cache fractions and latency/token histograms | Source-derived protocol controls pass; pinned engine capture pending; V1-specific and later-version mappings pending |
| Ray | `2.44.0` | `ReadServingExposition` | Logical resource assignment, tasks/actors by state and selected reporter node-GPU gauges | Source-derived protocol controls pass; pinned cluster capture pending |
| SGLang | `0.5.21` | `ReadServingExposition` | Aggregate request/token counters, scheduler and KV-cache gauges, request/queue/token histograms, disaggregated queue and KV transfer, plus documented host-cache capacity/eviction families; source labels retained | Pinned source-derived constructed controls pass; authentic producer capture pending; no per-request identity or independent outcomes |
| Dynamo | `1.5.1` | `ReadServingExposition` | Runtime worker and frontend request, cancellation/rejection, token, latency, routing/backpressure, KV-cache event, and per-worker aggregate families; runtime labels retained | Pinned source-derived constructed controls pass; authentic scrape and exact image/build identity pending; not an authenticated SGLang 0.5.21 image pairing |
| MLflow | `2.22.0` | `ReadMLflowHistory` | Native metric-history response, user-defined scalar values, Unix milliseconds, explicit steps and pagination | Native offline file-store SDK/serializer capture and public reader-to-step-report replay pass; endpoint capture pending |
| W&B | `0.19.11` | `ReadWandBHistory` | Native history JSONL numeric scalars, `_timestamp` seconds and `_step`; media/nested values remain raw | Source-derived protocol controls pass; pinned native history export pending |
| ClearML | `1.18.0/api-2.9` | `ReadClearMLScalars` | `events.get_task_events` response data or successful HTTP envelope; `training_stats_scalar`, variant, iteration, task, Unix milliseconds and scroll token | Source-derived protocol controls pass; pinned task export pending; scalar-histogram endpoint intentionally unsupported because its schema is opaque |
| PyTorch | `2.7.0` | `ReadPyTorchAnnotations` | Native Chrome-trace complete CPU `user_annotation` intervals; exact names, PID/TID and args retained | Source-derived protocol controls pass; pinned profiler export pending; a bounded profile is not a continuous metric stream |

Unknown provider/version combinations are errors. Unmapped Prometheus metric
names are reported in `UnmappedNames` and preserved in `Raw`. Summaries,
exemplars and incompatible histogram components are refused by the shared
exposition reader. Nothing in this table establishes hardware, backend,
production-soak, comparative or precise request/kernel qualification.

## Aggregate windows

`CounterWorkloadWindow` requires matching provider version, run/phase/rank,
counter labels, units and semantics, with observations exactly at both requested
bounds in one declared clock domain and epoch. It differences exact decimal
values without converting counts through float64. It refuses visible resets,
missing series, rolling versions and incompatible clocks. A reset that recovers
between observations cannot be detected from two endpoints alone.

`HistogramWorkloadWindow` also requires identical bucket boundaries and validates
both endpoint and interval-delta cumulative distributions, including `+Inf`,
count, sum and integral bucket counts. Histograms do not reconstruct individual
request latencies. Gauges remain individual observations; neither consumer
interpolates gauges or divides request counts by batch executions automatically.

Exports without producer timestamps remain untimed. `StampWorkloadSnapshot`
records a separately measured collector observation time without changing raw
bytes or replacing native timestamps. `TimeSource` identifies that distinction.
Neither a producer wall timestamp nor a collector scrape establishes a sensor
timestamp, a clock calibration or an exact request-to-kernel join.
`ExactWorkloadJoin` explicitly refuses that conversion.

Processed prefill tokens differ from generation tokens. vLLM processed-token
counters do not establish that every token belongs to a successful, correct or
SLO-qualified response. Finish-reason labels, including aborts, remain separate.
Triton failure reasons are labels, not inferred cancellation outcomes. Triton
cache hits, successful requests, uncached inference items and uncached batch
executions remain separate denominators. KV-cache fractions are not total device
memory. Ray logical GPU assignment is not GPU activity. No goodput or
tokens-per-joule policy is inferred by these readers.

SGLang `num_requests_total` counts requests processed; it is not a successful
request count. Its abort, retraction, retry, transfer-failure, queue, cache and
KV-transfer families remain distinct. The source does not give every usage or
ratio gauge a common numeric scale, so unspecified units stay `unknown`.
Pinned SGLang and Dynamo source dimensions are checked on the resolved family for every
`_bucket`, `_sum`, and `_count` sample while each output point retains the
producer's original sample name and raw scrape bytes. SGLang streaming
request/token counters and TTFT/E2E histograms require `is_streaming=true|false`;
per-stage latency requires `stage`; realtime and effective-token counters
require their source-defined `mode` values; graph-memory gauges require a
`phase` label. Dynamo migration metrics require `migration_type` and duration
histograms also require `outcome`; router overlap metrics require
`worker_type=prefill`; tokenizer cached/uncached token counters require `model`.

Dynamo v1.5.1's three engine gauges are sourced from
`components/src/dynamo/common/utils/prometheus.py` at commit
`2093662a4d4e5fd318732935213dde08dc7be818` (retained snapshot SHA256
`44965a5ff37346c2e6456f87c7f3a454ef0148e1777fbf44fb38f4afb530851c`) and
`lib/runtime/src/metrics/prometheus_names.rs` at that commit (snapshot SHA256
`f1ffbad29088e6512add04208364db766b67dee0635747795c9450e2f3fefdef`). The
cache-usage and total-block gauges require the hierarchy labels
`dynamo_namespace`, `dynamo_component`, and `dynamo_endpoint`, plus `model` and
`dp_rank`; model-load time requires the hierarchy labels and `model`. The
Python source's `component` label is the same `dynamo_component` hierarchy
label, not a separate `component` dimension. Cache usage remains a fraction
from 0 through 1.

Dynamo worker cancellation counts cancellation signals received, not proven
engine aborts. `dynamo_component_gpu_cache_usage_percent` is documented on a
0.0–1.0 scale and is kept as a fraction (`1`); it is not multiplied by 100.
Worker `dynamo_component_*` metrics require runtime hierarchy labels; frontend
worker gauges use `worker_id`, `dp_rank`, and `worker_type` without those
hierarchy labels. Request-populated time series may be absent until a matching
request is observed; absence remains distinct from a measured zero. The SGLang
v0.5.21 and Dynamo v1.5.1 pins are independent source contracts. The Dynamo
v1.5.1 image template takes a runtime image/tag as input and does not pin the
SGLang engine version. NIXL is a separate telemetry source; these metrics do
not establish transfer-event completeness or silent-buffer-loss detection.
Aggregates do not expand into the request/attempt records accepted by
`ServingContext`.
Neither pinned source exposes a universal request/attempt/batch join key in
these aggregate families. SGLang exposes request counters and sequence/token
aggregates but no general batch identity; Dynamo's route and worker metrics are
labelled aggregate populations. `ServingContext` can carry independently
supplied attempts and outcomes, but those records must come from the consumer's
own event log. This reader does not synthesize them, infer a successful output
from `requests_total`, or allocate shared GPU work to requests.

`InspectTrainingSteps` reports observed steps, gaps per rank and missing expected
ranks per observed step. Unknown rank remains unknown. Concurrent runs and
different provider versions/phases require separate reports. Absence of an
explicit step is preserved even when a vendor protobuf would default it to zero.
`StepKind` distinguishes MLflow's logged metric step, W&B's history index and
ClearML's reported iteration. These are not automatically optimizer steps or
training-progress evidence; mixed step semantics are refused by the report.
Paginated histories, missing explicit steps and ambiguous same-metric label
series are refused. Filter one ClearML variant before requesting its report;
train and validation series must not hide each other's gaps.

## Bounds and examples

Default reader limits are 4 MiB, 10000 observations and 4096 bytes per normalized
string. Hard caps are 64 MiB, 65536 observations and 4096 string bytes. Serving
exposition lines are additionally capped at 1 MiB and use the shared parser's
label bounds. JSON nesting is capped at 128 and duplicate keys are rejected.
Numeric text is bounded before arbitrary-precision conversion, including
decimal exponents, so short hostile exponents cannot allocate unbounded integers.
Caller-provided metadata must not be changed concurrently with a reader call.
Context cancellation is checked between serving reads; the caller must close a
blocked reader to interrupt its `Read`.

Runnable examples are in `workload_example_test.go`. Fixtures and their provenance
are in `testdata/workload`. `mlflow-native.json` was captured through pinned
MLflow 2.22.0 `MlflowClient.log_metric/get_metric_history` and the vendor response
serializer. Its values are controlled test metrics, not an actual model's
training accuracy. Other checked-in fixtures are schema-derived protocol
controls, not falsely labeled native producer captures. Numeric text may already
have been rounded by a producer; preserving it does not recover lost precision.

Run the public-reader replay controls with:

```sh
GOWORK=off go test ./x/nvtelemetry -run 'Workload|Training|PyTorch' -v
GOWORK=off go test -race ./x/nvtelemetry
```

No reader performs HTTP requests, starts a framework, requests credentials or
acquires a GPU. Explicit endpoint adapters and the pending pinned native export
cells must pass their own public-reader consumer round trips before qualification.

### Caller-owned HTTP responses

`FetchWorkload` consumes one caller-owned HTTP request with the pinned reader
selected by `WorkloadSource`. It preserves the exact response body and hash,
refuses redirects, caps timeout at 30 seconds, and applies the existing byte and
metric limits. Credentials, endpoint construction and pagination stay with the
caller; errors omit transport details and response bodies. Custom transports
must honor cancellation. Caller request bodies must be bounded separately.

Triton/vLLM/Ray use their pinned Prometheus response formats; MLflow uses its
metric-history response and ClearML its scalar-event response. W&B history and
PyTorch Chrome traces are artifact downloads, not newly invented metrics APIs.
The adapter does not establish producer timestamps, synchronize clocks, infer
exclusive allocations or turn aggregate snapshots into exact request joins.
Its seven-provider controls use an in-memory transport: real endpoints,
credentials, framework exports and runtime qualification remain separate gates.

### OTLP aggregate windows

`WriteWorkloadOTLP` recomputes `CounterWorkloadWindow` and
`HistogramWorkloadWindow` from pinned Triton/vLLM snapshots. It emits private
`nvtelemetry.workload.<provider>.<vendor-name>` metrics with delta temporality;
monotonic sums refuse visible resets. Ray's supported metrics are gauges and
training history is logged scalar data, so neither is invented as a counter or
histogram. NVML measurements do not become workload latency histograms.

The caller must supply a known Unix mapping, native domain/epoch, matching
interval durations and explicit calibration provenance. Zero native start time
is valid. Unknown clocks, endpoint inconsistencies and rolling schemas refuse
export. The mapping is a caller declaration, not measured clock proof. Aggregate
windows do not authorize exact request joins or successful-token efficiency.

Histogram counts and per-bucket counts are exact unsigned decimal JSON strings.
The writer differences cumulative bucket windows into disjoint bucket counts,
including the +Inf bucket, which is omitted from finite `explicitBounds`.
Counter integers use signed OTLP decimal strings; larger values refuse export.
Finite bounds and sums use OTLP doubles, refusing overflow, underflow and bound
collisions after conversion. Double conversion can round other finite values.
Units follow the pinned producer schema, with count annotations such as
`{token}` and `{request}`. No automatic unit scaling is performed.

Attributes identify provider/version, caller run cohort, optional phase/rank,
clock mapping, derived aggregate evidence and a SHA-256 series identity. Raw
vendor labels, requests, stacks and environment fields are omitted. Run/cohort
and provenance strings must be non-sensitive caller-selected identifiers.
Encoding validates and buffers the bounded payload before touching the writer;
short writes are reported. Receiver acceptance, storage semantics and query
results require their own immutable-candidate receipts. A float time-series
backend cannot establish exact uint64 storage merely by accepting the JSON.

### Bounded provider endpoint commands

The `nvtelemetry fetch-endpoint` CLI uses explicit bounded transport/source
configuration and preserves original page bytes and hashes. GET artifact
requests and caller-supplied bounded POST JSON bodies are supported. There is no
endpoint, credential or framework discovery. `FetchMLflowHistoryPages` follows
the pinned 2.22.0 `page_token` contract for one selected run/metric with aggregate
budgets, metric identity checks and cycle refusal. An entire run or a stable
server snapshot is not inferred from a terminal history token. See
[ENDPOINTS.md](../cmd/nvtelemetry/ENDPOINTS.md) for configuration and delivery.

`FetchClearMLScalarPages` follows the ClearML 1.18.0 API 2.9 task-event
scroll contract. It requires a fresh task-specific POST body and bounds raw
bytes, raw events and pages independently of scalar filtering. Original page
bytes, hashes and integer request values are retained; contradictory totals
and repeated pages are refused.
