# Telemetry endpoints

Build from the `nvidia` module:

```sh
go build -o nvtelemetry ./x/cmd/nvtelemetry
./nvtelemetry serve -mock -duration 10s
```

While it runs, scrape from another terminal:

```sh
curl -fsS http://127.0.0.1:9464/metrics
```

Mock values carry `backend="mock"` and derived evidence. For native NVML, replace
`-mock` with `-device-uuids GPU-<exact-uuid>`. The command selects and verifies
those handles before sampling; it never defaults to enumerating all devices.
The UUID must also match the workload allocation and any required GPU lease.

`/metrics` serves cached Prometheus 0.0.4 or negotiated OpenMetrics 1.0.0 bytes.
Scraping never queries hardware. `/readyz` checks the age of successful snapshot
publication, not sensor freshness or device health. Both routes support GET and
HEAD. Before publication, after invalid publication, or beyond `-max-age`, they
return 503. Events and bitmasks use diagnostic logs rather than numeric metrics.

The default listener is loopback. Other listeners require `-allow-remote`,
`-tls-cert`, `-tls-key`, and `-auth-token-file`. Both routes require the configured
bearer token. The token file is bounded to 4096 bytes; errors omit its contents.
HTTP header/read/write/idle deadlines are finite. `-duration`, `-budget`, byte
limits, and graceful shutdown bound collection. Exhausted budgets return an
error rather than silently claiming completion. Collection keeps one buffered
sample; a dropped historical buffer does not stop current snapshot publication.
This serving cache is not a complete historical recording.

## Endpoint configuration

`fetch-endpoint -config FILE` acquires one caller-selected provider or managed
response. The file is bounded to 64 KiB; unknown fields, duplicate keys including
case aliases, excessive nesting, trailing JSON, invalid bounds, and conflicting
sources are refused. Required transport fields are `url`, `timeout_nanos`
(1..30000000000), and `max_bytes` (1..67108864). No endpoint discovery occurs.

Optional fields:

- `bearer_token_file`: bounded token file; no inline secrets.
- `headers_file`: bounded 16 KiB JSON object of explicit header strings for
  API-key or caller-encoded Basic authentication. At most 32 headers, printable
  bounded names/values, no duplicates/case aliases or transport/hop overrides.
  Choose either this file or `bearer_token_file`. Endpoint configuration and
  arguments contain file references, not credentials. Provider endpoints and
  accepted authentication schemes remain caller-verified inputs.
- `ca_file`: additional PEM trust roots, bounded to 1 MiB. TLS verification stays enabled.
- `allow_http`: explicit permission for a remote plaintext endpoint. Loopback HTTP
  is allowed for local fixtures. Authentication on plaintext networks requires an
  operator decision; choose HTTPS for deployment.
- `method`: GET by default, or POST for a caller-owned API request.
- `body_file`: bounded 1 MiB JSON request body, only with POST.
- `max_pages`: 1..32 selects supported provider pagination; zero means one page.

Redirects are refused. HTTP requests and response parsing do not infer sensor
clock calibration, precise request/GPU joins, or server-wide completeness.
Nested source objects use the existing Go field names. For example:

```json
{
  "url": "https://tracking.example/api/2.0/mlflow/metrics/get-history?run_id=run-1&metric_key=loss",
  "timeout_nanos": 5000000000,
  "max_bytes": 4194304,
  "max_pages": 16,
  "bearer_token_file": "/run/secrets/tracking-token",
  "workload": {"Provider": "mlflow", "Version": "2.22.0", "RunID": "run-1"}
}
```

MLflow paging changes only `page_token` on a cloned GET request. All pages must
refer to the selected metric; cycles and aggregate raw-byte/metric/page limits
refuse the result. Every page retains original bytes and its independent SHA-256.
A terminal token covers this metric history traversal, not a stable snapshot of
an entire run. With single-page acquisition, inspect `NextPageToken` before
calling a response complete. Workload `Raw` is base64 in JSON output and retains
its original bytes when decoded. Provider/version support is listed in
[WORKLOADS.md](../../nvtelemetry/WORKLOADS.md).

For managed exporter acquisition, supply `managed` mappings and, when serving,
explicit `devices`. Example power-only mapping:

```json
{
  "url": "https://exporter.example/metrics",
  "timeout_nanos": 5000000000,
  "max_bytes": 1048576,
  "devices": [{"uuid": "GPU-<exact-uuid>"}],
  "managed": {
    "SourceID": "exporter-incarnation-1",
    "Producer": "dcgm-exporter",
    "ProducerVersion": "<actual-deployed-version>",
    "NativeOrigin": "dcgm",
    "Fields": [{"Metric": "DCGM_FI_DEV_POWER_USAGE", "Name": "power_milliwatts",
                "Type": "gauge", "Numerator": 1000, "Denominator": 1,
                "Integer": false, "SourceUnit": "watts"}]
  }
}
```

`serve -managed-config FILE` is an alternative to mock or NVML acquisition.
GET is required and paging is refused. Response UUIDs outside the configured
set are refused. Device metadata is an explicit caller declaration, not a
physical inventory receipt. Query bounds and wall observation are acquired
locally; managed sensor timestamps and clock calibration stay unknown.
Mappings, exact integers, original numeric spelling and source digests survive.
See [MANAGED.md](../../nvtelemetry/MANAGED.md).

## OTLP metrics and logs

A transport-only endpoint file omits `workload`, `managed`, `devices`, request
body and paging. Set `max_bytes` to at most 4 MiB. Example:

```json
{"url":"https://collector.example/v1/logs","timeout_nanos":5000000000,
 "max_bytes":1048576,"bearer_token_file":"/run/secrets/otel-token"}
```

Deliver an already encoded OTLP HTTP JSON request:

```sh
./nvtelemetry deliver -signal logs -config logs-endpoint.json -input logs-request.json
./nvtelemetry deliver -signal metrics -config metrics-endpoint.json -input metrics-request.json
```

For recurring delivery, pass `-otlp-metrics-config` and/or `-otlp-logs-config` to
`serve`. Metrics omit unavailable values and discrete fields; logs retain
canonical diagnostic status, bitmasks, events and exact values without inventing
severity or sensor time. Cumulative metric counters require independently known
`-counter-start-unix-nanos`; zero refuses counters. Collection start is not a
counter epoch. Managed source labels and raw native errors are not published.

Each signal has an independent bounded worker: at most 16 outstanding batches,
8 MiB outstanding bytes, 4096 observations per batch, 64 KiB response, three
attempts, 100 ms..1 s backoff, 30 s batch deadline and 5 s drain. Full queues
refuse admission. Mixed signals, malformed/oversized requests, authentication
failures, invalid responses, partial rejection and incomplete drains remain
errors. Retries can duplicate an accepted request; exactly-once delivery is not
claimed. An empty `partialSuccess` response with no rejection/message counts as
full acceptance. Nonempty warnings remain explicit incomplete outcomes.

Serving prints its listener URL, then per-signal nonsecret shutdown statistics
when delivery is enabled. `deliver` emits those statistics as JSON. Success
means all admitted requests received full acceptance, not durable remote storage
or a provider query receipt. No cloud provider credentials are discovered.

## Reproducible acceptance

```sh
python3 testdata/endpoint-acceptance/run.py \
  --output "$HOME/tmp/lib-bpf/<date>-endpoint-acceptance" --seconds 60
```

The output directory must be new. The harness builds this checkout's CLI for the
Docker architecture, runs it with synthetic readings, trusted HTTPS and bearer
authentication, and uses digest-pinned OTel Collector/Prometheus images. It checks
both scraped power series, actual collector decoding of diagnostic logs, target
health, paired delivery statistics, and bounded successful SIGTERM drain.
It retains source/image/harness pins, query snapshots, logs, inspections and
hashes; removes only owned containers/network, private keys, token and binary.
Fixture containers run as root solely to read mode-0600 fixture credentials.
This is a controlled CPU/backend soak, not production deployment qualification.

For an immutable receipt, run from an isolated checkout of a pinned commit and
retain its source manifest. Docker, Go and openssl are prerequisites. Authenticated
physical BMC, real provider credentials/versions, exact-head native device/MIG
qualification, clock calibration, overhead, prolonged soak and orchestrator
rollout remain separate gates. Redfish transport is in
`github.com/tmc/lib/ebpfhost/hostmetrics`; it never relabels chassis power as GPU
power and requires an explicit caller context and source identity.

## Deployment acceptance checklist

For each bare-metal, Kubernetes or Slurm deployment, retain the exact binary
revision/hash, deployed provider/driver version, node/boot/source identity,
selected allocation and UUIDs, configuration hash excluding secret contents,
endpoint/certificate identity, query receipt and final delivery counters.
A portable loopback fixture or container run does not fill that platform row.

Start one foreground collector per explicitly selected allocation. On bare
metal, have the service manager pass an operator-reviewed UUID/configuration;
do not enable silent restarts during qualification. Under Slurm, require the
UUID to match the admitted job allocation and use a finite `-duration` and
`-budget`. Under Kubernetes, schedule the operator-built image on a selected
node with its admitted NVIDIA allocation, mount endpoint/TLS/auth files as
read-only secrets, and bind the listener according to the intended network
policy. Replica count, shared credentials and duplicate-capture ownership must
be reviewed per deployment; this command does not infer scheduler ownership.

Before admission, confirm invalid source/auth/TLS configuration makes no native
or credential-bearing request. During the test, check authenticated GET/HEAD,
unauthenticated refusal, stale/invalid snapshot 503, selected UUIDs, actual
backend storage/query and rate/queue bounds. Exercise receiver interruption
and recovery, record retries/duplicates/partial acceptance, then stop the
collector and verify bounded drain, zero pending records and released owned
resources. A failed budget or incomplete delivery remains an error. Do not
restart a native qualification admission automatically or treat a failed row
as waived. Extend the controlled harness with `--seconds` only within its
explicit 10..3600 bound; a longer production soak requires an independent plan.

ClearML 1.18.0 API 2.9 scalar paging uses `max_pages` with `method: "POST"`
and a `body_file` containing a fresh task request, for example
`{"task":"task-1","event_type":"training_stats_scalar"}`. Set the workload
provider to `clearml`, version to `1.18.0/api-2.9`, and RunID to the same task.
Only `scroll_id` changes between requests; a repeated scroll identifier is
allowed, but repeated raw pages are refused. Aggregate raw event and byte
budgets include events that do not produce scalar metrics. Advertised totals
must remain consistent and match terminal counts. Original integers and page
bytes are preserved. A completed traversal does not establish a stable task
snapshot.
