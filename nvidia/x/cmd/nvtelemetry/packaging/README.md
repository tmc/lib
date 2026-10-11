# Deploying `nvtelemetry` Under Least Privilege

This directory provides production packaging templates and deployment guidance for running the `nvtelemetry` node agent daemon with least privilege.

## Packaging Templates

- [`nvtelemetry.service`](nvtelemetry.service): Systemd service unit template for Linux host nodes.
- [`daemonset.yaml`](daemonset.yaml): Kubernetes DaemonSet template for containerized node deployments.

---

## Least-Privilege Architecture

### 1. Running Without Root

`nvtelemetry` does **not** require root UID privileges to sample NVIDIA accelerator telemetry, publish Prometheus/OpenMetrics endpoints, or deliver OTLP telemetry batches.

- **Character Device Access**: The NVIDIA driver exposes management interfaces via `/dev/nvidiactl`, `/dev/nvidia-uvm`, `/dev/nvidia-uvm-tools`, and per-GPU device nodes `/dev/nvidia[0-9]+`.
- **Group Membership**: Access to these device files is granted by group ownership (typically `video` or `render`) or udev rules (`MODE="0666"` or `MODE="0660"` with appropriate GID). A service user (such as `nvtelemetry` or UID `10001`) with read/write access to these nodes can initialize NVML and query all GPU telemetry metrics.
- **Read-Only Filesystem**: The daemon requires no local state disk writes during normal operation. Both the systemd unit and Kubernetes templates enforce a read-only root filesystem (`readOnlyRootFilesystem: true` / `ProtectSystem=strict`).

### 2. Linux Capabilities: `CAP_PERFMON` vs. `CAP_SYS_ADMIN`

- **Telemetry Querying Without Capabilities**: Standard NVML status queries (utilization, clock rates, temperatures, power consumption, energy counters) and mock mode do not require any elevated Linux capabilities or root privileges, needing only character device node read/write permissions.
- **`CAP_PERFMON` (Linux 5.8+) & `CAP_SYS_ADMIN`**: Templates include `CAP_PERFMON` (or legacy `CAP_SYS_ADMIN` on pre-5.8 kernels) for deployments where performance counter access is restricted by kernel policy (`perf_event_paranoid`). Neither capability is proven or required for mock or standard NVML operation.
- **Hardware Qualification Status**: Formal qualification of kernel capability bounds, driver versions, and physical accelerator behavior remains explicitly pending native gate qualification on physical GPU hardware.

---

## Health and Readiness Probes

`nvtelemetry` exposes distinct endpoints for liveness and readiness:

- **`/healthz` (Liveness)**:
  - Verifies that the agent process, HTTP listener, and lifecycle event loop are responsive and healthy.
  - Returns HTTP `200 OK` ("`ok\n`") while running.
  - Returns HTTP `503 Service Unavailable` ("`shutting down\n`") during graceful shutdown to inform upstream proxies and schedulers before connections drop.
- **`/readyz` (Readiness)**:
  - Verifies that a valid, fresh telemetry snapshot has been collected and is available for scraping.
  - Returns HTTP `503 Service Unavailable` ("`telemetry unavailable\n`") before initial sample collection, when snapshot expires beyond `max_age`, or during collector transition.
  - Returns HTTP `200 OK` ("`ready\n`") when fresh telemetry is ready to be scraped.

---

## Dynamic Reload and Target Ownership

Configuration reload can be triggered via **SIGHUP** or HTTP **`POST /-/reload`**:

```bash
# Via systemd
systemctl reload nvtelemetry

# Via PID signal
kill -HUP $(pidof nvtelemetry)

# Via HTTP
curl -X POST http://127.0.0.1:9464/-/reload
```

### Safety and Invariant Guarantees:

1. **Strict Pre-Validation**: The new configuration is parsed and strictly validated before touching running state. Unknown fields, duplicate keys, syntax errors, and out-of-bounds parameters are refused.
2. **State Preservation on Invalid Reload**: If a replacement configuration is invalid, the running collector, HTTP server, and active targets remain 100% intact and continue serving. The `nvtelemetry_agent_reloads_total{result="failure"}` counter increments.
3. **Safe Target Transition**: When sampling targets change (e.g. device allowlist or managed endpoint changes), the active target is cleanly stopped and drained within a bounded timeout before new ownership begins.
4. **No Silent Release for Blocked Sources**: If an active target or sampler is blocked and fails to drain within the timeout, ownership is **not** silently abandoned or transferred. The reload returns an error and preserves the existing source ownership.

---

## Self-Metrics

The Prometheus and OpenMetrics exposition at `/metrics` includes agent self-instrumentation distinguishing collection outcomes from delivery outcomes:

- `nvtelemetry_agent_collection_attempts_total`: Total collection ticks executed.
- `nvtelemetry_agent_collection_successes_total`: Samples collected with valid measurements.
- `nvtelemetry_agent_collection_failures_total`: Collection errors, sensor read failures (`EvidenceUnavailable`), or export formatting issues.
- `nvtelemetry_agent_delivery_enqueued_total`: Batches admitted to background delivery queues.
- `nvtelemetry_agent_delivery_successes_total`: Batches acknowledged by OTLP endpoints.
- `nvtelemetry_agent_delivery_failures_total`: Transport errors, endpoint rejections, or auth failures.
- `nvtelemetry_agent_delivery_dropped_total`: Batches refused due to queue capacity limits.
- `nvtelemetry_agent_reloads_total{result="success|failure"}`: Dynamic configuration reload attempts and outcomes.
- `nvtelemetry_agent_up`: Gauge indicating daemon lifecycle status (1 = running).
