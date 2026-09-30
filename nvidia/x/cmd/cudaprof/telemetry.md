# Capture-edge telemetry

The capture controller polls the target's journal for profiler start/stop
edges. It queries NVML after an edge becomes visible, outside the target's
profiler callback. Each successful edge receipt reports:

- `timestamp_domain`: CLOCK_MONOTONIC, shared with the agent's edge timestamp.
- `sampled_at`, `lag_nanos`, `read_nanos`: query start, delay from the edge,
  and duration of querying all devices.
- `requested_poll_nanos`: the journal polling interval, currently 1 ms.
- `observed_poll_gap_nanos`: elapsed monotonic time between the two most
  recent polling passes; absent until two passes have been observed.
- `sensor_observation` and `sensor_refresh`: explicit explanations that
  sensor observation time is unavailable and refresh cadence is unmeasured.

The observed poll gap includes scheduling delays and work in the preceding
pass. It is one interval near the edge, not a summary of the whole capture.
Polling the journal does not continuously sample NVML. Faster journal polling
can reduce edge discovery lag; it cannot force fresher hardware readings.
The device's driver can return a cached value for several queries.

Unsupported device readings remain absent with per-field errors. An unread
edge carries `unavailable`; older captures lacking these receipt fields have
unknown polling and sensor timing. No per-kernel clock is assigned.

`wall_clock` separately pairs CLOCK_MONOTONIC with Unix time. Its midpoint
mapping has pairing uncertainty of half `read_nanos` and assumes no wall-clock
step during capture. That pairing does not establish when a sensor observed
its value. These receipts are not a continuous telemetry track; a timeline
export must not interpolate them into per-kernel measurements.

`cudaprof export -telemetry` appends one bounded host-query slice per retained
edge receipt. It requires explicit unified CLOCK_MONOTONIC capability and
query timestamps; old or unsupported receipts are refused. The annotations
retain device UUID, readings, errors, lag, poll gaps, and unavailable sensor
time. These slices do not interpolate readings or assign clocks to kernels.
