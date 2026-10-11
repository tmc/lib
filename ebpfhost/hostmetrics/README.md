# Node observations

Artifact readers consume explicitly supplied io.Reader values. They never open
paths, perform network requests, discover hardware, or run privileged probes.
The optional Redfish HTTP client below performs explicit bounded requests. Version 1
snapshots retain caller-supplied node/source/version/clock identities and query
bounds. Query bounds do not identify sensor sample times. Missing statistics are
absent; malformed, oversized, duplicate and overflowing statistics are refused.

Supported source formats, pinned to Linux v6.8:

- [PSI](https://github.com/torvalds/linux/blob/v6.8/Documentation/accounting/psi.rst):
  system cpu/memory/io some/full averages and cumulative microseconds. System CPU
  full is undefined and omitted. No cgroup PSI interpretation.
- [meminfo](https://github.com/torvalds/linux/blob/v6.8/Documentation/filesystems/proc.rst):
  MemTotal, MemAvailable, MemFree, SwapTotal and SwapFree, kB converted to bytes.
- [network statistics](https://github.com/torvalds/linux/blob/v6.8/Documentation/ABI/testing/sysfs-class-net-statistics):
  selected rx/tx bytes, packets and errors.
- [RDMA counters](https://github.com/torvalds/linux/blob/v6.8/Documentation/ABI/stable/sysfs-class-infiniband):
  selected port receive/transmit data in four-byte words and packet counts.
- [block statistics](https://github.com/torvalds/linux/blob/v6.8/Documentation/admin-guide/iostats.rst):
  completed reads/writes and 512-byte sectors from a selected device's stat file.

The [DMTF Sensor.v1_9_0 schema](https://redfish.dmtf.org/schemas/v1/Sensor.v1_9_0.json)
reader accepts caller-selected URI/ID power (W) and temperature (Cel) projections.
It requires implementation metadata and preserves reported status. Missing/null
status remains unknown. Numeric readings from inactive sensors are retained in
the artifact projection but become unavailable in health composition. Null readings remain unavailable,
synthesized readings remain derived, and vendor health remains a reported
judgment. ReadingTime is retained verbatim without inventing clock calibration.
Other fields are ignored after bounded duplicate/depth checks; this is a typed
projection, not a complete Redfish schema validator. The artifact reader issues
no BMC requests.

Counter deltas require caller-established source/resource continuity and reset
handling. Readers do not derive rates or attribute node activity to GPUs.
JoinNodeHealth and JoinRedfishHealth in nvtelemetry require explicit node/device
and clock joins. Chassis power is never renamed to GPU power or energy. Health
composition leaves sensor time unknown; caller retains the Redfish artifact for
vendor health and source timestamp interpretation.

Source-derived unit, overflow and public health composition controls are host
verification. Physical PSI/sysfs/RDMA/block/BMC endpoint qualification,
authentication and deployment integration remain separate gates.

## Explicit Redfish HTTP acquisition

`NewRedfishHTTPClient` takes an explicit base URL, client, authentication headers,
per-response byte bound, request deadline and finite collection/page limits.
It copies client/headers, refuses redirects and rejects cross-origin, noncanonical,
escaped or out-of-namespace paths before sending credentials. No BMC discovery
occurs. Supply HTTPS with verified trust and an explicitly authorized BMC;
plaintext HTTP is suitable only when the caller authorizes that transport.

`GetSensor` uses the existing typed sensor projection with explicit caller
context, expected URI and ID. `GetCollection` follows same-origin next links,
refuses page cycles/duplicate members, retains known advertised counts and
refuses contradictory or truncated counts. Exceeding bounds returns an error,
never a complete partial collection. Missing advertised counts remain unknown.
Transport errors omit response bodies, URLs and credentials. Query cancellation
is checked before acquisition and return; custom transports must honor context.

Runnable examples and actual local authenticated HTTP fixtures qualify protocol
behavior. They do not qualify physical BMC credentials, vendor firmware, sensor
calibration, chassis/GPU attribution or production deployment. No skipped test
is recorded as a performed hardware test.
