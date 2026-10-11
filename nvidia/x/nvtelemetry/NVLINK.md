# Selected-link NVML topology

`NewNVLinkReader` accepts an explicitly selected read-only device surface, link
indices, source/library version, node/topology identity, clock and exclusive
validity interval. `nvml.Device` implements the surface through the existing
state, remote PCI, remote device type and nominal version methods. Caller owns
NVML initialization and the handle. No enumeration, events, counters, resets,
configuration, repair or transport resources are created; no Close is needed.

The reader checks the queried source UUID before and after each bounded batch.
Query intervals must be ordered, within declared validity and contain no clock
epoch change. Cancellation is checked between calls; it cannot interrupt a
native call. Native field failures have typed availability and bounded numeric
NVML codes. Nonavailable queries cannot carry success code zero. Codes 3/13
require unsupported status, code 4 requires denied status, and other known or
future failure codes require failed status. Unknown numeric failure codes are
retained verbatim in canonical decimal form. Arbitrary diagnostic strings are
not retained. GPU loss refuses a
trusted topology result. Future raw link states and remote type enum values
remain raw; they do not authorize active GPU edges.

Remote PCI v2 BusId is NUL-terminated, canonicalized and checked against numeric
domain/bus/device fields. Original bus spelling is retained. The indeterminate
`PciSubSystemId` field is not consumed. Source semantics are the existing NVML
binding comments for [remote PCI](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__NvLink.html#_CPPv435nvmlDeviceGetNvLinkRemotePciInfo_v212nvmlDevice_tjP13nvmlPciInfo_t),
remote device type, link state and version; no new binding or ABI is introduced.

Peer PCI-to-UUID assignments must be supplied by the caller from independently
established physical GPU inventory. An unmapped, ambiguous, switch/NPU, unknown
or unsupported peer remains raw report evidence. Multiple UUIDs assigned to one
PCI address cannot yield a precise UUID. One UUID assigned contradictory PCI
addresses is refused. Known MIG UUIDs are refused as source or peer assignments: physical PCI identity
cannot identify a MIG instance. No namespace/MIG/PCI alias guess becomes exact identity.

`NVLinkSnapshot` is a bounded version 1 JSON report; `ReadNVLinkSnapshot` rejects
unknown/duplicate/aliased fields, excess bytes, invalid query intervals, hidden
unavailable values and contradictory peer assignments. Retain this report
alongside any health graph so unknown endpoints and per-field failures survive.
`JoinNVLinkHealth` requires exact node, clock epoch/domain, topology epoch,
source uniqueness, validity and present source/peer device scope. It emits only
mapped active GPU edges. A disabled link is not GPU loss. The nominal link
version is capability metadata, never bytes, bandwidth, traffic or utilization.

Public fake-device controls and strict native-health roundtrip establish bounded
source composition. They are not physical NVLink receipts, independent NVML ABI
qualification, global fabric discovery or production link/mapping proof. Actual
supported-device remote PCI/type/version reads, authoritative peer inventory,
MIG behavior, hot reconfiguration and native overhead remain separate gates.
