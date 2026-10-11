// Package nvtelemetrybridge publishes bounded NVML polling snapshots in a BPF
// hash map for independent readers.
//
// The map contains one key, uint32(0), and a fixed little-endian value. Each
// publication replaces the entire hash-map value, so readers never combine
// fields from different publications. Array, per-CPU and LRU maps are refused.
// The binary layout is versioned independently of the JSONL telemetry schema.
// BTF-backed maps are refused; no CO-RE compatibility is inferred.
//
// Store is satisfied by ebpfhost/bpfload.Object. The caller opens or creates and
// pins the map, supplies its inspected Info, and retains responsibility for
// choosing a namespace, permissions and an exclusive publisher lease. Reused
// pins are never removed. An
// explicitly owned pin may be removed on Close through an identity-checking
// callback. Close always releases the supplied map descriptor.
//
// Polling bounds are host query times. They do not establish sensor freshness,
// a synchronized clock, or native GPU attribution. Readers use PublishedUnix
// only to refuse stale bridge publications, not to join GPU events precisely.
package nvtelemetrybridge
