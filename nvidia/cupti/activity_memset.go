package cupti

// Layout size constants for CUPTI ActivityMemset record versions, measured with
// the compiler against the toolkit headers on an sbsa-linux host:
//
//	gcc -I/usr/local/cuda-13.X/targets/sbsa-linux/include, sizeof(CUpti_ActivityMemset{4,5})
const (
	ActivityMemset4Size = 88  // CUDA 13.0–13.3
	ActivityMemset5Size = 104 // CUDA 13.4
)

// ActivityMemset5 represents the activity record for memset (CUDA 13.4).
//
// It extends ActivityMemset4 by appending the same CUDA-graph source-node
// provenance ActivityMemcpy7 gained, at the same shared-prefix guarantee: every
// field the two versions share keeps its offset.
type ActivityMemset5 struct {
	Kind              ActivityKind
	Value             uint32
	Bytes             uint64
	Start             uint64
	End               uint64
	DeviceId          uint32
	ContextId         uint32
	StreamId          uint32
	CorrelationId     uint32
	Flags             uint16
	MemoryKind        uint16
	Pad               uint32
	Reserved0         uintptr
	GraphNodeId       uint64
	GraphId           uint32
	ChannelID         uint32
	ChannelType       ChannelType
	IsDeviceLaunched  uint8
	Pad2              [3]uint8
	SourceGraphId     uint32
	Pad3              uint32
	SourceGraphNodeId uint64
}
