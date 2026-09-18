package cupti

// Layout size constants for CUPTI ActivityMemcpy record versions, measured with
// the compiler against the toolkit headers on an sbsa-linux host:
//
//	gcc -I/usr/local/cuda-13.X/targets/sbsa-linux/include, sizeof(CUpti_ActivityMemcpy{6,7})
const (
	ActivityMemcpy6Size = 96  // CUDA 13.0–13.3
	ActivityMemcpy7Size = 112 // CUDA 13.4
)

// ActivityMemcpy7 represents the activity record for memory copies (CUDA 13.4).
//
// It extends ActivityMemcpy6 by appending CUDA-graph source-node provenance;
// every field the two share keeps its offset, which is what lets a decoder read
// the ActivityMemcpy6 prefix out of an ActivityMemcpy7 record. Verified against
// the 13.2 and 13.4 headers: the 13.4 struct is the 13.2 struct plus this tail,
// with no field reordered or resized.
type ActivityMemcpy7 struct {
	Kind                 ActivityKind
	CopyKind             uint8
	SrcKind              uint8
	DstKind              uint8
	Flags                uint8
	Bytes                uint64
	Start                uint64
	End                  uint64
	DeviceId             uint32
	ContextId            uint32
	StreamId             uint32
	CorrelationId        uint32
	RuntimeCorrelationId uint32
	Pad                  uint32
	Reserved0            uintptr
	GraphNodeId          uint64
	GraphId              uint32
	ChannelID            uint32
	ChannelType          ChannelType
	IsDeviceLaunched     uint8
	Pad2                 [3]uint8
	CopyCount            uint64
	SourceGraphId        uint32
	Pad3                 uint32
	SourceGraphNodeId    uint64
}
