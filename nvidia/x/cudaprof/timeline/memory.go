package timeline

// UnifiedMemory retains a native migration or fault-service record. It is not
// device occupancy or a current-residency observation. Stream is explicitly
// unusable: NVIDIA documents the native streamId field as invalid.
type UnifiedMemory struct {
	Sequence         uint64 `json:"sequence"`
	Journal          string `json:"journal"`
	Process          uint32 `json:"process"`
	Start            int64  `json:"start_ns"`
	End              int64  `json:"end_ns"`
	Valid            bool   `json:"valid"`
	CounterKind      uint32 `json:"counter_kind"`
	Value            uint64 `json:"value"`
	Address          uint64 `json:"address"`
	RawEnd           uint64 `json:"raw_end"`
	EndValid         bool   `json:"end_valid"`
	Source           uint32 `json:"source"`
	Destination      uint32 `json:"destination"`
	SourceValid      bool   `json:"source_valid"`
	DestinationValid bool   `json:"destination_valid"`
	RawStream        uint32 `json:"raw_stream"`
	Flags            uint32 `json:"flags"`
}

type memoryPayload struct {
	CounterKind      uint32 `json:"um_counter_kind"`
	Value            uint64 `json:"um_value"`
	Address          uint64 `json:"um_address"`
	RawEnd           uint64 `json:"um_raw_end"`
	EndValid         bool   `json:"um_end_valid"`
	Source           uint32 `json:"um_src_id"`
	Destination      uint32 `json:"um_dst_id"`
	SourceValid      bool   `json:"um_src_valid"`
	DestinationValid bool   `json:"um_dst_valid"`
	RawStream        uint32 `json:"um_raw_stream"`
	Flags            uint32 `json:"um_flags"`
}

// MemoryPolicy is an observed query result or advisory/prefetch request, not
// current physical residency. A successful async prefetch request does not
// prove completion. Value is present only for a supported successful query.
type MemoryPolicy struct {
	Sequence        uint64 `json:"sequence"`
	Journal         string `json:"journal"`
	Process         uint32 `json:"process_id"`
	Thread          uint32 `json:"thread_id"`
	Correlation     uint32 `json:"correlation"`
	Callback        uint32 `json:"callback_id"`
	Context         uint32 `json:"context"`
	TimestampDomain string `json:"timestamp_domain"`
	Time            int64  `json:"start"`
	Event           string `json:"policy_event"`
	Pointer         uint64 `json:"policy_pointer"`
	Bytes           uint64 `json:"policy_bytes"`
	Attribute       int32  `json:"policy_attribute"`
	Advice          int32  `json:"policy_advice"`
	LocationType    int32  `json:"policy_location_type"`
	LocationID      int32  `json:"policy_location_id"`
	Value           *int32 `json:"policy_value"`
	Result          int32  `json:"policy_result"`
	ResultKnown     bool   `json:"policy_result_known"`
}
