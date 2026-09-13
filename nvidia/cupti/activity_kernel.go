package cupti

// Layout size constants for CUPTI ActivityKernel record versions measured
// across CUDA 13.0–13.4 toolkits.
const (
	ActivityKernel10Size = 216 // CUDA 13.0–13.4
	ActivityKernel11Size = 224 // CUDA 13.1–13.4
	ActivityKernel12Size = 232 // CUDA 13.4
	ActivityKernel13Size = 248 // CUDA 13.4
)

// ActivityKernel12 represents the activity record for a kernel (CUDA 13.4).
// It extends ActivityKernel11 with execution model fields.
type ActivityKernel12 struct {
	Kind                            ActivityKind
	CacheConfig                     [1]byte
	SharedMemoryConfig              uint8
	RegistersPerThread              uint16
	PartitionedGlobalCacheRequested ActivityPartitionedGlobalCacheConfig
	PartitionedGlobalCacheExecuted  ActivityPartitionedGlobalCacheConfig
	Start                           uint64
	End                             uint64
	Completed                       uint64
	DeviceId                        uint32
	ContextId                       uint32
	StreamId                        uint32
	GridX                           int32
	GridY                           int32
	GridZ                           int32
	BlockX                          int32
	BlockY                          int32
	BlockZ                          int32
	StaticSharedMemory              int32
	DynamicSharedMemory             int32
	LocalMemoryPerThread            uint32
	LocalMemoryTotal                uint32
	CorrelationId                   uint32
	GridId                          int64
	Name                            uintptr
	Reserved0                       uintptr
	Queued                          uint64
	Submitted                       uint64
	LaunchType                      uint8
	IsSharedMemoryCarveoutRequested uint8
	SharedMemoryCarveoutRequested   uint8
	Padding                         uint8
	SharedMemoryExecuted            uint32
	GraphNodeId                     uint64
	ShmemLimitConfig                FuncShmemLimitConfig
	GraphId                         uint32
	PAccessPolicyWindow             uintptr
	ChannelID                       uint32
	ChannelType                     ChannelType
	ClusterX                        uint32
	ClusterY                        uint32
	ClusterZ                        uint32
	ClusterSchedulingPolicy         uint32
	LocalMemoryTotalV2              uint64
	MaxPotentialClusterSize         uint32
	MaxActiveClusters               uint32
	IsDeviceLaunched                uint8
	Padding3                        [7]uint8
	Priority                        int32
	Padding4                        uint32
	ExecutionModel                  uint32
	Padding5                        uint32
}

// ActivityKernel13 represents the activity record for a kernel (CUDA 13.4).
// It extends ActivityKernel12 with source graph identification fields.
type ActivityKernel13 struct {
	Kind                            ActivityKind
	CacheConfig                     [1]byte
	SharedMemoryConfig              uint8
	RegistersPerThread              uint16
	PartitionedGlobalCacheRequested ActivityPartitionedGlobalCacheConfig
	PartitionedGlobalCacheExecuted  ActivityPartitionedGlobalCacheConfig
	Start                           uint64
	End                             uint64
	Completed                       uint64
	DeviceId                        uint32
	ContextId                       uint32
	StreamId                        uint32
	GridX                           int32
	GridY                           int32
	GridZ                           int32
	BlockX                          int32
	BlockY                          int32
	BlockZ                          int32
	StaticSharedMemory              int32
	DynamicSharedMemory             int32
	LocalMemoryPerThread            uint32
	LocalMemoryTotal                uint32
	CorrelationId                   uint32
	GridId                          int64
	Name                            uintptr
	Reserved0                       uintptr
	Queued                          uint64
	Submitted                       uint64
	LaunchType                      uint8
	IsSharedMemoryCarveoutRequested uint8
	SharedMemoryCarveoutRequested   uint8
	Padding                         uint8
	SharedMemoryExecuted            uint32
	GraphNodeId                     uint64
	ShmemLimitConfig                FuncShmemLimitConfig
	GraphId                         uint32
	PAccessPolicyWindow             uintptr
	ChannelID                       uint32
	ChannelType                     ChannelType
	ClusterX                        uint32
	ClusterY                        uint32
	ClusterZ                        uint32
	ClusterSchedulingPolicy         uint32
	LocalMemoryTotalV2              uint64
	MaxPotentialClusterSize         uint32
	MaxActiveClusters               uint32
	IsDeviceLaunched                uint8
	Padding3                        [7]uint8
	Priority                        int32
	Padding4                        uint32
	ExecutionModel                  uint32
	Padding5                        uint32
	SourceGraphId                   uint32
	Padding6                        uint32
	SourceGraphNodeId               uint64
}
