package cudaoccupancy

import "testing"

// TestPredefinedProperties pins every field of the predefined properties, so
// that reading the per-SM limits from cudadevicetable cannot move one.
func TestPredefinedProperties(t *testing.T) {
	tests := []struct {
		got, want DeviceProperties
	}{
		{SM70, DeviceProperties{Architecture: "sm_70", MaxThreadsPerBlock: 1024, WarpSize: 32, MaxWarpsPerSM: 64, MaxBlocksPerSM: 32, MaxRegistersPerSM: 65536, MaxSharedMemoryPerSM: 98304, MaxSharedMemoryPerBlock: 49152, MaxSharedMemoryOptin: 98304, RegAllocationUnit: 256, SmemAllocationUnit: 128}},
		{SM75, DeviceProperties{Architecture: "sm_75", MaxThreadsPerBlock: 1024, WarpSize: 32, MaxWarpsPerSM: 32, MaxBlocksPerSM: 16, MaxRegistersPerSM: 65536, MaxSharedMemoryPerSM: 65536, MaxSharedMemoryPerBlock: 49152, MaxSharedMemoryOptin: 65536, RegAllocationUnit: 256, SmemAllocationUnit: 128}},
		{SM80, DeviceProperties{Architecture: "sm_80", MaxThreadsPerBlock: 1024, WarpSize: 32, MaxWarpsPerSM: 64, MaxBlocksPerSM: 32, MaxRegistersPerSM: 65536, MaxSharedMemoryPerSM: 167936, MaxSharedMemoryPerBlock: 49152, MaxSharedMemoryOptin: 166912, RegAllocationUnit: 256, SmemAllocationUnit: 128}},
		{SM86, DeviceProperties{Architecture: "sm_86", MaxThreadsPerBlock: 1024, WarpSize: 32, MaxWarpsPerSM: 48, MaxBlocksPerSM: 16, MaxRegistersPerSM: 65536, MaxSharedMemoryPerSM: 102400, MaxSharedMemoryPerBlock: 49152, MaxSharedMemoryOptin: 101376, RegAllocationUnit: 256, SmemAllocationUnit: 128}},
		{SM89, DeviceProperties{Architecture: "sm_89", MaxThreadsPerBlock: 1024, WarpSize: 32, MaxWarpsPerSM: 48, MaxBlocksPerSM: 24, MaxRegistersPerSM: 65536, MaxSharedMemoryPerSM: 102400, MaxSharedMemoryPerBlock: 49152, MaxSharedMemoryOptin: 101376, RegAllocationUnit: 256, SmemAllocationUnit: 128}},
		{SM87, DeviceProperties{Architecture: "sm_87", MaxThreadsPerBlock: 1024, WarpSize: 32, MaxWarpsPerSM: 48, MaxBlocksPerSM: 16, MaxRegistersPerSM: 65536, MaxSharedMemoryPerSM: 167936, MaxSharedMemoryPerBlock: 49152, MaxSharedMemoryOptin: 166912, RegAllocationUnit: 256, SmemAllocationUnit: 128}},
		{SM90, DeviceProperties{Architecture: "sm_90", MaxThreadsPerBlock: 1024, WarpSize: 32, MaxWarpsPerSM: 64, MaxBlocksPerSM: 32, MaxRegistersPerSM: 65536, MaxSharedMemoryPerSM: 233472, MaxSharedMemoryPerBlock: 49152, MaxSharedMemoryOptin: 232448, RegAllocationUnit: 256, SmemAllocationUnit: 128, MaxClusterSize: 16}},
		{SM100, DeviceProperties{Architecture: "sm_100", MaxThreadsPerBlock: 1024, WarpSize: 32, MaxWarpsPerSM: 64, MaxBlocksPerSM: 32, MaxRegistersPerSM: 65536, MaxSharedMemoryPerSM: 233472, MaxSharedMemoryPerBlock: 49152, MaxSharedMemoryOptin: 232448, RegAllocationUnit: 256, SmemAllocationUnit: 128, TMEMColumnsPerSM: 512, MaxClusterSize: 16}},
		{SM103, DeviceProperties{Architecture: "sm_103", MaxThreadsPerBlock: 1024, WarpSize: 32, MaxWarpsPerSM: 64, MaxBlocksPerSM: 32, MaxRegistersPerSM: 65536, MaxSharedMemoryPerSM: 233472, MaxSharedMemoryPerBlock: 49152, MaxSharedMemoryOptin: 232448, RegAllocationUnit: 256, SmemAllocationUnit: 128, TMEMColumnsPerSM: 512, MaxClusterSize: 16}},
		{SM110, DeviceProperties{Architecture: "sm_110", MaxThreadsPerBlock: 1024, WarpSize: 32, MaxWarpsPerSM: 48, MaxBlocksPerSM: 24, MaxRegistersPerSM: 65536, MaxSharedMemoryPerSM: 233472, MaxSharedMemoryPerBlock: 49152, MaxSharedMemoryOptin: 232448, RegAllocationUnit: 256, SmemAllocationUnit: 128, TMEMColumnsPerSM: 512, MaxClusterSize: 8}},
		{SM120, DeviceProperties{Architecture: "sm_120", MaxThreadsPerBlock: 1024, WarpSize: 32, MaxWarpsPerSM: 48, MaxBlocksPerSM: 24, MaxRegistersPerSM: 65536, MaxSharedMemoryPerSM: 102400, MaxSharedMemoryPerBlock: 49152, MaxSharedMemoryOptin: 101376, RegAllocationUnit: 256, SmemAllocationUnit: 128, MaxClusterSize: 8}},
		{SM121, DeviceProperties{Architecture: "sm_121", MaxThreadsPerBlock: 1024, WarpSize: 32, MaxWarpsPerSM: 48, MaxBlocksPerSM: 24, MaxRegistersPerSM: 65536, MaxSharedMemoryPerSM: 102400, MaxSharedMemoryPerBlock: 49152, MaxSharedMemoryOptin: 101376, RegAllocationUnit: 256, SmemAllocationUnit: 128, MaxClusterSize: 8}},
	}
	for _, tt := range tests {
		if tt.got != tt.want {
			t.Errorf("%s:\n got %+v\nwant %+v", tt.want.Architecture, tt.got, tt.want)
		}
	}
}
