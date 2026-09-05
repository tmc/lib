package driver

import "testing"

func TestRestoredEnumValues(t *testing.T) {
	for _, tt := range []struct {
		name      string
		got, want int64
	}{
		{"graph upload", int64(GraphInstantiateFlagUpload), 2},
		{"graph device launch", int64(GraphInstantiateFlagDeviceLaunch), 4},
		{"graph priority", int64(GraphInstantiateFlagUseNodePriority), 8},
		{"cluster dimension", int64(LaunchAttributeClusterDimension), 4},
		{"launch completion", int64(LaunchAttributeLaunchCompletionEvent), 12},
		{"shared memory mode", int64(LaunchAttributeSharedMemoryMode), 18},
		{"hardware decompression", int64(PointerAttributeIsHwDecompressCapable), 21},
		{"compute 100a", int64(TargetCompute100A), 0x10000 + 100},
		{"compute 120f", int64(TargetCompute120F), 0x20000 + 120},
		{"lightweight coredump", int64(CoredumpLightweightFlags), int64(CoredumpSkipNonrelocatedElfImages | CoredumpSkipGlobalMemory | CoredumpSkipSharedMemory | CoredumpSkipLocalMemory | CoredumpSkipConstbankMemory)},
	} {
		if tt.got != tt.want {
			t.Errorf("%s=%d want %d", tt.name, tt.got, tt.want)
		}
	}
}
