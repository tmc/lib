package cupti

import (
	"testing"
	"unsafe"
)

func TestActivityKernelLayouts(t *testing.T) {
	tests := []struct {
		name     string
		gotSize  uintptr
		wantSize uintptr
		gotAlign uintptr
	}{
		{"ActivityKernel10", unsafe.Sizeof(ActivityKernel10{}), ActivityKernel10Size, unsafe.Alignof(ActivityKernel10{})},
		{"ActivityKernel11", unsafe.Sizeof(ActivityKernel11{}), ActivityKernel11Size, unsafe.Alignof(ActivityKernel11{})},
		{"ActivityKernel12", unsafe.Sizeof(ActivityKernel12{}), ActivityKernel12Size, unsafe.Alignof(ActivityKernel12{})},
		{"ActivityKernel13", unsafe.Sizeof(ActivityKernel13{}), ActivityKernel13Size, unsafe.Alignof(ActivityKernel13{})},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.gotSize != tt.wantSize {
				t.Errorf("sizeof(%s) = %d, want %d", tt.name, tt.gotSize, tt.wantSize)
			}
			if tt.gotAlign != 8 {
				t.Errorf("alignof(%s) = %d, want 8", tt.name, tt.gotAlign)
			}
		})
	}
}

func TestActivityKernelFieldOffsets(t *testing.T) {
	// Base fields verified on ActivityKernel10
	var k10 ActivityKernel10
	if got := unsafe.Offsetof(k10.Kind); got != 0 {
		t.Errorf("offsetof(ActivityKernel10.Kind) = %d, want 0", got)
	}
	if got := unsafe.Offsetof(k10.Start); got != 16 {
		t.Errorf("offsetof(ActivityKernel10.Start) = %d, want 16", got)
	}
	if got := unsafe.Offsetof(k10.End); got != 24 {
		t.Errorf("offsetof(ActivityKernel10.End) = %d, want 24", got)
	}
	if got := unsafe.Offsetof(k10.DeviceId); got != 40 {
		t.Errorf("offsetof(ActivityKernel10.DeviceId) = %d, want 40", got)
	}
	if got := unsafe.Offsetof(k10.ContextId); got != 44 {
		t.Errorf("offsetof(ActivityKernel10.ContextId) = %d, want 44", got)
	}
	if got := unsafe.Offsetof(k10.StreamId); got != 48 {
		t.Errorf("offsetof(ActivityKernel10.StreamId) = %d, want 48", got)
	}
	if got := unsafe.Offsetof(k10.GridX); got != 52 {
		t.Errorf("offsetof(ActivityKernel10.GridX) = %d, want 52", got)
	}
	if got := unsafe.Offsetof(k10.BlockX); got != 64 {
		t.Errorf("offsetof(ActivityKernel10.BlockX) = %d, want 64", got)
	}
	if got := unsafe.Offsetof(k10.StaticSharedMemory); got != 76 {
		t.Errorf("offsetof(ActivityKernel10.StaticSharedMemory) = %d, want 76", got)
	}
	if got := unsafe.Offsetof(k10.DynamicSharedMemory); got != 80 {
		t.Errorf("offsetof(ActivityKernel10.DynamicSharedMemory) = %d, want 80", got)
	}
	if got := unsafe.Offsetof(k10.CorrelationId); got != 92 {
		t.Errorf("offsetof(ActivityKernel10.CorrelationId) = %d, want 92", got)
	}
	if got := unsafe.Offsetof(k10.GridId); got != 96 {
		t.Errorf("offsetof(ActivityKernel10.GridId) = %d, want 96", got)
	}
	if got := unsafe.Offsetof(k10.Name); got != 104 {
		t.Errorf("offsetof(ActivityKernel10.Name) = %d, want 104", got)
	}
	if got := unsafe.Offsetof(k10.Queued); got != 120 {
		t.Errorf("offsetof(ActivityKernel10.Queued) = %d, want 120", got)
	}
	if got := unsafe.Offsetof(k10.Submitted); got != 128 {
		t.Errorf("offsetof(ActivityKernel10.Submitted) = %d, want 128", got)
	}
	if got := unsafe.Offsetof(k10.IsDeviceLaunched); got != 208 {
		t.Errorf("offsetof(ActivityKernel10.IsDeviceLaunched) = %d, want 208", got)
	}
	if got := unsafe.Offsetof(k10.Padding3); got != 209 {
		t.Errorf("offsetof(ActivityKernel10.Padding3) = %d, want 209", got)
	}

	// ActivityKernel11 additions
	var k11 ActivityKernel11
	if got := unsafe.Offsetof(k11.Priority); got != 216 {
		t.Errorf("offsetof(ActivityKernel11.Priority) = %d, want 216", got)
	}
	if got := unsafe.Offsetof(k11.Padding4); got != 220 {
		t.Errorf("offsetof(ActivityKernel11.Padding4) = %d, want 220", got)
	}

	// ActivityKernel12 additions
	var k12 ActivityKernel12
	if got := unsafe.Offsetof(k12.Priority); got != 216 {
		t.Errorf("offsetof(ActivityKernel12.Priority) = %d, want 216", got)
	}
	if got := unsafe.Offsetof(k12.ExecutionModel); got != 224 {
		t.Errorf("offsetof(ActivityKernel12.ExecutionModel) = %d, want 224", got)
	}
	if got := unsafe.Offsetof(k12.Padding5); got != 228 {
		t.Errorf("offsetof(ActivityKernel12.Padding5) = %d, want 228", got)
	}

	// ActivityKernel13 additions
	var k13 ActivityKernel13
	if got := unsafe.Offsetof(k13.Priority); got != 216 {
		t.Errorf("offsetof(ActivityKernel13.Priority) = %d, want 216", got)
	}
	if got := unsafe.Offsetof(k13.ExecutionModel); got != 224 {
		t.Errorf("offsetof(ActivityKernel13.ExecutionModel) = %d, want 224", got)
	}
	if got := unsafe.Offsetof(k13.SourceGraphId); got != 232 {
		t.Errorf("offsetof(ActivityKernel13.SourceGraphId) = %d, want 232", got)
	}
	if got := unsafe.Offsetof(k13.Padding6); got != 236 {
		t.Errorf("offsetof(ActivityKernel13.Padding6) = %d, want 236", got)
	}
	if got := unsafe.Offsetof(k13.SourceGraphNodeId); got != 240 {
		t.Errorf("offsetof(ActivityKernel13.SourceGraphNodeId) = %d, want 240", got)
	}
}
