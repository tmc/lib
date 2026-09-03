package metal_test

import (
	"testing"

	applemetal "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
)

func TestThreads(t *testing.T) {
	tests := []struct {
		name string
		n    int
		want applemetal.MTLSize
	}{
		{"exact", 2048, applemetal.MTLSize{Width: 2048, Height: 1, Depth: 1}},
		{"one", 1, applemetal.MTLSize{Width: 1, Height: 1, Depth: 1}},
		{"zero", 0, applemetal.MTLSize{Width: 1, Height: 1, Depth: 1}},
		{"negative", -8, applemetal.MTLSize{Width: 1, Height: 1, Depth: 1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := metal.Threads(tt.n); got != tt.want {
				t.Errorf("Threads(%d) = %+v, want %+v", tt.n, got, tt.want)
			}
		})
	}
}

func TestThreadgroups(t *testing.T) {
	tests := []struct {
		name       string
		n, group   int
		wantGroups uint
	}{
		{"exact", 1024, 128, 8},
		{"partial", 1025, 128, 9},
		{"smaller than group", 3, 128, 1},
		{"zero group", 100, 0, 100},
		{"zero threads", 0, 128, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := metal.Threadgroups(tt.n, tt.group)
			want := applemetal.MTLSize{Width: tt.wantGroups, Height: 1, Depth: 1}
			if got != want {
				t.Errorf("Threadgroups(%d, %d) = %+v, want %+v", tt.n, tt.group, got, want)
			}
		})
	}
}

// TestSetScalarRejectsInt documents that [metal.Scalar] excludes int and uint.
// The commented calls below must not compile; MSL's int and uint are 32 bits
// while Go's are 64, so allowing them would silently corrupt the argument
// table.
func TestSetScalarRejectsInt(t *testing.T) {
	// var enc applemetal.MTLComputeCommandEncoder
	// metal.SetScalar(enc, 1, 0)          // int
	// metal.SetScalar(enc, uint(1), 0)    // uint
	var _ func(applemetal.MTLComputeCommandEncoder, uint32, int) = metal.SetScalar[uint32]
	var _ func(applemetal.MTLComputeCommandEncoder, float32, int) = metal.SetScalar[float32]
}
