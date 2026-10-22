package main

import "testing"

func TestBatchQ81FFNNeeded(t *testing.T) {
	tests := []struct {
		name          string
		gather        bool
		moeD2R        bool
		denseD2R      bool
		denseGateByte int
		want          bool
	}{
		{"both D2R", true, true, true, 1, false},
		{"no gather", false, true, true, 1, true},
		{"MoE fallback", true, false, true, 1, true},
		{"dense fallback", true, true, false, 1, true},
		{"missing dense route", true, true, true, 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := batchQ81FFNNeeded(tt.gather, tt.moeD2R, tt.denseD2R, tt.denseGateByte); got != tt.want {
				t.Fatalf("batchQ81FFNNeeded() = %v, want %v", got, tt.want)
			}
		})
	}
}
