package main

import (
	"errors"
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
)

func TestProbeResultCode(t *testing.T) {
	for _, tt := range []struct {
		name string
		err  error
		want *int32
	}{
		{"success", nil, probeCode(0)},
		{"native", fmt.Errorf("configure: %w", &cupti.CallError{Func: "cuptiActivityConfigureUnifiedMemoryCounter", Result: cupti.ErrNotSupported}), probeCode(int32(cupti.ErrNotSupported))},
		{"unavailable symbol", errors.New("missing symbol"), nil},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got := probeResultCode(tt.err)
			if (got == nil) != (tt.want == nil) || (got != nil && *got != *tt.want) {
				t.Fatalf("code=%v want %v", got, tt.want)
			}
		})
	}
}
func probeCode(n int32) *int32 { return &n }

func TestProbeInvalidArguments(t *testing.T) {
	for _, args := range [][]string{{"-device=-1"}, {"-max-passes=0"}, {"-max-passes=1025"}, {"-um", "-metrics=sm__cycles_elapsed.avg"}, {"capture.pprof"}, {"-um", "-managed"}, {"-metrics=a,a"}, {"-metrics=a,"}} {
		if err := runProbeCommand(args); err == nil {
			t.Fatalf("accepted %v", args)
		}
	}
}
