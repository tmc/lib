package cudatrace

import (
	"context"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
)

func TestClockInvalidInputs(t *testing.T) {
	for _, tt := range []struct {
		name  string
		clock *Clock
		event *cuda.Event
	}{
		{"nil clock", nil, &cuda.Event{}},
		{"nil event", &Clock{}, nil},
		{"zero clock", &Clock{}, &cuda.Event{}},
		{"wrong device", &Clock{anchor: &cuda.Event{}, device: &cuda.Device{}}, &cuda.Event{}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := tt.clock.HostNs(tt.event); err == nil {
				t.Fatal("HostNs succeeded with invalid input")
			}
		})
	}
	if _, err := Calibrate(context.Background(), &cuda.Stream{}, &cuda.Event{}); err == nil {
		t.Fatal("Calibrate succeeded without a device context")
	}
}
