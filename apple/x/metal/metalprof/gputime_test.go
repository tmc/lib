package metalprof

import (
	"testing"
	"time"
)

func TestCommandBufferTiming(t *testing.T) {
	tests := []struct {
		name      string
		timing    CommandBufferTiming
		wantValid bool
		wantGPU   time.Duration
	}{
		{"completed", CommandBufferTiming{Start: time.Second, End: time.Second + 250*time.Microsecond}, true, 250 * time.Microsecond},
		{"not completed", CommandBufferTiming{}, false, 0},
		{"end before start", CommandBufferTiming{Start: 2 * time.Second, End: time.Second}, false, -time.Second},
		{"zero width", CommandBufferTiming{Start: time.Second, End: time.Second}, false, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.timing.Valid(); got != tt.wantValid {
				t.Errorf("Valid() = %v, want %v", got, tt.wantValid)
			}
			if got := tt.timing.GPU(); got != tt.wantGPU {
				t.Errorf("GPU() = %v, want %v", got, tt.wantGPU)
			}
		})
	}
}

func TestSeconds(t *testing.T) {
	// CFTimeInterval is seconds as a float64; Metal reports GPU windows in the
	// microsecond range, so the conversion must not lose them.
	if got, want := seconds(0.000250), 250*time.Microsecond; got != want {
		t.Errorf("seconds(0.000250) = %v, want %v", got, want)
	}
}
