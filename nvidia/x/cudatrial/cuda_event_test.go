package cudatrial_test

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudatrial"
)

func TestCUDAEventTiming(t *testing.T) {
	var r cudatrial.Receipt
	if _, err := r.DurationFor(cudatrial.TimingCUDAEvent); err == nil {
		t.Fatal("missing event accepted")
	}
	if err := r.RecordTiming(cudatrial.TimingCUDAEvent, 0); err != nil {
		t.Fatal(err)
	}
	if d, err := r.DurationFor(cudatrial.TimingCUDAEvent); err != nil || d != 0 || !r.HasTiming(cudatrial.TimingCUDAEvent) {
		t.Fatalf("zero event = %v, %v", d, err)
	}
	for _, kind := range []cudatrial.TimingKind{cudatrial.TimingWallElapsed, cudatrial.TimingSummedKernel, cudatrial.TimingGPUBusy, cudatrial.TimingAPICall} {
		if r.HasTiming(kind) {
			t.Errorf("unexpected %s", kind)
		}
	}
	r.Name = "arm"
	if s := cudatrial.FormatSummary([]cudatrial.Receipt{r}); !strings.Contains(s, "event") {
		t.Fatalf("event summary: %q", s)
	}
}

func ExampleTimingCUDAEvent() {
	var r cudatrial.Receipt
	_ = r.RecordTiming(cudatrial.TimingCUDAEvent, 125*time.Nanosecond)
	d, _ := r.DurationFor(cudatrial.TimingCUDAEvent)
	fmt.Println(d)
	// Output: 125ns
}
