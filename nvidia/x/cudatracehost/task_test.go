package cudatracehost

import (
	"github.com/tmc/lib/ebpfhost/bpfprobe"
	"github.com/tmc/lib/nvidia/x/cudaprobe"
	"testing"
)

func TestQueuedTaskIdentity(t *testing.T) {
	// No /proc entry exists for this PID. Known zero-valued generation/exec
	// measurements are valid; omitted measurements are not.
	good := bpfprobe.Event{PIDTGID: uint64(4000000000)<<32 | 4000000001, ThreadGenerationKnown: true, ExecIDKnown: true}
	for _, tt := range []struct {
		name      string
		change    func(*bpfprobe.Event)
		wantError bool
	}{
		{"queued exited task", func(*bpfprobe.Event) {}, false},
		{"wrong process", func(e *bpfprobe.Event) { e.PIDTGID ^= uint64(1) << 32 }, true},
		{"unknown birth", func(e *bpfprobe.Event) { e.ThreadGenerationKnown = false }, true},
		{"unknown exec", func(e *bpfprobe.Event) { e.ExecIDKnown = false }, true},
		{"changed exec", func(e *bpfprobe.Event) { e.ExecID = 1 }, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c := &Collector{process: cudaprobe.Process{PID: 4000000000}}
			if err := c.validateTaskEvent(good); err != nil {
				t.Fatal(err)
			}
			event := good
			tt.change(&event)
			if err := c.validateTaskEvent(event); (err != nil) != tt.wantError {
				t.Fatalf("got %v, want error %t", err, tt.wantError)
			}
		})
	}
}
