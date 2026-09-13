package cudaprof_test

import (
	"bytes"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof"
)

type countWriter struct {
	n int
	b bytes.Buffer
}

func (w *countWriter) Write(p []byte) (int, error) { w.n++; return w.b.Write(p) }

func newSweepRecorder(t *testing.T) *cudaprof.FlightRecorder {
	fr := cudaprof.NewFlightRecorder(cudaprof.WithWindowSize(2))
	if err := fr.Start(); err != nil {
		t.Fatal(err)
	}
	fr.Record(cudaprof.SampleRecord{
		TimestampNs: 1, KernelName: "kernel",
		Values: cudaprof.SampleValues{
			Samples: 1, VRAMAllocBytes: 1, DRAMReadBytes: 1,
			DRAMWriteBytes: 1, TensorWarps: 1, L2SectorsRead: 1,
			L2SectorsMiss: 1, ActiveWarps: 1, EligibleWarps: 1,
			IssuedWarps: 1, InstExecuted: 1, BankConflicts: 1,
			RegSpillBytes: 1, InterconnectTxBytes: 1, InterconnectRxBytes: 1,
			StallMath: 1, StallMem: 1, LaunchDelayNanos: 1, QueueDelayNanos: 1,
		},
	})
	return fr
}

func TestSweepEveryWriteIndex(t *testing.T) {
	var c countWriter
	if err := newSweepRecorder(t).WritePerfettoTo(&c); err != nil {
		t.Fatalf("baseline: %v", err)
	}
	t.Logf("total underlying writes = %d", c.n)

	var missed []int
	for i := 1; i <= c.n; i++ {
		var w failAtWriter
		w.failAt = i
		err := newSweepRecorder(t).WritePerfettoTo(&w)
		if !errors.Is(err, errPerfettoWrite) {
			missed = append(missed, i)
		}
	}
	if len(missed) > 0 {
		t.Errorf("%d of %d write indices did NOT surface the error: %v", len(missed), c.n, missed)
	} else {
		t.Logf("all %d write indices surface the error", c.n)
	}
}
