package cudaprof_test

import (
	"bytes"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof"
)

var errPerfettoWrite = errors.New("perfetto write failed")

type failAtWriter struct {
	w      bytes.Buffer
	n      int
	failAt int
}

func (w *failAtWriter) Write(p []byte) (int, error) {
	w.n++
	if w.n == w.failAt {
		return 0, errPerfettoWrite
	}
	return w.w.Write(p)
}

func TestWritePerfettoToWriteErrors(t *testing.T) {
	tests := []struct {
		name   string
		failAt int
		want   error
	}{
		{name: "success", failAt: 0},
		{name: "failure in sample", failAt: 21, want: errPerfettoWrite},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fr := cudaprof.NewFlightRecorder(cudaprof.WithWindowSize(2))
			if err := fr.Start(); err != nil {
				t.Fatal(err)
			}
			fr.Record(cudaprof.SampleRecord{
				TimestampNs: 1,
				KernelName:  "kernel",
				Values: cudaprof.SampleValues{
					Samples: 1, VRAMAllocBytes: 1, DRAMReadBytes: 1,
					DRAMWriteBytes: 1, TensorWarps: 1, L2SectorsRead: 1,
					L2SectorsMiss: 1, ActiveWarps: 1, EligibleWarps: 1,
					IssuedWarps: 1, InstExecuted: 1, BankConflicts: 1,
					RegSpillBytes: 1, InterconnectTxBytes: 1, InterconnectRxBytes: 1,
					StallMath: 1, StallMem: 1, LaunchDelayNanos: 1, QueueDelayNanos: 1,
				},
			})
			var w failAtWriter
			w.failAt = tt.failAt
			err := fr.WritePerfettoTo(&w)
			if tt.want == nil {
				if err != nil {
					t.Fatalf("WritePerfettoTo: %v", err)
				}
				return
			}
			if !errors.Is(err, tt.want) {
				t.Fatalf("WritePerfettoTo error %v, want %v", err, tt.want)
			}
		})
	}
}
