package cudaflightrecorder_test

import (
	"bytes"
	"context"
	"slices"
	"sort"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
)

// TestFaultSnapshotCallsNoProvider checks that FaultSnapshot reads only the
// host ring and attachments. Each provider stands for a path that reaches
// the Go runtime tracer, CUPTI or the GPU, so a fault dump must not call any
// of them. The Snapshot arm proves the counters observe provider calls.
func TestFaultSnapshotCallsNoProvider(t *testing.T) {
	providers := []struct {
		file string
		data []byte
		set  func(*cudaflightrecorder.Recorder, func(context.Context) ([]byte, error))
	}{
		{"go.trace", []byte("go 1.27 trace\x00\x00\x00"), (*cudaflightrecorder.Recorder).SetGoTraceProvider},
		{"gpu.pprof", []byte("not a profile"), (*cudaflightrecorder.Recorder).SetGPUPprofProvider},
		{"timeline.pftrace", []byte("\x0a\x02\x08\x01"), (*cudaflightrecorder.Recorder).SetPerfettoProvider},
	}
	for _, tt := range []struct {
		name      string
		fault     bool
		wantCalls int64
	}{
		{"snapshot", false, 1},
		{"fault", true, 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
				Window:   5 * time.Second,
				Capacity: 8,
			})
			if err != nil {
				t.Fatal(err)
			}
			if err := rec.Start(); err != nil {
				t.Fatal(err)
			}
			defer rec.Stop()
			rec.Push(cudaflightrecorder.Record{
				TimestampNs:   time.Now().UnixNano(),
				KernelName:    "faulty_gemm_kernel",
				CorrelationID: 42,
				DurationNs:    150_000,
			})

			calls := make([]atomic.Int64, len(providers))
			for i, p := range providers {
				p.set(rec, func(context.Context) ([]byte, error) {
					calls[i].Add(1)
					return bytes.Clone(p.data), nil
				})
			}

			var buf bytes.Buffer
			if tt.fault {
				_, err = rec.FaultSnapshot(context.Background(), &buf, cudaflightrecorder.FaultInfo{
					ErrorCode:       700,
					ErrorName:       "CUDA_ERROR_ILLEGAL_ADDRESS",
					OffendingKernel: "faulty_gemm_kernel",
					CorrelationID:   42,
				})
			} else {
				_, err = rec.Snapshot(context.Background(), &buf, "control")
			}
			if err != nil {
				t.Fatal(err)
			}
			for i, p := range providers {
				if got := calls[i].Load(); got != tt.wantCalls {
					t.Errorf("%s provider calls = %d, want %d", p.file, got, tt.wantCalls)
				}
			}

			bundle, err := cudaflightrecorder.ReadBundle(&buf)
			if err != nil {
				t.Fatal(err)
			}
			if !tt.fault {
				return
			}
			var names []string
			for name := range bundle.Files {
				names = append(names, name)
			}
			sort.Strings(names)
			want := []string{"fault.json", "gpu_records.json", "gpu_records.txt", "negative_rules.json"}
			if !slices.Equal(names, want) {
				t.Errorf("fault bundle files = %q, want %q", names, want)
			}
		})
	}
}
