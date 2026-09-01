package perfetto_test

import (
	"bytes"
	"testing"

	"github.com/tmc/lib/nvidia/x/perfetto"
)

func TestPerfettoWriter(t *testing.T) {
	var buf bytes.Buffer
	w := perfetto.NewWriter(&buf)

	// Register track
	if err := w.WriteTrackDescriptor(1001, "GPU Stream 0", 0); err != nil {
		t.Fatalf("WriteTrackDescriptor failed: %v", err)
	}

	// Write Slice
	if err := w.WriteSliceBegin(1001, 1000000, "VectorAddKernel", "gpu_kernel"); err != nil {
		t.Fatalf("WriteSliceBegin failed: %v", err)
	}

	if err := w.WriteSliceEnd(1001, 2000000); err != nil {
		t.Fatalf("WriteSliceEnd failed: %v", err)
	}

	if err := w.WriteInstant(1001, 2500000, "SyncMarker", "gpu_sync"); err != nil {
		t.Fatalf("WriteInstant failed: %v", err)
	}

	if buf.Len() == 0 {
		t.Fatal("expected non-empty trace output")
	}
}

func TestClockSnapshot(t *testing.T) {
	var buf bytes.Buffer
	w := perfetto.NewWriter(&buf)

	if err := w.WriteClockSnapshot(1700000000000, 500000000); err != nil {
		t.Fatalf("WriteClockSnapshot failed: %v", err)
	}

	if buf.Len() == 0 {
		t.Fatal("expected non-empty clock snapshot output")
	}
}

func TestProcessThreadAndCounterTracks(t *testing.T) {
	var buf bytes.Buffer
	w := perfetto.NewWriter(&buf)

	// Process & Thread
	if err := w.WriteProcessDescriptor(100, 1234, "llama-infer"); err != nil {
		t.Fatalf("WriteProcessDescriptor: %v", err)
	}
	if err := w.WriteThreadDescriptor(101, 1234, 5678, "cuda-worker-0"); err != nil {
		t.Fatalf("WriteThreadDescriptor: %v", err)
	}

	// Counter track & events
	if err := w.WriteCounterDescriptor(200, "VRAM Usage", "bytes", 100); err != nil {
		t.Fatalf("WriteCounterDescriptor: %v", err)
	}
	if err := w.WriteCounterValue(200, 1000000, 1024*1024*512); err != nil {
		t.Fatalf("WriteCounterValue: %v", err)
	}
	if err := w.WriteDoubleCounterValue(200, 2000000, 1024.5); err != nil {
		t.Fatalf("WriteDoubleCounterValue: %v", err)
	}

	if buf.Len() == 0 {
		t.Fatal("expected non-empty output")
	}
}

