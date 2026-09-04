package perfetto_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"sync"
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

func TestConcurrentSequenceReset(t *testing.T) {
	var buf bytes.Buffer
	w := perfetto.NewWriter(&buf)
	var wg sync.WaitGroup
	for i := 1; i <= 64; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			if err := w.WriteTrackDescriptor(uint64(id), "", 0); err != nil {
				t.Errorf("WriteTrackDescriptor: %v", err)
			}
		}(i)
	}
	wg.Wait()
	if got := countPacketVarintField(t, buf.Bytes(), perfetto.FieldPacketSequenceFlags, perfetto.SequenceFlagIncrementalStateCleared); got != 1 {
		t.Fatalf("incremental-state-cleared flags = %d, want 1", got)
	}
}

func TestSequenceResetRetriesAfterWriteFailure(t *testing.T) {
	want := errors.New("write failed")
	fw := &failOnceWriter{err: want}
	w := perfetto.NewWriter(fw)
	if err := w.WriteTrackDescriptor(1, "", 0); !errors.Is(err, want) {
		t.Fatalf("first write error = %v, want %v", err, want)
	}
	if err := w.WriteTrackDescriptor(2, "", 0); err != nil {
		t.Fatal(err)
	}
	if got := countPacketVarintField(t, fw.buf.Bytes(), perfetto.FieldPacketSequenceFlags, perfetto.SequenceFlagIncrementalStateCleared); got != 1 {
		t.Fatalf("incremental-state-cleared flags = %d, want 1", got)
	}
}

type failOnceWriter struct {
	err  error
	buf  bytes.Buffer
	once sync.Once
}

func (w *failOnceWriter) Write(p []byte) (int, error) {
	failed := false
	w.once.Do(func() { failed = true })
	if failed {
		return 0, w.err
	}
	return w.buf.Write(p)
}

func countPacketVarintField(t *testing.T, trace []byte, field int, value uint64) int {
	t.Helper()
	count := 0
	for len(trace) > 0 {
		tag, n := binary.Uvarint(trace)
		if n <= 0 || tag != uint64(perfetto.FieldTracePacket<<3|2) {
			t.Fatal("invalid trace packet")
		}
		trace = trace[n:]
		length, n := binary.Uvarint(trace)
		if n <= 0 || length > uint64(len(trace)-n) {
			t.Fatal("invalid trace packet length")
		}
		packet := trace[n : n+int(length)]
		trace = trace[n+int(length):]
		for len(packet) > 0 {
			packetTag, n := binary.Uvarint(packet)
			if n <= 0 {
				t.Fatal("invalid packet field")
			}
			packet = packet[n:]
			switch packetTag & 7 {
			case 0:
				got, n := binary.Uvarint(packet)
				if n <= 0 {
					t.Fatal("invalid packet varint")
				}
				if int(packetTag>>3) == field && got == value {
					count++
				}
				packet = packet[n:]
			case 1:
				if len(packet) < 8 {
					t.Fatal("short fixed64 packet field")
				}
				packet = packet[8:]
			case 2:
				length, n := binary.Uvarint(packet)
				if n <= 0 || length > uint64(len(packet)-n) {
					t.Fatal("invalid packet bytes field")
				}
				packet = packet[n+int(length):]
			case 5:
				if len(packet) < 4 {
					t.Fatal("short fixed32 packet field")
				}
				packet = packet[4:]
			default:
				t.Fatalf("unsupported wire type %d", packetTag&7)
			}
		}
	}
	return count
}

var _ io.Writer = (*failOnceWriter)(nil)

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
