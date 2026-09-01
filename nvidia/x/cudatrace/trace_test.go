package cudatrace_test

import (
	"bytes"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudatrace"
)

func TestTracer(t *testing.T) {
	var buf bytes.Buffer
	tracer := cudatrace.New(&buf)

	_, err := tracer.RegisterStream(nil, "Default Stream")
	if err != nil {
		t.Fatalf("RegisterStream failed: %v", err)
	}

	err = tracer.Span(nil, "VectorAddKernel", "kernel", func() {
		time.Sleep(1 * time.Millisecond)
	})
	if err != nil {
		t.Fatalf("Span failed: %v", err)
	}

	err = tracer.Instant(nil, "StreamSynchronize", "sync")
	if err != nil {
		t.Fatalf("Instant failed: %v", err)
	}

	if buf.Len() == 0 {
		t.Fatal("expected non-empty trace buffer")
	}

	var crashBuf bytes.Buffer
	if err := tracer.DumpPostMortem(&crashBuf); err != nil {
		t.Fatalf("DumpPostMortem failed: %v", err)
	}

	if crashBuf.Len() == 0 {
		t.Fatal("expected non-empty post-mortem crash buffer")
	}
}

func TestExplicitTimelineAndCounters(t *testing.T) {
	var buf bytes.Buffer
	tracer := cudatrace.New(&buf)

	gpuTrack, err := tracer.RegisterTimeline("Explicit GPU Queue", 0)
	if err != nil {
		t.Fatalf("RegisterTimeline: %v", err)
	}

	start := uint64(1700000000000)
	end := uint64(1700000050000)
	if err := tracer.RecordSpan(gpuTrack, start, end, "FlashAttentionForward", "gpu_compute"); err != nil {
		t.Fatalf("RecordSpan: %v", err)
	}

	vramCounter, err := tracer.RegisterCounter("Active VRAM", "bytes", gpuTrack)
	if err != nil {
		t.Fatalf("RegisterCounter: %v", err)
	}

	if err := tracer.RecordCounter(vramCounter, start, 1024*1024*256); err != nil {
		t.Fatalf("RecordCounter: %v", err)
	}
	if err := tracer.RecordDoubleCounter(vramCounter, end, 1024.75); err != nil {
		t.Fatalf("RecordDoubleCounter: %v", err)
	}

	if buf.Len() == 0 {
		t.Fatal("expected non-empty output")
	}
}

