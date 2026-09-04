package cudatrace_test

import (
	"bytes"
	"sync"
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

func TestTracerConcurrentWrites(t *testing.T) {
	var buf bytes.Buffer
	tracer := cudatrace.New(&buf)
	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := tracer.Span(nil, "submit", "cuda", func() {}); err != nil {
				t.Errorf("Span: %v", err)
			}
			if err := tracer.Instant(nil, "queued", "cuda"); err != nil {
				t.Errorf("Instant: %v", err)
			}
		}()
	}
	wg.Wait()
	if buf.Len() == 0 {
		t.Fatal("empty trace")
	}
}

func TestRecordSpanRejectsReverseTime(t *testing.T) {
	tracer := cudatrace.New(new(bytes.Buffer))
	track, err := tracer.RegisterTimeline("gpu", 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := tracer.RecordSpan(track, 2, 1, "kernel", "gpu"); err == nil {
		t.Fatal("RecordSpan accepted end before start")
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
