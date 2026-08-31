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

