package cudaflightrecorder_test

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
)

func TestBundleRoundTrip(t *testing.T) {
	manifest := cudaflightrecorder.Manifest{
		Timestamp:          time.Now().Truncate(time.Second),
		Reason:             "test_anomaly",
		WindowRequestedNs:  int64(10 * time.Second),
		GoSpanNs:           int64(10 * time.Second),
		GPUSpanNs:          int64(5 * time.Second),
		OverlapSpanNs:      int64(5 * time.Second),
		GPURecords:         42,
		GPUDropped:         0,
		ClockUncertaintyNs: 12000,
		OverlapReliable:    true,
	}

	bundle := cudaflightrecorder.NewBundle(manifest)
	bundle.Files["gpu.pprof"] = []byte("\x1f\x8b\x08fake_pprof_bytes")
	bundle.Files["go.trace"] = []byte("fake_go_trace_v2")

	var buf bytes.Buffer
	if err := cudaflightrecorder.WriteBundle(&buf, bundle); err != nil {
		t.Fatalf("WriteBundle failed: %v", err)
	}

	readBack, err := cudaflightrecorder.ReadBundle(&buf)
	if err != nil {
		t.Fatalf("ReadBundle failed: %v", err)
	}

	if readBack.Manifest.Reason != manifest.Reason {
		t.Errorf("Reason = %q, want %q", readBack.Manifest.Reason, manifest.Reason)
	}
	if readBack.Manifest.GPURecords != 42 {
		t.Errorf("GPURecords = %d, want 42", readBack.Manifest.GPURecords)
	}
	if !bytes.Equal(readBack.Files["gpu.pprof"], bundle.Files["gpu.pprof"]) {
		t.Errorf("gpu.pprof content mismatch")
	}
	if !bytes.Equal(readBack.Files["go.trace"], bundle.Files["go.trace"]) {
		t.Errorf("go.trace content mismatch")
	}
}

func TestRecorderRingBufferAndSnapshot(t *testing.T) {
	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
		Window:   5 * time.Second,
		Capacity: 10,
	})
	if err != nil {
		t.Fatal(err)
	}

	if err := rec.Start(); err != nil {
		t.Fatalf("Start failed: %v", err)
	}
	defer rec.Stop()

	now := time.Now()
	// Push 15 records into capacity-10 buffer (should drop 5)
	for i := 0; i < 15; i++ {
		rec.Push(cudaflightrecorder.Record{
			TimestampNs:   now.Add(time.Duration(i) * time.Millisecond).UnixNano(),
			KernelName:    "gemm_kernel",
			CorrelationID: uint64(100 + i),
			DurationNs:    5000,
		})
	}

	var buf bytes.Buffer
	manifest, err := rec.Snapshot(context.Background(), &buf, "watchdog_fired")
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}

	if manifest.GPURecords != 10 {
		t.Errorf("GPURecords = %d, want 10", manifest.GPURecords)
	}
	if manifest.GPUDropped != 5 {
		t.Errorf("GPUDropped = %d, want 5", manifest.GPUDropped)
	}

	// Verify TAR archive can be read back cleanly
	bundle, err := cudaflightrecorder.ReadBundle(&buf)
	if err != nil {
		t.Fatalf("ReadBundle on snapshot failed: %v", err)
	}
	if bundle.Manifest.Reason != "watchdog_fired" {
		t.Errorf("Reason = %q, want watchdog_fired", bundle.Manifest.Reason)
	}
	if len(bundle.Files["gpu_records.txt"]) == 0 {
		t.Errorf("expected gpu_records.txt in bundle")
	}
}

func TestHTTPHandler(t *testing.T) {
	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
		Window:   1 * time.Second,
		Capacity: 100,
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
		KernelName:    "test_kernel",
		CorrelationID: 1,
		DurationNs:    1000,
	})

	handler := rec.Handler()
	req := httptest.NewRequest(http.MethodGet, "/debug/flightrecorder/snapshot?reason=manual_dump", nil)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("HTTP status = %d, want 200", w.Code)
	}
	if ct := w.Header().Get("Content-Type"); ct != "application/x-tar" {
		t.Errorf("Content-Type = %q, want application/x-tar", ct)
	}

	bundle, err := cudaflightrecorder.ReadBundle(w.Body)
	if err != nil {
		t.Fatalf("failed to read HTTP response TAR: %v", err)
	}
	if bundle.Manifest.Reason != "manual_dump" {
		t.Errorf("Reason = %q, want manual_dump", bundle.Manifest.Reason)
	}
}
