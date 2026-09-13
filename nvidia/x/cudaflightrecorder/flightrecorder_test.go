package cudaflightrecorder_test

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"runtime/trace"
	"testing"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
	"github.com/tmc/lib/nvidia/x/cuptipmsampling"
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

func TestCompleteBundleStreaming(t *testing.T) {
	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
		Window:         2 * time.Second,
		Capacity:       50,
		GoTraceEnabled: true,
		RecordGPUPprof: true,
		RecordPerfetto: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := rec.Start(); err != nil {
		t.Fatal(err)
	}
	defer rec.Stop()

	rec.SetGoTraceProvider(func(ctx context.Context) ([]byte, error) {
		return []byte("fake_v2_trace_payload"), nil
	})
	rec.SetGPUPprofProvider(func(ctx context.Context) ([]byte, error) {
		return []byte("\x1f\x8b\x08fake_gpu_pprof_gzip"), nil
	})
	rec.SetPerfettoProvider(func(ctx context.Context) ([]byte, error) {
		return []byte("fake_pftrace_timeline_protobuf"), nil
	})
	rec.AttachFile("kernels.ptx", []byte(".version 9.4\n.target sm_100\n"))

	// Register on ServeMux at default /debug/flightrecorder
	mux := http.NewServeMux()
	rec.Register(mux)

	req := httptest.NewRequest(http.MethodPost, cudaflightrecorder.DefaultEndpoint+"?reason=oom_watchdog", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("HTTP status = %d, want 200", w.Code)
	}

	bundle, err := cudaflightrecorder.ReadBundle(w.Body)
	if err != nil {
		t.Fatalf("ReadBundle failed: %v", err)
	}

	if bundle.Manifest.Reason != "oom_watchdog" {
		t.Errorf("Reason = %q, want oom_watchdog", bundle.Manifest.Reason)
	}

	// Verify all 4 primary multimodal files exist in the TAR archive:
	// manifest.json, go.trace, gpu.pprof, timeline.pftrace
	requiredFiles := []string{"go.trace", "gpu.pprof", "timeline.pftrace", "kernels.ptx", "gpu_records.txt"}
	for _, name := range requiredFiles {
		if _, ok := bundle.Files[name]; !ok {
			t.Errorf("expected %q in streamed bundle archive", name)
		}
	}

	if string(bundle.Files["go.trace"]) != "fake_v2_trace_payload" {
		t.Errorf("go.trace content mismatch")
	}
	if string(bundle.Files["gpu.pprof"]) != "\x1f\x8b\x08fake_gpu_pprof_gzip" {
		t.Errorf("gpu.pprof content mismatch")
	}
	if string(bundle.Files["timeline.pftrace"]) != "fake_pftrace_timeline_protobuf" {
		t.Errorf("timeline.pftrace content mismatch")
	}
}

func TestAlignSpans(t *testing.T) {
	tests := []struct {
		name               string
		nowNs              int64
		windowNs           int64
		goStartNs          int64
		goHasData          bool
		records            []cudaflightrecorder.Record
		clockUncertaintyNs int64
		wantGoSpanNs       int64
		wantGPUSpanNs      int64
		wantOverlapSpanNs  int64
		wantReliable       bool
	}{
		{
			name:      "both_overlapping",
			nowNs:     200_000_000,
			windowNs:  100_000_000, // [100ms, 200ms]
			goStartNs: 100_000_000,
			goHasData: true,
			records: []cudaflightrecorder.Record{
				{TimestampNs: 120_000_000, DurationNs: 30_000_000}, // [120ms, 150ms]
			},
			clockUncertaintyNs: 10_000, // 10µs < 100µs and < 30ms
			wantGoSpanNs:       100_000_000,
			wantGPUSpanNs:      30_000_000,
			wantOverlapSpanNs:  30_000_000,
			wantReliable:       true,
		},
		{
			name:      "gpu_partially_outside_go",
			nowNs:     200_000_000,
			windowNs:  50_000_000, // [150ms, 200ms]
			goStartNs: 150_000_000,
			goHasData: true,
			records: []cudaflightrecorder.Record{
				{TimestampNs: 120_000_000, DurationNs: 50_000_000}, // [120ms, 170ms]
			},
			clockUncertaintyNs: 5_000,
			wantGoSpanNs:       50_000_000,
			wantGPUSpanNs:      50_000_000,
			wantOverlapSpanNs:  20_000_000, // [150ms, 170ms] = 20ms
			wantReliable:       true,
		},
		{
			name:      "gpu_completely_before_go",
			nowNs:     200_000_000,
			windowNs:  50_000_000, // [150ms, 200ms]
			goStartNs: 150_000_000,
			goHasData: true,
			records: []cudaflightrecorder.Record{
				{TimestampNs: 100_000_000, DurationNs: 20_000_000}, // [100ms, 120ms]
			},
			clockUncertaintyNs: 5_000,
			wantGoSpanNs:       50_000_000,
			wantGPUSpanNs:      20_000_000,
			wantOverlapSpanNs:  0,
			wantReliable:       false,
		},
		{
			name:               "no_gpu_records",
			nowNs:              200_000_000,
			windowNs:           100_000_000,
			goStartNs:          100_000_000,
			goHasData:          true,
			records:            nil,
			clockUncertaintyNs: 5_000,
			wantGoSpanNs:       100_000_000,
			wantGPUSpanNs:      0,
			wantOverlapSpanNs:  0,
			wantReliable:       false,
		},
		{
			name:      "no_go_trace_data",
			nowNs:     200_000_000,
			windowNs:  100_000_000,
			goStartNs: 100_000_000,
			goHasData: false,
			records: []cudaflightrecorder.Record{
				{TimestampNs: 120_000_000, DurationNs: 30_000_000},
			},
			clockUncertaintyNs: 5_000,
			wantGoSpanNs:       0,
			wantGPUSpanNs:      30_000_000,
			wantOverlapSpanNs:  0,
			wantReliable:       false,
		},
		{
			name:      "excessive_clock_uncertainty",
			nowNs:     200_000_000,
			windowNs:  100_000_000,
			goStartNs: 100_000_000,
			goHasData: true,
			records: []cudaflightrecorder.Record{
				{TimestampNs: 120_000_000, DurationNs: 30_000_000},
			},
			clockUncertaintyNs: 150_000, // 150µs >= 100µs threshold
			wantGoSpanNs:       100_000_000,
			wantGPUSpanNs:      30_000_000,
			wantOverlapSpanNs:  30_000_000,
			wantReliable:       false,
		},
		{
			name:      "clock_uncertainty_exceeds_overlap",
			nowNs:     200_000_000,
			windowNs:  100_000_000,
			goStartNs: 100_000_000,
			goHasData: true,
			records: []cudaflightrecorder.Record{
				{TimestampNs: 199_995_000, DurationNs: 5_000}, // 5µs overlap
			},
			clockUncertaintyNs: 8_000, // 8µs > 5µs overlap
			wantGoSpanNs:       100_000_000,
			wantGPUSpanNs:      5_000,
			wantOverlapSpanNs:  5_000,
			wantReliable:       false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			goSpan, gpuSpan, overlap, reliable := cudaflightrecorder.AlignSpans(
				tt.nowNs,
				tt.windowNs,
				tt.goStartNs,
				tt.goHasData,
				tt.records,
				tt.clockUncertaintyNs,
			)
			if goSpan != tt.wantGoSpanNs {
				t.Errorf("goSpanNs = %d, want %d", goSpan, tt.wantGoSpanNs)
			}
			if gpuSpan != tt.wantGPUSpanNs {
				t.Errorf("gpuSpanNs = %d, want %d", gpuSpan, tt.wantGPUSpanNs)
			}
			if overlap != tt.wantOverlapSpanNs {
				t.Errorf("overlapSpanNs = %d, want %d", overlap, tt.wantOverlapSpanNs)
			}
			if reliable != tt.wantReliable {
				t.Errorf("overlapReliable = %v, want %v", reliable, tt.wantReliable)
			}
		})
	}
}

func TestGoTraceFlightRecording(t *testing.T) {
	tests := []struct {
		name              string
		cfg               cudaflightrecorder.Config
		snapshotAfterStop bool
	}{
		{
			name: "flight_recorder_active",
			cfg: cudaflightrecorder.Config{
				Window:         2 * time.Second,
				Capacity:       100,
				GoTraceEnabled: true,
			},
		},
		{
			name: "standard_trace_adapter",
			cfg: cudaflightrecorder.Config{
				Window:          2 * time.Second,
				Capacity:        100,
				GoTraceEnabled:  true,
				GoTraceStandard: true,
			},
		},
		{
			name: "custom_window_and_max_bytes",
			cfg: cudaflightrecorder.Config{
				Window:          3 * time.Second,
				Capacity:        100,
				GoTraceEnabled:  true,
				GoTraceWindow:   1 * time.Second,
				GoTraceMaxBytes: 1048576,
			},
		},
		{
			name: "snapshot_after_stop",
			cfg: cudaflightrecorder.Config{
				Window:         2 * time.Second,
				Capacity:       100,
				GoTraceEnabled: true,
			},
			snapshotAfterStop: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec, err := cudaflightrecorder.New(tt.cfg)
			if err != nil {
				t.Fatalf("New failed: %v", err)
			}

			if err := rec.Start(); err != nil {
				t.Fatalf("Start failed: %v", err)
			}

			// Generate runtime activity while tracing
			trace.Log(context.Background(), "cudaflightrecorder_test", tt.name)
			ch := make(chan int, 1)
			go func() {
				ch <- 42
			}()
			<-ch
			time.Sleep(15 * time.Millisecond)

			// Push GPU records
			now := time.Now()
			rec.Push(cudaflightrecorder.Record{
				TimestampNs:   now.UnixNano(),
				KernelName:    "elementwise_add",
				CorrelationID: 1001,
				DurationNs:    8000,
			})
			rec.Push(cudaflightrecorder.Record{
				TimestampNs:   now.Add(2 * time.Millisecond).UnixNano(),
				KernelName:    "gemm_wmma",
				CorrelationID: 1002,
				DurationNs:    12000,
			})

			if tt.snapshotAfterStop {
				rec.Stop()
			} else {
				defer rec.Stop()
			}

			var buf bytes.Buffer
			manifest, err := rec.Snapshot(context.Background(), &buf, "integration_test")
			if err != nil {
				t.Fatalf("Snapshot failed: %v", err)
			}

			if manifest.GoTraceBytes <= 0 {
				t.Errorf("manifest.GoTraceBytes = %d, want > 0", manifest.GoTraceBytes)
			}
			if manifest.GPURecords != 2 {
				t.Errorf("manifest.GPURecords = %d, want 2", manifest.GPURecords)
			}
			if manifest.GoSpanNs <= 0 {
				t.Errorf("manifest.GoSpanNs = %d, want > 0", manifest.GoSpanNs)
			}
			if manifest.OverlapSpanNs <= 0 {
				t.Errorf("manifest.OverlapSpanNs = %d, want > 0", manifest.OverlapSpanNs)
			}
			if !manifest.OverlapReliable {
				t.Errorf("manifest.OverlapReliable = false, want true")
			}

			// Verify TAR bundle contents
			bundle, err := cudaflightrecorder.ReadBundle(&buf)
			if err != nil {
				t.Fatalf("ReadBundle failed: %v", err)
			}

			if bundle.Manifest.GoTraceBytes != manifest.GoTraceBytes {
				t.Errorf("bundle GoTraceBytes = %d, want %d", bundle.Manifest.GoTraceBytes, manifest.GoTraceBytes)
			}

			traceData, ok := bundle.Files["go.trace"]
			if !ok || len(traceData) == 0 {
				t.Fatalf("missing or empty go.trace in bundle")
			}
			if !bytes.HasPrefix(traceData, []byte("go 1.")) {
				t.Errorf("go.trace header does not start with 'go 1.', got %q", string(traceData[:min(len(traceData), 16)]))
			}

			gpuRecs, ok := bundle.Files["gpu_records.txt"]
			if !ok || len(gpuRecs) == 0 {
				t.Fatalf("missing or empty gpu_records.txt in bundle")
			}
			if !bytes.Contains(gpuRecs, []byte("gemm_wmma")) {
				t.Errorf("gpu_records.txt missing kernel name 'gemm_wmma'")
			}
		})
	}
}

func TestPushSASSMetric_And_PushPMSample(t *testing.T) {
	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
		Window:   5 * time.Second,
		Capacity: 100,
	})
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	if err := rec.Start(); err != nil {
		t.Fatalf("Start failed: %v", err)
	}
	defer rec.Stop()

	// 1. Push SASS metric
	sm := cuptipcsampling.SASSMetric{
		PC:           0x300,
		KernelName:   "gemm_fwd",
		TotalSamples: 100,
		Selected:     25,
		SelectedPct:  25.0,
		StallSamples: 75,
		Dominant: &cuptipcsampling.PipelineStall{
			Name:     "long_scoreboard",
			Category: cuptipcsampling.StallCategoryMemory,
			Samples:  50,
			Percent:  66.67,
		},
	}
	nowNs := time.Now().UnixNano()
	rec.PushSASSMetric(nowNs, 42, sm)

	// 2. Push PM sample
	pm := cuptipmsampling.Sample{
		Index:          0,
		StartTimestamp: uint64(nowNs),
		EndTimestamp:   uint64(nowNs + 1_000_000),
		DurationNs:     1_000_000,
		Values:         map[string]float64{"sm_busy": 82.5},
	}
	rec.PushPMSample(pm)

	var buf bytes.Buffer
	manifest, err := rec.Snapshot(context.Background(), &buf, "metrics_test")
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}
	if manifest.GPURecords != 2 {
		t.Errorf("manifest.GPURecords = %d, want 2", manifest.GPURecords)
	}

	bundle, err := cudaflightrecorder.ReadBundle(&buf)
	if err != nil {
		t.Fatalf("ReadBundle failed: %v", err)
	}
	gpuRecs := string(bundle.Files["gpu_records.txt"])
	if !strings.Contains(gpuRecs, "gemm_fwd") {
		t.Errorf("gpu_records.txt does not contain gemm_fwd: %s", gpuRecs)
	}
	if !strings.Contains(gpuRecs, "device.pm_sampling") {
		t.Errorf("gpu_records.txt does not contain device.pm_sampling: %s", gpuRecs)
	}
}
