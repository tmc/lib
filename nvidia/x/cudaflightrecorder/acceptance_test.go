package cudaflightrecorder_test

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
)

func TestProviderUnavailable(t *testing.T) {
	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
		Window:         5 * time.Second,
		Capacity:       100,
		GoTraceEnabled: true,
		RecordGPUPprof: true,
		RecordPerfetto: true,
	})
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	if err := rec.Start(); err != nil {
		t.Fatalf("Start failed: %v", err)
	}
	defer rec.Stop()

	// Provider fails with drain error
	rec.SetGPUPprofProvider(func(ctx context.Context) ([]byte, error) {
		return nil, errors.New("cupti activity buffer drain failed")
	})

	var buf bytes.Buffer
	manifest, err := rec.Snapshot(context.Background(), &buf, "test_provider_unavailable")
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}

	// Verify provider error recorded in manifest
	status, ok := manifest.ProviderStatus["gpu.pprof"]
	if !ok || !strings.Contains(status, "unavailable") {
		t.Errorf("ProviderStatus[gpu.pprof] = %q, want unavailable", status)
	}

	// Verify bundle does NOT contain fake placeholder or failed gpu.pprof
	bundle, err := cudaflightrecorder.ReadBundle(&buf)
	if err != nil {
		t.Fatalf("ReadBundle failed: %v", err)
	}
	if _, exists := bundle.Files["gpu.pprof"]; exists {
		t.Errorf("gpu.pprof should not be present in bundle when provider failed")
	}
}

func TestInvalidArtifactRejected(t *testing.T) {
	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
		Window:         5 * time.Second,
		Capacity:       100,
		GoTraceEnabled: true,
		RecordGPUPprof: true,
	})
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	if err := rec.Start(); err != nil {
		t.Fatalf("Start failed: %v", err)
	}
	defer rec.Stop()

	// Supply old placeholder string for go.trace
	rec.SetGoTraceProvider(func(ctx context.Context) ([]byte, error) {
		return []byte("go_trace_v2\x00"), nil
	})
	// Supply malformed/corrupted non-gzip bytes for gpu.pprof
	rec.SetGPUPprofProvider(func(ctx context.Context) ([]byte, error) {
		return []byte("corrupt_raw_bytes_without_gzip_magic"), nil
	})

	var buf bytes.Buffer
	manifest, err := rec.Snapshot(context.Background(), &buf, "test_invalid_artifact")
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}

	// Verify rejection recorded in provider status
	goStatus := manifest.ProviderStatus["go.trace"]
	if !strings.Contains(goStatus, "rejected") {
		t.Errorf("expected go.trace rejected, got: %q", goStatus)
	}

	gpuStatus := manifest.ProviderStatus["gpu.pprof"]
	if !strings.Contains(gpuStatus, "rejected") {
		t.Errorf("expected gpu.pprof rejected, got: %q", gpuStatus)
	}

	bundle, err := cudaflightrecorder.ReadBundle(&buf)
	if err != nil {
		t.Fatalf("ReadBundle failed: %v", err)
	}
	if _, exists := bundle.Files["go.trace"]; exists {
		t.Errorf("go.trace should be rejected and not included in bundle")
	}
	if _, exists := bundle.Files["gpu.pprof"]; exists {
		t.Errorf("gpu.pprof should be rejected and not included in bundle")
	}
}

func TestSnapshotBudgets(t *testing.T) {
	// 1. Ring buffer capacity budget & forced overflow
	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
		Window:                 10 * time.Second,
		Capacity:               5,
		MaxProviderOutputBytes: 100,
		MaxSnapshotBytes:       2048,
	})
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	if err := rec.Start(); err != nil {
		t.Fatalf("Start failed: %v", err)
	}
	defer rec.Stop()

	now := time.Now()
	// Force overflow: push 15 records into capacity-5 ring buffer
	for i := 0; i < 15; i++ {
		rec.Push(cudaflightrecorder.Record{
			TimestampNs:   now.Add(time.Duration(i) * time.Millisecond).UnixNano(),
			KernelName:    "kernel_gemm",
			CorrelationID: uint64(i + 1),
			DurationNs:    1000,
		})
	}

	// 2. Provider output budget: provider returns 200 bytes, exceeding 100-byte budget
	rec.SetGoTraceProvider(func(ctx context.Context) ([]byte, error) {
		return make([]byte, 200), nil
	})

	var buf bytes.Buffer
	manifest, err := rec.Snapshot(context.Background(), &buf, "test_budgets")
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}

	if manifest.GPURecords != 5 {
		t.Errorf("GPURecords = %d, want 5", manifest.GPURecords)
	}
	if manifest.GPUDropped != 10 {
		t.Errorf("GPUDropped = %d, want 10", manifest.GPUDropped)
	}
	if !strings.Contains(manifest.ProviderStatus["go.trace"], "budget") {
		t.Errorf("expected provider budget rejection, got: %q", manifest.ProviderStatus["go.trace"])
	}

	// 3. Snapshot total size budget: attach file that exceeds 2048-byte limit
	rec.AttachFile("huge.bin", make([]byte, 3000))
	var buf2 bytes.Buffer
	_, err = rec.Snapshot(context.Background(), &buf2, "exceed_budget")
	if !errors.Is(err, cudaflightrecorder.ErrSnapshotBudgetExceeded) {
		t.Errorf("expected ErrSnapshotBudgetExceeded, got: %v", err)
	}

	// 4. Disk retention budget
	tempDir := t.TempDir()

	diskMgr, err := cudaflightrecorder.NewSnapshotDiskManager(tempDir, 1000)
	if err != nil {
		t.Fatalf("NewSnapshotDiskManager failed: %v", err)
	}

	// Save file of 400 bytes (retained = 400)
	_, err = diskMgr.SaveSnapshot("snap1.tar", make([]byte, 400))
	if err != nil {
		t.Fatalf("save snap1: %v", err)
	}
	time.Sleep(10 * time.Millisecond)

	// Save file of 400 bytes (retained = 800)
	_, err = diskMgr.SaveSnapshot("snap2.tar", make([]byte, 400))
	if err != nil {
		t.Fatalf("save snap2: %v", err)
	}
	time.Sleep(10 * time.Millisecond)

	// Save file of 400 bytes (total would be 1200 > 1000, should prune snap1 to fit)
	_, err = diskMgr.SaveSnapshot("snap3.tar", make([]byte, 400))
	if err != nil {
		t.Fatalf("save snap3: %v", err)
	}

	if diskMgr.RetainedBytes() > 1000 {
		t.Errorf("RetainedBytes() = %d, exceeds max 1000", diskMgr.RetainedBytes())
	}

	// Mutant control: single snapshot exceeding entire disk budget
	_, err = diskMgr.SaveSnapshot("huge.tar", make([]byte, 2000))
	if !errors.Is(err, cudaflightrecorder.ErrDiskBudgetExceeded) {
		t.Errorf("expected ErrDiskBudgetExceeded, got: %v", err)
	}
}

func TestClockUncertainty(t *testing.T) {
	// Configured uncertainty
	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
		Window:             5 * time.Second,
		Capacity:           10,
		ClockUncertaintyNs: 25000, // 25µs
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := rec.Start(); err != nil {
		t.Fatal(err)
	}
	defer rec.Stop()

	now := time.Now().UnixNano()
	rec.Push(cudaflightrecorder.Record{
		TimestampNs:   now - 1_000_000,
		KernelName:    "kernel_k",
		CorrelationID: 1,
		DurationNs:    500_000,
	})

	var buf bytes.Buffer
	manifest, err := rec.Snapshot(context.Background(), &buf, "clock_test")
	if err != nil {
		t.Fatal(err)
	}

	checkClockUncertainty(t, manifest, 25000)
}

func TestFaultSnapshotHostOnly(t *testing.T) {
	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
		Window:   5 * time.Second,
		Capacity: 50,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := rec.Start(); err != nil {
		t.Fatal(err)
	}
	defer rec.Stop()

	now := time.Now().UnixNano()
	rec.Push(cudaflightrecorder.Record{
		TimestampNs:   now - 1_000_000,
		KernelName:    "faulty_gemm_kernel",
		CorrelationID: 42,
		DurationNs:    150_000,
	})

	fault := cudaflightrecorder.FaultInfo{
		TimestampNs:      now,
		ErrorCode:        700,
		ErrorName:        "CUDA_ERROR_ILLEGAL_ADDRESS",
		ErrorMessage:     "an illegal memory access was encountered",
		OffendingKernel:  "faulty_gemm_kernel",
		CorrelationID:    42,
		ModuleID:         "mod-attn-01",
		ModuleDigest:     "a1b2c3d4e5f60718293a4b5c6d7e8f90",
		ModuleGeneration: 2,
		FailedRuleNames:  []string{"rule_vectorize_ld_128", "rule_tile_wmma_16x16"},
	}

	var buf bytes.Buffer
	manifest, err := rec.FaultSnapshot(context.Background(), &buf, fault)
	if err != nil {
		t.Fatalf("FaultSnapshot failed: %v", err)
	}

	if manifest.Reason != "fault:CUDA_ERROR_ILLEGAL_ADDRESS" {
		t.Errorf("Reason = %q, want fault:CUDA_ERROR_ILLEGAL_ADDRESS", manifest.Reason)
	}
	if manifest.ClockUncertaintySource != "host_only" {
		t.Errorf("ClockUncertaintySource = %q, want host_only", manifest.ClockUncertaintySource)
	}

	// Verify tar bundle contains fault.json, negative_rules.json, gpu_records.txt
	bundle, err := cudaflightrecorder.ReadBundle(&buf)
	if err != nil {
		t.Fatalf("ReadBundle on fault snapshot failed: %v", err)
	}

	for _, req := range []string{"fault.json", "negative_rules.json", "gpu_records.txt"} {
		if _, ok := bundle.Files[req]; !ok {
			t.Errorf("expected %s in fault snapshot bundle", req)
		}
	}

	if !strings.Contains(string(bundle.Files["negative_rules.json"]), "rule_vectorize_ld_128") {
		t.Errorf("negative_rules.json missing quarantined rule name")
	}
	if !strings.Contains(string(bundle.Files["fault.json"]), "CUDA_ERROR_ILLEGAL_ADDRESS") {
		t.Errorf("fault.json missing error name")
	}
}

func TestCollectorCleanupFailure(t *testing.T) {
	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
		Window:   1 * time.Second,
		Capacity: 10,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := rec.Start(); err != nil {
		t.Fatal(err)
	}

	// Simulate collector cleanup error
	cleanupErr := errors.New("cupti collector detach failure")
	rec.SetCleanupErr(cleanupErr)

	err = rec.Close()
	if !errors.Is(err, cleanupErr) {
		t.Errorf("Close() = %v, want %v", err, cleanupErr)
	}
	if !errors.Is(rec.CleanupErr(), cleanupErr) {
		t.Errorf("CleanupErr() = %v, want %v", rec.CleanupErr(), cleanupErr)
	}
}
