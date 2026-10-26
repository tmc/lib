package gpucampaign

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestRun(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("needs /bin/sh")
	}
	sh := func(script string) []string { return []string{"/bin/sh", "-c", script} }
	tests := []struct {
		name       string
		budget     time.Duration
		phase      Phase
		status     Status
		code       int
		timedOut   bool
		reasonPart string
	}{
		{"pass", time.Minute, Phase{Name: "p", Argv: sh("exit 0"), Cap: 10 * time.Second}, Pass, 0, false, ""},
		{"fail", time.Minute, Phase{Name: "p", Argv: sh("exit 3"), Cap: 10 * time.Second}, Hold, 3, false, "exit status 3"},
		{"cap does not fit", time.Second, Phase{Name: "p", Argv: sh("exit 0"), Cap: time.Minute}, NotRun, -1, false, "does not fit"},
		{"timeout", time.Minute, Phase{Name: "p", Argv: sh("sleep 30"), Cap: 200 * time.Millisecond}, Hold, -1, true, "process group killed"},
		{"missing program", time.Minute, Phase{Name: "p", Argv: []string{"/nonexistent/x"}, Cap: time.Second}, Hold, -1, false, "no such file"},
		{"empty", time.Minute, Phase{Name: "p", Cap: time.Second}, Hold, -1, false, "empty command"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := NewRunner(tt.budget)
			got := r.Run(context.Background(), tt.phase)
			if got.Status != tt.status || got.ExitCode != tt.code || got.TimedOut != tt.timedOut || !strings.Contains(got.Reason, tt.reasonPart) {
				t.Errorf("Run() = %+v, want status %s code %d timedOut %v reason %q", got, tt.status, tt.code, tt.timedOut, tt.reasonPart)
			}
		})
	}
}

// TestRunKillsDescendants checks that a timed-out phase's children die with
// it: a grandchild that outlived its parent would keep running, and holding
// the GPU, after the campaign moved on.
func TestRunKillsDescendants(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("process groups")
	}
	dir := t.TempDir()
	mark := filepath.Join(dir, "survivor")
	r := NewRunner(time.Minute)
	start := time.Now()
	got := r.Run(context.Background(), Phase{
		Name: "p",
		Argv: []string{"/bin/sh", "-c", "(sleep 1; touch " + mark + ") & sleep 30"},
		Cap:  200 * time.Millisecond,
	})
	if !got.TimedOut {
		t.Fatalf("Run() = %+v, want timeout", got)
	}
	if d := time.Since(start); d > 5*time.Second {
		t.Errorf("Run() took %v after its cap", d)
	}
	time.Sleep(1500 * time.Millisecond)
	if _, err := os.Stat(mark); err == nil {
		t.Error("grandchild survived the phase's cap")
	}
}

func TestRunStopsGPUPhasesAfterHealthFailure(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("needs /bin/sh")
	}
	healthy := true
	r := NewRunner(time.Minute)
	r.Health = func(context.Context) error {
		if healthy {
			return nil
		}
		return errors.New("/dev/nvidia0: open denied")
	}
	fail := Phase{Name: "gpu", Argv: []string{"/bin/sh", "-c", "exit 1"}, Cap: 10 * time.Second, GPU: true}
	ok := Phase{Name: "gpu", Argv: []string{"/bin/sh", "-c", "exit 0"}, Cap: 10 * time.Second, GPU: true}
	cpu := Phase{Name: "cpu", Argv: []string{"/bin/sh", "-c", "exit 0"}, Cap: 10 * time.Second}

	// A failure with a healthy GPU is the phase's own.
	if got := r.Run(context.Background(), fail); got.Status != Hold || r.Lost() != "" {
		t.Fatalf("healthy failure = %+v, lost %q", got, r.Lost())
	}
	if got := r.Run(context.Background(), ok); got.Status != Pass {
		t.Fatalf("after healthy failure = %+v", got)
	}
	healthy = false
	got := r.Run(context.Background(), fail)
	if got.Status != Hold || !strings.Contains(got.Reason, "GPU health failed: /dev/nvidia0: open denied") {
		t.Fatalf("unhealthy failure = %+v", got)
	}
	if got := r.Run(context.Background(), ok); got.Status != NotRun || !strings.Contains(got.Reason, "GPU lost earlier") {
		t.Errorf("GPU phase after loss = %+v, want NOT_RUN", got)
	}
	if got := r.Run(context.Background(), cpu); got.Status != Pass {
		t.Errorf("CPU phase after loss = %+v, want PASS", got)
	}
}

func TestDetectTerminalCUDAError(t *testing.T) {
	tests := []struct {
		name   string
		output string
		want   string
	}{
		{
			name:   "clean JSON output",
			output: `{"case":"attn","us_median":21.36,"status":"PASS"}`,
			want:   "",
		},
		{
			name:   "non-JSON harmless output",
			output: "building extension...\nwrote submission.py\n",
			want:   "",
		},
		{
			name:   "quoted source metadata ignored",
			output: `{"status":"PASS","source_code":"// CUDA error: an illegal memory access was encountered","expected_errors":["cudaErrorIllegalAddress"]}`,
			want:   "",
		},
		{
			name: "A5 mixed JSON stdout then illegal memory access traceback",
			output: "{\"case\": \"fa3\", \"batch\": 1, \"ctx\": 80000, \"us_median\": 21.36}\n" +
				"Traceback (most recent call last):\n" +
				"  File \"fa3_time.py\", line 37, in main\n" +
				"    torch.cuda.synchronize()\n" +
				"torch.AcceleratorError: CUDA error: an illegal memory access was encountered\n" +
				"Returning 700 (CUDA_ERROR_ILLEGAL_ADDRESS) from cuCtxSynchronize_v2\n",
			want: "CUDA error: an illegal memory access was encountered",
		},
		{
			name:   "device-side assert",
			output: "RuntimeError: CUDA error: device-side assert triggered\n",
			want:   "device-side assert",
		},
		{
			name:   "Xid in kernel log",
			output: "[12345.67] NVRM: Xid (PCI:0000:01:00): 31, pid=42, Ch 00000001\n",
			want:   "NVRM: Xid",
		},
		{
			name:   "lost device in JSON receipt",
			output: `{"status":"HOLD","device_health":{"initialization_error":"RuntimeError: No CUDA GPUs are available","error":"RuntimeError: CUDA unavailable"}}`,
			want:   "No CUDA GPUs are available",
		},
		{
			name:   "CUPTI_ERROR_UNKNOWN after device loss",
			output: `{"capture_success":false,"capture_error":"cupti_unavailable: cuptiActivityRegisterCallbacks: CUPTI_ERROR_UNKNOWN"}`,
			want:   "CUPTI_ERROR_UNKNOWN",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DetectTerminalCUDAError([]byte(tt.output))
			if tt.want == "" {
				if got != "" {
					t.Errorf("DetectTerminalCUDAError() = %q, want empty", got)
				}
				return
			}
			if !strings.Contains(got, tt.want) {
				t.Errorf("DetectTerminalCUDAError() = %q, want substring %q", got, tt.want)
			}
		})
	}
}

func TestTerminalCUDAErrorLatchesRunnerDespiteHealthyProbeAndZeroExit(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("needs /bin/sh")
	}
	r := NewRunner(time.Minute)
	// Post-failure fresh health probe returns nil (healthy), as in A5 when a
	// fresh subprocess re-initialized CUDA after the previous worker faulted.
	r.Health = func(context.Context) error { return nil }

	// Even if a wrapper script exits 0 after logging a terminal CUDA error,
	// Run must detect it before trusting exit status, mark the phase HOLD,
	// and latch subsequent GPU phases into NOT_RUN.
	fault := Phase{
		Name: "fa3-time",
		Argv: []string{"/bin/sh", "-c", `echo '{"case":"fa3"}'; echo 'torch.AcceleratorError: CUDA error: an illegal memory access was encountered' >&2; exit 0`},
		Cap:  10 * time.Second,
		GPU:  true,
	}
	got := r.Run(context.Background(), fault)
	if got.Status != Hold {
		t.Fatalf("fault phase Status = %s, want %s (res=%+v)", got.Status, Hold, got)
	}
	if !strings.Contains(got.Reason, "terminal CUDA error") || !strings.Contains(got.TerminalError, "illegal memory access") {
		t.Fatalf("fault phase Reason = %q, TerminalError = %q", got.Reason, got.TerminalError)
	}
	if r.Lost() == "" {
		t.Fatal("Runner.Lost() is empty after terminal CUDA error")
	}

	nextGPU := Phase{Name: "next-gpu", Argv: []string{"/bin/sh", "-c", "exit 0"}, Cap: 10 * time.Second, GPU: true}
	if got2 := r.Run(context.Background(), nextGPU); got2.Status != NotRun || !strings.Contains(got2.Reason, "GPU lost earlier") {
		t.Errorf("subsequent GPU phase = %+v, want NOT_RUN with GPU lost earlier", got2)
	}

	cpuPhase := Phase{Name: "archive-cpu", Argv: []string{"/bin/sh", "-c", "exit 0"}, Cap: 10 * time.Second, GPU: false}
	if got3 := r.Run(context.Background(), cpuPhase); got3.Status != Pass {
		t.Errorf("subsequent CPU phase = %+v, want PASS", got3)
	}
}

func TestDeadlineWriteCheckpointAndLateReturn(t *testing.T) {
	dir := t.TempDir()
	cpPath := filepath.Join(dir, "checkpoint.json")

	now := time.Date(2026, 10, 23, 12, 0, 0, 0, time.UTC)
	r := &Runner{
		deadline: now.Add(180 * time.Second),
		now:      func() time.Time { return now },
	}

	// 1. On-time checkpoint succeeds.
	now = now.Add(30 * time.Second)
	stage1 := Result{Name: "parity", Status: Pass, ExitCode: 0}
	if err := r.WriteCheckpoint(cpPath, stage1); err != nil {
		t.Fatalf("on-time WriteCheckpoint: %v", err)
	}

	// 2. A checkpoint whose write/sync crosses the monotonic deadline (181 > 180)
	// must fail and roll back to stage1 rather than committing stage2.
	calls := 0
	baseNow := now
	r.now = func() time.Time {
		calls++
		if calls == 1 {
			return baseNow // before deadline at entry
		}
		return baseNow.Add(151 * time.Second) // 181s > 180s after write/sync
	}
	stage2 := Result{Name: "late-timing", Status: Pass, ExitCode: 0}
	if err := r.WriteCheckpoint(cpPath, stage2); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("late-sync WriteCheckpoint error = %v, want context.DeadlineExceeded", err)
	}
	data, err := os.ReadFile(cpPath)
	if err != nil {
		t.Fatal(err)
	}
	var committed Result
	if err := json.Unmarshal(data, &committed); err != nil {
		t.Fatal(err)
	}
	if committed.Name != "parity" {
		t.Errorf("checkpoint after late sync = %+v, want rollback to parity", committed)
	}

	// 3. A checkpoint attempted after the deadline has already expired is
	// refused before touching disk.
	freshPath := filepath.Join(dir, "never.json")
	if err := r.WriteCheckpoint(freshPath, stage2); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expired WriteCheckpoint error = %v, want context.DeadlineExceeded", err)
	}
	if _, err := os.Stat(freshPath); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("expired WriteCheckpoint created %s", freshPath)
	}
}

func TestWriteJSON(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "receipt.json")
	for i, v := range []Result{{Name: "a", Status: Pass}, {Name: "b", Status: Hold}} {
		if err := WriteJSON(p, v); err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		var got Result
		if err := json.Unmarshal(data, &got); err != nil || got.Name != v.Name {
			t.Errorf("write %d: read back %+v, %v", i, got, err)
		}
	}
	entries, _ := os.ReadDir(dir)
	if len(entries) != 1 {
		t.Errorf("directory holds %d entries, want only the receipt", len(entries))
	}
	if err := WriteJSON(filepath.Join(dir, "missing", "r.json"), 1); err == nil {
		t.Error("WriteJSON into a missing directory succeeded")
	}
}
