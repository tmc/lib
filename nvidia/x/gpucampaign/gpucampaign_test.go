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
