package cudaready

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"testing"
	"time"
)

// TestHostRace runs testdata/ready_test.cc, which drives ready.cuh with
// host threads standing in for blocks: 300 epochs of producers that start
// in random order after random delays, with counters never reset. Every
// slot a consumer reads must hold its epoch's value. Built with NO_WAIT,
// the consumer skips the wait, and the test must then see stale slots, or
// it could not see a broken wait either. It checks the counting, not
// memory ordering: dropping a fence does not make it fail.
func TestHostRace(t *testing.T) {
	cxx, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("no c++")
	}
	tests := []struct {
		name   string
		flags  []string
		ok     bool
		stales func(n int) bool
	}{
		{"wait", nil, true, func(n int) bool { return n == 0 }},
		{"no wait", []string{"-DNO_WAIT"}, false, func(n int) bool { return n > 0 }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bin := filepath.Join(t.TempDir(), "ready_test")
			args := append([]string{"-std=c++20", "-O2", "-pthread", "-o", bin}, tt.flags...)
			args = append(args, filepath.Join("testdata", "ready_test.cc"))
			if out, err := exec.Command(cxx, args...).CombinedOutput(); err != nil {
				t.Fatalf("c++: %v\n%s", err, out)
			}
			out, err := exec.Command(bin).CombinedOutput()
			if (err == nil) != tt.ok {
				t.Fatalf("ran with err = %v, want ok %v\n%s", err, tt.ok, out)
			}
			m := regexp.MustCompile(`(\d+) slots checked, (\d+) stale`).FindSubmatch(out)
			if m == nil {
				t.Fatalf("no tally in output:\n%s", out)
			}
			stale, _ := strconv.Atoi(string(m[2]))
			if !tt.stales(stale) {
				t.Errorf("%s slots checked, %d stale", m[1], stale)
			}
			t.Logf("%s", out)
		})
	}
}

// TestDeviceLitmus runs testdata/ready_litmus.cu, a message-passing
// litmus test, on the first CUDA device: producer blocks write a payload
// and signal, consumer blocks wait and must read every word of it. It is
// the only test of ready.cuh's memory ordering, and it skips without nvcc
// or a device. Built with NO_WAIT the consumers must read stale words;
// built with NO_FENCE they may, so that run is only logged.
func TestDeviceLitmus(t *testing.T) {
	nvcc, err := exec.LookPath("nvcc")
	if err != nil {
		nvcc = filepath.Join(os.Getenv("CUDA_HOME"), "bin", "nvcc")
		if _, err := os.Stat(nvcc); os.Getenv("CUDA_HOME") == "" || err != nil {
			t.Skip("no nvcc")
		}
	}
	tests := []struct {
		name   string
		flags  []string
		stales func(n int) bool // nil: report only
	}{
		{"wait", nil, func(n int) bool { return n == 0 }},
		{"no wait", []string{"-DNO_WAIT"}, func(n int) bool { return n > 0 }},
		{"no fence", []string{"-DNO_FENCE"}, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bin := filepath.Join(t.TempDir(), "ready_litmus")
			// PTX for compute_80, which CUDA 12 and 13 both build,
			// runs on any device since Ampere.
			args := append([]string{"-std=c++17", "-O2", "-arch=compute_80", "-o", bin}, tt.flags...)
			args = append(args, filepath.Join("testdata", "ready_litmus.cu"))
			if out, err := exec.Command(nvcc, args...).CombinedOutput(); err != nil {
				t.Fatalf("nvcc: %v\n%s", err, out)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			defer cancel()
			out, err := exec.CommandContext(ctx, bin).CombinedOutput()
			var exit *exec.ExitError
			switch {
			case ctx.Err() != nil:
				t.Fatalf("timed out: a wait never returned\n%s", out)
			case errors.As(err, &exit) && exit.ExitCode() == 77:
				t.Skipf("%s", out)
			}
			m := regexp.MustCompile(`(\d+) words checked, (\d+) stale`).FindSubmatch(out)
			if m == nil {
				t.Fatalf("no tally in output (%v):\n%s", err, out)
			}
			stale, _ := strconv.Atoi(string(m[2]))
			t.Logf("%s", out)
			if tt.stales != nil && !tt.stales(stale) {
				t.Errorf("%s words checked, %d stale", m[1], stale)
			}
		})
	}
}
