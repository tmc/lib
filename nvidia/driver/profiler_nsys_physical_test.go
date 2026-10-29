package driver_test

import (
	"bufio"
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

// nsysChildEnv selects the child mode of TestNsysProfilerStopChild: "stop"
// calls ProfilerStop before the process exits, "exit" does not.
const nsysChildEnv = "DRIVER_NSYS_PROFILER_STOP_CHILD"

// nsysMemsets is how many memsets the child issues.
const nsysMemsets = 16

// TestNsysProfilerStop runs memsets under Nsight Systems in a child copy of
// this test binary, once calling ProfilerStop before exit and once not, and
// counts the memsets in each report. Nsight Systems flushes its CUDA trace
// from a C atexit handler, which a Go exit skips, so the report of the run
// without the stop holds no memsets; that run is the control showing the
// count can be zero.
func TestNsysProfilerStop(t *testing.T) {
	if os.Getenv("CUDA_DRIVER_HARDWARE_TEST") != "1" {
		t.Skip("set CUDA_DRIVER_HARDWARE_TEST=1 for native driver validation")
	}
	nsys, err := exec.LookPath("nsys")
	if err != nil {
		t.Skipf("no Nsight Systems: %v", err)
	}
	if err := driver.Init(0); err != nil {
		t.Skipf("no CUDA driver: %v", err)
	}
	if _, err := driver.DeviceGet(0); err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	for _, tt := range []struct {
		mode string
		want func(n int) bool
		desc string
	}{
		{"stop", func(n int) bool { return n == nsysMemsets }, fmt.Sprint(nsysMemsets)},
		{"exit", func(n int) bool { return n == 0 }, "0"},
	} {
		t.Run(tt.mode, func(t *testing.T) {
			rep := filepath.Join(t.TempDir(), "r")
			cmd := exec.Command(nsys, "profile", "--trace=cuda-sw", "-o", rep, "-f", "true",
				os.Args[0], "-test.run=^TestNsysProfilerStopChild$", "-test.v")
			cmd.Env = append(os.Environ(), nsysChildEnv+"="+tt.mode)
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("nsys profile: %v\n%s", err, out)
			}
			// A child that skipped or failed would make an empty report
			// look like the exit run's.
			if !bytes.Contains(out, []byte("--- PASS: TestNsysProfilerStopChild")) {
				t.Fatalf("child did not pass:\n%s", out)
			}
			stats, err := exec.Command(nsys, "stats", "--report", "cuda_gpu_mem_time_sum",
				"--format", "csv", "-o", "-", rep+".nsys-rep").CombinedOutput()
			if err != nil {
				t.Fatalf("nsys stats: %v\n%s", err, stats)
			}
			n, err := memsetCount(stats)
			if err != nil {
				t.Fatal(err)
			}
			if !tt.want(n) {
				t.Errorf("%s run: report holds %d memsets, want %s\n%s", tt.mode, n, tt.desc, stats)
			}
		})
	}
}

// memsetCount returns the Count column of the "[CUDA memset]" row of an
// nsys cuda_gpu_mem_time_sum CSV report, or 0 if there is no such row.
func memsetCount(report []byte) (int, error) {
	sc := bufio.NewScanner(bytes.NewReader(report))
	for sc.Scan() {
		f := strings.Split(sc.Text(), ",")
		if len(f) < 9 || f[len(f)-1] != "[CUDA memset]" {
			continue
		}
		n, err := strconv.Atoi(f[2])
		if err != nil {
			return 0, fmt.Errorf("memset count %q: %w", f[2], err)
		}
		return n, nil
	}
	return 0, sc.Err()
}

func TestMemsetCount(t *testing.T) {
	for _, tt := range []struct {
		name, report string
		want         int
	}{
		{"memset", "Time (%),Total Time (ns),Count,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Operation\n" +
			"81.1,408384,1850,220.7,192.0,160,4672,226.3,[CUDA memcpy Host-to-Device]\n" +
			"18.9,95008,16,1900.2,608.0,576,6624,2234.6,[CUDA memset]\n", 16},
		{"memcpy only", "Time (%),Total Time (ns),Count,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Operation\n" +
			"100.0,408384,1850,220.7,192.0,160,4672,226.3,[CUDA memcpy Host-to-Device]\n", 0},
		{"no memory data", "SKIPPED: r.sqlite does not contain GPU memory data.\n", 0},
	} {
		got, err := memsetCount([]byte(tt.report))
		if err != nil || got != tt.want {
			t.Errorf("%s: memsetCount = %d, %v; want %d", tt.name, got, err, tt.want)
		}
	}
}

// TestNsysProfilerStopChild is the process TestNsysProfilerStop profiles.
func TestNsysProfilerStopChild(t *testing.T) {
	mode := os.Getenv(nsysChildEnv)
	if mode == "" {
		t.Skip("run by TestNsysProfilerStop under nsys")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := driver.Init(0); err != nil {
		t.Fatal(err)
	}
	dev, err := driver.DeviceGet(0)
	if err != nil {
		t.Fatal(err)
	}
	ctx, err := dev.PrimaryCtxRetain()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.PrimaryCtxRelease()
	if err := ctx.SetCurrent(); err != nil {
		t.Fatal(err)
	}
	p, err := driver.MemAlloc[byte](1 << 20)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Free()
	for i := range nsysMemsets {
		if err := p.MemsetD8(uint8(i), 1<<20); err != nil {
			t.Fatal(err)
		}
	}
	if err := driver.CtxSynchronize(); err != nil {
		t.Fatal(err)
	}
	if mode == "stop" {
		if err := driver.ProfilerStop(); err != nil {
			t.Fatal(err)
		}
	}
}
