package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
	"github.com/tmc/lib/nvidia/x/internal/cudatest"
)

// The tests in this file check the kernel durations cudaprof reports against
// a kernel whose length is known without a profiler, testdata/spin.cu. It
// spins for a fixed number of SM clock cycles and reads the device's global
// timer at its first and last instruction.
//
// They need a CUDA device and run only when CUDAPROF_GPU_VALIDATE=1. With the
// variable set they fail, never skip, when the host cannot run them: a skipped
// test reports ok, and a validation that did not run must not read as one
// that passed.

// spinTools is the spin program and the agent that captures it, built for
// the host's device and toolkit.
type spinTools struct {
	dir, spin, agent string
}

// buildSpinTools builds testdata/spin.cu and the CUPTI agent, or skips the
// test when GPU validation is off.
func buildSpinTools(t *testing.T) spinTools {
	t.Helper()
	if os.Getenv("CUDAPROF_GPU_VALIDATE") != "1" {
		t.Skip("GPU validation off: set CUDAPROF_GPU_VALIDATE=1 on a CUDA host, under gpu-lock")
	}
	if runtime.GOOS != "linux" {
		t.Fatalf("CUDAPROF_GPU_VALIDATE=1 on %s: capture needs Linux with CUPTI", runtime.GOOS)
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("CUDAPROF_GPU_VALIDATE=1 but GPU_LOCK_HELD is unset: run under gpu-lock")
	}
	nvcc, _ := cudatest.NVCC(t)
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(home, "tmp"), 0o755); err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "cudaprof-validate-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })

	tools := spinTools{dir: dir, spin: filepath.Join(dir, "spin"), agent: filepath.Join(dir, "cupti_agent.so")}
	// -arch=native compiles for the device nvcc finds, so a host without one
	// fails here rather than later.
	if out, err := exec.Command(nvcc, "-O2", "-arch=native", "-o", tools.spin, filepath.Join("testdata", "spin.cu")).CombinedOutput(); err != nil {
		t.Fatalf("nvcc spin.cu: %v\n%s", err, out)
	}
	build := exec.Command("sh", filepath.Join("..", "..", "..", "internal", "cudainject", "native", "build_agent.sh"), dir)
	build.Env = append(os.Environ(), "CUDA_HOME="+filepath.Dir(filepath.Dir(nvcc)))
	if out, err := build.CombinedOutput(); err != nil {
		if bytes.Contains(out, []byte("cupti_agent needs CUPTI 12.8 or newer")) {
			t.Skipf("toolkit %s: cupti_agent needs CUPTI 12.8 or newer", filepath.Dir(filepath.Dir(nvcc)))
		}
		t.Fatalf("build agent: %v\n%s", err, out)
	}
	return tools
}

// spinRun is what spin prints: the host wall time of its profiler range and,
// per launch, the cycles it spun and its global timer interval.
type spinRun struct {
	Mode     string `json:"mode"`
	Cycles   int64  `json:"cycles"`
	Replays  int    `json:"replays"`
	RegionNS int64  `json:"region_ns"`
	Launches []struct {
		Cycles int64  `json:"cycles"`
		Start  uint64 `json:"start"`
		End    uint64 `json:"end"`
	} `json:"launches"`
}

// captureSpin runs spin with args under cudaprof capture with the given
// capture flags, and returns the capture and what spin printed.
func captureSpin(t *testing.T, tools spinTools, flags []string, args ...string) (string, spinRun) {
	t.Helper()
	out := filepath.Join(tools.dir, strings.Join(append(slices.Clone(args), flags...), "")+".pprof")
	argv := append([]string{"--inject", tools.agent, "-o", out}, flags...)
	argv = append(argv, "--", tools.spin)
	opts, err := parseCaptureArgs(append(argv, args...))
	if err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	opts.Stdout, opts.Stderr = &stdout, &stderr
	opts.Timeout = 2 * time.Minute
	result, err := runCapture(context.Background(), opts)
	if err != nil {
		t.Fatalf("capture spin %v: %v\n%s", args, err, stderr.Bytes())
	}
	if result.TargetExitCode != 0 || !result.CaptureSuccess {
		t.Fatalf("capture spin %v %v: exit %d, capture_success=%v: %s (outcome %s)\n%s", flags, args, result.TargetExitCode, result.CaptureSuccess, result.CaptureError, result.Outcome, stderr.Bytes())
	}
	if err := writeCaptureStatus(out, result, captureArtifactBudget{}); err != nil {
		t.Fatal(err)
	}
	var run spinRun
	if err := json.Unmarshal(stdout.Bytes(), &run); err != nil {
		t.Fatalf("read spin output: %v\n%s", err, stdout.Bytes())
	}
	return out, run
}

// measureSpin measures the capture's profiler range as compare does and
// returns the side and the per-launch times of its one op, in start order.
func measureSpin(t *testing.T, path string) (*CompareSide, []int64) {
	t.Helper()
	cap, err := readCapturePath(path)
	if err != nil {
		t.Fatal(err)
	}
	side, err := measureCompareSide(cap, nil, nil, compareOptions{phase: -1})
	if err != nil {
		t.Fatalf("measure capture: %v", err)
	}
	if side.Window != "phase" {
		t.Fatalf("window is %q (%s), want the profiler range", side.Window, side.WindowNote)
	}
	if len(side.launches) != 1 || side.launches["spin"] == nil {
		t.Fatalf("ops in range = %v, want spin alone", mapKeys(side.launches))
	}
	return side, side.launches["spin"].exclusive
}

// spinRecords decodes a capture's journals.
func spinRecords(t *testing.T, path string) timeline.Records {
	t.Helper()
	cap, err := readCapturePath(path)
	if err != nil {
		t.Fatal(err)
	}
	journals, err := requireJournals(cap)
	if err != nil {
		t.Fatal(err)
	}
	records, err := timeline.Decode(journals, controllerMaxJournalRecords, cudacapture.MaxCaptureInputBytes)
	if err != nil {
		t.Fatal(err)
	}
	return records
}

func mapKeys[V any](m map[string]V) []string {
	var keys []string
	for k := range m {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	return keys
}

// edgeSMClock returns the SM clock NVML read at the range's start and stop.
func edgeSMClock(t *testing.T, side *CompareSide) (start, stop uint32) {
	t.Helper()
	if side.Clocks == nil || side.Clocks.NotRecorded != "" {
		t.Fatalf("no SM clock at the range edges: %+v", side.Clocks)
	}
	start, ok1 := side.Clocks.Start.smClock()
	stop, ok2 := side.Clocks.Stop.smClock()
	if !ok1 || !ok2 {
		t.Fatalf("SM clock unread at a range edge: start %+v, stop %+v", side.Clocks.Start, side.Clocks.Stop)
	}
	return start, stop
}

// rangeFlag collects only spin's profiler range, so the warmup is left out
// and the range edges carry NVML's clock readings.
const rangeFlag = "--capture-range=" + captureRangeProfilerAPI

// Tolerances for a kernel's reported duration against the global timer
// interval the kernel read itself, from the measurements in PRODUCTION.md
// ("Validation results"). On GB10 the report is longer than the interval
// by a fixed amount, not a fraction: median 736 ns at every length from
// 4 us to 4 ms (576-608 ns for graph nodes), never less than 544 ns. The
// interval starts after the kernel is launched and ends before it retires.
// nsys -t cuda-sw reports the same 736 ns.
//
// Some launches are reported as much as 165 us longer while the interval is
// unchanged: CUPTI's kernel start and end bracket a delay the kernel's own
// instructions do not see. They come in bursts of up to 126 consecutive
// launches within a few milliseconds, plus single launches every 2735 that
// fall at the same index in every run, so a limit per thousand launches
// fails on one burst. nsys -t cuda-sw, with its own CUPTI and no cudaprof
// code, reports the same bursts and the same indices. The worst run seen
// had 3.7% of its launches over timerSlackNS.
const (
	// timerSlackNS bounds the median of reported minus interval, and
	// marks a launch over it as an outlier.
	timerSlackNS = 2000
	// outlierPercent is how many launches per hundred may exceed
	// timerSlackNS, plus one.
	outlierPercent = 5
)

// checkLaunches compares reported per-launch durations with spin's global
// timer intervals, pairwise. A report shorter than the interval is always
// wrong: the kernel ran at least that long. A median over timerSlackNS is a
// shift in every report, which no burst explains.
func checkLaunches(t *testing.T, got []int64, run spinRun) {
	t.Helper()
	if len(got) != len(run.Launches) {
		t.Fatalf("%d reported launches for %d spin launches", len(got), len(run.Launches))
	}
	var diff, timer, reported []float64
	outliers, bursts := 0, 0
	var last uint64 // start of the previous outlier
	for i, l := range run.Launches {
		span := int64(l.End - l.Start)
		d := got[i] - span
		diff = append(diff, float64(d))
		timer = append(timer, float64(span))
		reported = append(reported, float64(got[i]))
		switch {
		case d < 0:
			t.Errorf("launch %d: reported %d ns, shorter than its global timer interval %d ns", i, got[i], span)
		case d > timerSlackNS:
			// Outliers less than 1 ms apart are one burst.
			if outliers == 0 || l.Start-last >= 1e6 {
				bursts++
			}
			outliers++
			last = l.Start
			t.Logf("launch %d: reported %d ns, global timer %d ns: %d ns over at %d", i, got[i], span, d, l.Start)
		}
	}
	if m := median(diff); m > timerSlackNS {
		t.Errorf("median reported minus global timer interval is %.0f ns, want at most %d", m, timerSlackNS)
	}
	if limit := 1 + len(got)*outlierPercent/100; outliers > limit {
		t.Errorf("%d of %d launches reported more than %d ns over their global timer interval, want at most %d", outliers, len(got), timerSlackNS, limit)
	}
	t.Logf("reported ns     %s", summary(reported))
	t.Logf("global timer ns %s", summary(timer))
	t.Logf("reported-timer  %s; %d over %d ns in %d bursts", summary(diff), outliers, timerSlackNS, bursts)
}

func TestSpinDurations(t *testing.T) {
	tools := buildSpinTools(t)
	tests := []struct {
		cycles   int64
		launches int
	}{
		{1e4, 1},
		{1e4, 100},
		{1e4, 10000},
		{1e5, 1000},
		{1e6, 200},
		{1e7, 20},
	}
	for _, tt := range tests {
		t.Run(fmt.Sprintf("%dx%d", tt.cycles, tt.launches), func(t *testing.T) {
			path, run := captureSpin(t, tools, []string{rangeFlag}, "stream", strconv.FormatInt(tt.cycles, 10), strconv.Itoa(tt.launches))
			side, got := measureSpin(t, path)
			if len(got) != tt.launches || len(run.Launches) != tt.launches {
				t.Fatalf("cudaprof reports %d launches, spin reports %d; want %d", len(got), len(run.Launches), tt.launches)
			}
			checkLaunches(t, got, run)

			// cycles / clock is logged, not checked: NVML's SM clock at the
			// edges reads 1-8% below the rate clock64 advances at against
			// the global timer, and for the one-launch range it has read
			// 227 and 312 MHz while the kernel ran at 2.4-2.6 GHz.
			start, stop := edgeSMClock(t, side)
			var ratio, rate []float64
			for i, l := range run.Launches {
				ratio = append(ratio, float64(got[i])/(float64(l.Cycles)*1e3/float64(start)))
				rate = append(rate, float64(l.Cycles)*1e3/float64(l.End-l.Start))
			}
			t.Logf("SM clock %d MHz at start, %d at stop; clock64 rate %s MHz", start, stop, summary(rate))
			t.Logf("reported/(cycles/clock) %s", summaryf(ratio, "%.4f"))
		})
	}
}

// TestSpinGraphNodes replays a graph of spin launches. In the default graph
// trace mode each node is a kernel record carrying its graph; in the
// envelope mode compare refuses a per-op census rather than report one
// without the nodes.
func TestSpinGraphNodes(t *testing.T) {
	tools := buildSpinTools(t)
	const nodes, replays = 10, 5
	args := []string{"graph", "100000", strconv.Itoa(nodes), strconv.Itoa(replays)}

	path, run := captureSpin(t, tools, []string{rangeFlag}, args...)
	side, got := measureSpin(t, path)
	if len(got) != nodes*replays {
		t.Fatalf("node mode: %d spin launches, want %d nodes x %d replays", len(got), nodes, replays)
	}
	if side.GraphTraceMode != graphTraceNode || side.UnresolvedNanos != 0 {
		t.Errorf("node mode: graph trace mode %q, %d ns unresolved; want node and 0", side.GraphTraceMode, side.UnresolvedNanos)
	}
	for _, a := range spinRecords(t, path).Activities {
		if a.Subtype == "kernel" && a.GraphID == 0 {
			t.Errorf("node mode: kernel %s at %d carries no graph id", a.Name, a.Start)
		}
	}
	// spin records the nodes of its last replay.
	checkLaunches(t, got[len(got)-nodes:], run)

	// The envelope mode collects nothing inside a profiler range (see
	// PRODUCTION.md), so this capture is of the whole run.
	path, _ = captureSpin(t, tools, []string{"--graph-trace=graph"}, args...)
	cap, err := readCapturePath(path)
	if err != nil {
		t.Fatal(err)
	}
	_, refusal := measureCompareSide(cap, nil, nil, compareOptions{phase: -1})
	if refusal == nil || !strings.Contains(refusal.Error(), "graph replay") {
		t.Errorf("graph mode: measure error = %v, want a refusal naming graph replay", refusal)
	}
	records := spinRecords(t, path)
	if len(records.Graphs) != replays {
		t.Errorf("graph mode: %d replay envelopes, want %d", len(records.Graphs), replays)
	}
	for _, a := range records.Activities {
		if a.Subtype == "kernel" && a.Name == "spin" {
			t.Errorf("graph mode: spin kernel record at %d; the envelope mode hides the nodes", a.Start)
			break
		}
	}
	t.Logf("graph mode: %d envelopes; compare: %v", len(records.Graphs), refusal)
}

func summary(v []float64) string { return summaryf(v, "%.0f") }

// summaryf formats the minimum, median and maximum of v.
func summaryf(v []float64, format string) string {
	return fmt.Sprintf("min "+format+" median "+format+" max "+format, slices.Min(v), median(v), slices.Max(v))
}
