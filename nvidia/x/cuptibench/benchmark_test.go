package cuptibench

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cuptirange"
)

type benchmarkContext struct{}

func (benchmarkContext) Run(f func() error) error { return f() }
func (benchmarkContext) Release() error           { return nil }

type benchmarkCollector struct{ captures int }

func (c *benchmarkCollector) CaptureReplay(_ *cuda.Device, _ cuptirange.Limits, options cuptirange.CaptureOptions, r cuptirange.Replay) (*cuptirange.Capture, error) {
	c.captures++
	for pass := uint(0); pass < 2; pass++ {
		if err := r.Synchronize(); err != nil {
			return nil, err
		}
		if err := r.Prepare(pass); err != nil {
			return nil, err
		}
		if err := r.Synchronize(); err != nil {
			return nil, err
		}
		if err := r.Run(pass, &cuptirange.Scope{}); err != nil {
			return nil, err
		}
		if err := r.Synchronize(); err != nil {
			return nil, err
		}
	}
	info := cuptirange.RangeInfo{ID: "root", Name: "operation", Kind: "operation", Inclusive: true}
	if options.Mode == cuptirange.Kernels {
		info = cuptirange.RangeInfo{ID: "kernel1", Name: "kernel", Kind: "kernel"}
	}
	return &cuptirange.Capture{Image: []byte{1}, Ranges: 1, RangeInfo: []cuptirange.RangeInfo{info}, Quality: cuptirange.CollectionQuality{PassesStarted: 2, PassesStopped: 2, AllPassesSubmitted: true, Decoded: true}}, nil
}
func (*benchmarkCollector) Evaluate([]byte, uint) ([]float64, error) { return []float64{42}, nil }
func (*benchmarkCollector) Close() error                             { return nil }

type syntheticTimer struct {
	closed int
	fail   bool
}

func (t *syntheticTimer) Measure(f func() error) (time.Duration, error) {
	if err := f(); err != nil {
		return 0, err
	}
	return 18 * time.Microsecond, nil
}
func (t *syntheticTimer) Close() error {
	t.closed++
	if t.fail {
		return errors.New("timer cleanup")
	}
	return nil
}

func benchmarkFixture() (*Plan, Workload, *benchmarkCollector, *int, *int) {
	dev := &cuda.Device{}
	c := &benchmarkCollector{}
	p := &Plan{dev: dev, config: Config{Metrics: []string{"test_counter"}, Operations: 3, Mode: Whole, Limits: cuptirange.Limits{MaxRanges: 16, MaxRangeTreeNodes: 16, MaxPasses: 4, MaxCounterBytes: 1024}}, context: benchmarkContext{}, backend: c, units: []string{"count"}, metadata: map[string]string{"chip": "synthetic"}}
	prepared, runs := new(int), new(int)
	w := Workload{Streams: []*cuda.Stream{dev.DefaultStream()}, Preparation: "synthetic-control", Prepare: func() error { *prepared++; return nil }, Run: func(*Scope) error { *runs++; return nil }, Synchronize: func() error { return nil }}
	return p, w, c, prepared, runs
}

// This benchmark is run only by the subprocess acceptance test. It exercises the
// actual Go runner's sub-benchmark selection and fixed iteration semantics.
func BenchmarkCuptiHierarchy(b *testing.B) {
	if os.Getenv("CUPTIBENCH_RUNNER_CONTROL") != "1" {
		b.Skip("runner control")
	}
	for _, name := range []string{"full", "isolated/kernel", "attributed"} {
		b.Run(name, func(b *testing.B) {
			p, w, c, prepared, runs := benchmarkFixture()
			expectedCaptures := 1
			if name == "attributed" {
				p.config.Mode = Kernels
				expectedCaptures = 2
			}
			timer := &syntheticTimer{}
			r, err := benchmark(b, p, w, func(*Plan, Workload) (benchmarkTimer, error) { return timer, nil })
			if err != nil {
				b.Fatal(err)
			}
			if c.captures != expectedCaptures || *prepared != b.N+expectedCaptures*2 || *runs != (b.N+expectedCaptures*2)*3 || r.Timing.Operations != int64(b.N*3) || timer.closed != 1 {
				b.Fatalf("accounting capture=%d prepare=%d runs=%d N=%d timing=%+v close=%d", c.captures, *prepared, *runs, b.N, r.Timing, timer.closed)
			}
			if r.Operations != 3 || r.Ranges[0].Metrics[0].Value != 42 {
				b.Fatal("replay changed counter denominator")
			}
			if name == "attributed" && (r.Aggregate == nil || r.Aggregate.Metadata["mode"] != "whole" || r.Aggregate.Metadata["capture_id"] == r.Metadata["capture_id"]) {
				b.Fatal("missing separately identified aggregate")
			}
			if err := p.Close(); err != nil {
				b.Fatal(err)
			}
		})
	}
}

func TestBenchmarkSubbenchmarkSelection(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ name, pattern, want, absent string }{
		{"all", "^BenchmarkCuptiHierarchy$", "BenchmarkCuptiHierarchy/full", ""},
		{"isolated", "^BenchmarkCuptiHierarchy$/^isolated$/^kernel$", "BenchmarkCuptiHierarchy/isolated/kernel", "BenchmarkCuptiHierarchy/full-"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cmd := exec.Command(exe, "-test.run=^$", "-test.bench="+tc.pattern, "-test.benchtime=2x", "-test.count=2")
			cmd.Env = append(os.Environ(), "CUPTIBENCH_RUNNER_CONTROL=1")
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("runner: %v\n%s", err, out)
			}
			text := string(out)
			for _, want := range []string{tc.want, "gpu-ns/op", "profile-ops", "test_counter"} {
				if !strings.Contains(text, want) {
					t.Errorf("missing %s:\n%s", want, text)
				}
			}
			if tc.absent != "" && strings.Contains(text, tc.absent) {
				t.Fatalf("selected predecessor ran:\n%s", text)
			}
		})
	}
}

func TestMeasureSequenceFailureAndScope(t *testing.T) {
	failure := errors.New("launch")
	syncs := 0
	var saved *Scope
	w := Workload{Run: func(s *Scope) error { saved = s; return failure }, Synchronize: func() error { syncs++; return nil }}
	d, err := measureSequence(&syntheticTimer{}, w, 3)
	if d != 0 || !errors.Is(err, failure) || syncs != 1 {
		t.Fatalf("measure=%v,%v sync=%d", d, err, syncs)
	}
	if err := saved.Region("late", func() error { return nil }); err == nil {
		t.Fatal("scope escaped lifetime")
	}
}

func ExampleBenchmark() {
	// Register each independent workload in its own b.Run leaf. The application
	// supplies a Plan and Workload; Benchmark owns the iteration loop.
	fmt.Println("BenchmarkDecode/full")
	fmt.Println("BenchmarkDecode/isolated/layer00/attention")
	// Output:
	// BenchmarkDecode/full
	// BenchmarkDecode/isolated/layer00/attention
}
