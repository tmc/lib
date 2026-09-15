package cuptibench

import (
	"crypto/rand"
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
)

type benchmarkTimer interface {
	Measure(func() error) (time.Duration, error)
	Close() error
}

type timerFactory func(*Plan, Workload) (benchmarkTimer, error)

// Benchmark measures w in a leaf benchmark, then collects a bounded counter
// sample. It owns b.Loop; do not call it inside another iteration loop or
// RunParallel. Go ns/op measures a sequence of Config.Operations, including
// submission and completion waits. gpu-ns/op is per logical operation. Counter
// values describe the separately replayed sequence; profile-ops names its size.
// Setup, preparation, and profiling are excluded from the timed loop. Errors
// fail the benchmark. Use Plan.Collect for counter assertions in ordinary tests.
func Benchmark(b *testing.B, p *Plan, w Workload) *Result {
	b.Helper()
	selected, owned, err := benchmarkPlan(p)
	if err != nil {
		b.Fatal(err)
		return nil
	}
	if owned {
		defer p.end()
		defer func() {
			if selected != nil {
				if err := closeOverride(p, selected); err != nil {
					b.Errorf("cuptibench: close override plan: %v", err)
				}
			}
		}()
	}
	result, err := benchmark(b, selected, w, func(p *Plan, w Workload) (benchmarkTimer, error) { return cuda.NewEventTimer(p.dev, w.Streams) })
	if err != nil {
		b.Fatal(err)
		return nil
	}
	if owned {
		err := closeOverride(p, selected)
		selected = nil
		if err != nil {
			b.Fatal(err)
			return nil
		}
	}
	if err := recordBenchmarkResult(result, p); err != nil {
		b.Fatal(err)
		return nil
	}
	return result
}

func benchmarkPlan(p *Plan) (*Plan, bool, error) {
	override, err := benchmarkMetricOverride()
	if err != nil {
		return nil, false, err
	}
	if override == "" {
		return p, false, nil
	}
	if err := p.begin(); err != nil {
		return nil, false, err
	}
	keep := false
	defer func() {
		if !keep {
			p.end()
		}
	}()
	metrics, err := Resolve(strings.Split(override, ",")...)
	if err != nil {
		return nil, false, err
	}
	config := p.config
	config.Metrics = metrics
	q, err := New(p.dev, config)
	if err != nil {
		if q != nil {
			err = errors.Join(err, closeOverride(p, q))
		}
		return nil, false, err
	}
	q.metadata["metric_selection"] = "benchmark-override"
	q.metadata["metric_override"] = override
	keep = true
	return q, true, nil
}

func closeOverride(owner, child *Plan) error {
	if err := child.Close(); err != nil {
		owner.pendingOverride = child
		return err
	}
	return nil
}

func benchmark(b *testing.B, p *Plan, w Workload, newTimer timerFactory) (result *Result, err error) {
	b.Helper()
	if err = p.begin(); err != nil {
		return nil, err
	}
	defer p.end()
	defer func() {
		if value := recover(); value != nil {
			p.closed = true
			panic(value)
		}
	}()
	if err = p.validateWorkload(w); err != nil {
		return nil, err
	}
	w.Streams = append([]*cuda.Stream(nil), w.Streams...)
	w.Graphs = append([]*cudagraph.Exec(nil), w.Graphs...)
	var timer benchmarkTimer
	defer func() {
		if timer != nil {
			if cleanup := timer.Close(); cleanup != nil {
				p.pendingTimer = timer
				err = errors.Join(err, cleanup)
			}
		}
		if err != nil {
			result = nil
		}
	}()
	var elapsed time.Duration
	var operations int64
	err = p.context.Run(func() error {
		var e error
		timer, e = newTimer(p, w)
		if e != nil {
			return e
		}
		for b.Loop() {
			b.StopTimer()
			if e = p.call(w.Synchronize); e != nil {
				return fmt.Errorf("cuptibench: drain: %w", e)
			}
			if w.Prepare != nil {
				if e = p.call(w.Prepare); e != nil {
					return fmt.Errorf("cuptibench: prepare: %w", e)
				}
			}
			if e = p.call(w.Synchronize); e != nil {
				return fmt.Errorf("cuptibench: prepare synchronize: %w", e)
			}
			b.StartTimer()
			timedWork := w
			timedWork.Run = func(scope *Scope) error { return p.call(func() error { return w.Run(scope) }) }
			timedWork.Synchronize = func() error { return p.call(w.Synchronize) }
			d, e := measureSequence(timer, timedWork, p.config.Operations)
			if e != nil {
				return fmt.Errorf("cuptibench: timing: %w", e)
			}
			if d < 0 || d > time.Duration(math.MaxInt64)-elapsed || int64(p.config.Operations) > math.MaxInt64-operations {
				return errors.New("cuptibench: timing overflow")
			}
			elapsed += d
			operations += int64(p.config.Operations)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if err = timer.Close(); err != nil {
		p.pendingTimer = timer
		timer = nil
		return nil, err
	}
	timer = nil
	result, err = p.collect(w)
	if err != nil {
		return result, err
	}
	if p.config.Mode == Kernels {
		aggregate, collectErr := p.collectMode(w, Whole)
		if collectErr != nil {
			return nil, fmt.Errorf("cuptibench: aggregate capture: %w", collectErr)
		}
		result.Aggregate = aggregate
		result.Metadata["aggregate_capture_id"] = aggregate.Metadata["capture_id"]
	}
	result.Timing = &Timing{Operations: operations, Total: elapsed, Method: "cuda-events"}
	result.Metadata["benchmark"] = b.Name()
	id := make([]byte, 16)
	if _, err = rand.Read(id); err != nil {
		return nil, err
	}
	result.Metadata["run_id"] = fmt.Sprintf("%x", id)
	result.Metadata["timing_capture_id"] = "timing-" + result.Metadata["run_id"]
	if result.Aggregate != nil {
		result.Aggregate.Metadata["benchmark"] = b.Name()
		result.Aggregate.Metadata["run_id"] = result.Metadata["run_id"]
		result.Aggregate.Metadata["timing_capture_id"] = result.Metadata["timing_capture_id"]
	}
	if err = report(b, result); err != nil {
		return nil, err
	}
	return result, nil
}

func report(b *testing.B, r *Result) error {
	if r == nil || r.Operations <= 0 {
		return errors.New("cuptibench: invalid result for reporting")
	}
	// Report only whole-operation counters as ordinary columns. Attributed ranges
	// are retained in Result and the profile; overwriting columns with each child
	// would silently report whichever kernel happened to run last.
	var root *Range
	counters := r
	if r.Aggregate != nil {
		counters = r.Aggregate
	}
	for i := range counters.Ranges {
		v := &counters.Ranges[i]
		if v.Kind == "operation" && v.ParentID == "" {
			if root != nil {
				return errors.New("cuptibench: multiple operation roots")
			}
			root = v
		}
	}
	if root == nil && len(counters.Ranges) == 1 {
		root = &counters.Ranges[0]
	}
	if root != nil {
		for _, m := range root.Metrics {
			if m.Name == "" || strings.IndexFunc(m.Name, func(r rune) bool { return r == ' ' || r == '\t' || r == '\n' || r == '\r' }) >= 0 || math.IsNaN(m.Value) || math.IsInf(m.Value, 0) {
				return errors.New("cuptibench: invalid benchmark metric")
			}
		}
	}
	if r.Timing != nil && r.Timing.Operations > 0 && r.Timing.Total > 0 {
		b.ReportMetric(float64(r.Timing.Total)/float64(r.Timing.Operations), "gpu-ns/op")
	}
	b.ReportMetric(float64(r.Operations), "profile-ops")
	if root != nil {
		for _, m := range root.Metrics {
			b.ReportMetric(m.Value, m.Name)
		}
	}
	return nil
}

func measureSequence(timer benchmarkTimer, w Workload, operations int) (duration time.Duration, err error) {
	defer func() {
		err = errors.Join(err, w.Synchronize())
		if err != nil {
			duration = 0
		}
	}()
	return timer.Measure(func() error {
		for i := 0; i < operations; i++ {
			if err := runWorkload(w, nil); err != nil {
				return err
			}
		}
		return nil
	})
}
