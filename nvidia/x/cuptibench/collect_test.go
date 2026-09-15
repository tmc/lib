package cuptibench

import (
	"errors"
	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cuptirange"
	"math"
	"reflect"
	"slices"
	"testing"
)

type testContext struct{ release error }

func (c *testContext) Run(f func() error) error { return f() }
func (c *testContext) Release() error           { return c.release }

type testCollector struct {
	passes                 uint
	captureErr, errorClose error
	values                 []float64
	mutate                 func(*cuptirange.Capture)
	calls                  int
}

func (c *testCollector) Close() error                             { return c.errorClose }
func (c *testCollector) Evaluate([]byte, uint) ([]float64, error) { return c.values, nil }
func (c *testCollector) CaptureReplay(_ *cuda.Device, _ cuptirange.Limits, _ cuptirange.CaptureOptions, r cuptirange.Replay) (*cuptirange.Capture, error) {
	c.calls++
	for i := uint(0); i < c.passes; i++ {
		if e := r.Synchronize(); e != nil {
			return nil, e
		}
		if e := r.Prepare(i); e != nil {
			return nil, e
		}
		if e := r.Synchronize(); e != nil {
			return nil, e
		}
		if e := r.Run(i, nil); e != nil {
			return nil, e
		}
		if e := r.Synchronize(); e != nil {
			return nil, e
		}
	}
	result := &cuptirange.Capture{Image: []byte{1}, Ranges: 1, RangeInfo: []cuptirange.RangeInfo{{Index: 0, ID: "operation#1", Name: "operation", Kind: "operation", Inclusive: true}}, Quality: cuptirange.CollectionQuality{PassesStarted: c.passes, PassesStopped: c.passes, AllPassesSubmitted: true, Decoded: true}}
	if c.mutate != nil {
		c.mutate(result)
	}
	return result, c.captureErr
}
func testPlan(ops int) (*Plan, *testCollector, Workload) {
	dev := new(cuda.Device)
	backend := &testCollector{passes: 2, values: []float64{12}}
	p := &Plan{dev: dev, config: Config{Metrics: []string{"counter"}, Operations: ops, Limits: cuptirange.Limits{MaxRanges: 8, MaxRangeTreeNodes: 16, MaxCounterBytes: 1024, MaxPasses: 8}, MaxConfigBytes: 1024}, context: &testContext{}, backend: backend, units: []string{"count"}, metadata: map[string]string{"chip": "test"}}
	w := Workload{Streams: []*cuda.Stream{dev.DefaultStream()}, Run: func(*Scope) error { return nil }, Synchronize: func() error { return nil }, Preparation: "repeatable fixture; cache policy unspecified"}
	return p, backend, w
}
func TestCollectReplayAccounting(t *testing.T) {
	for _, passes := range []uint{1, 2, 4} {
		t.Run(string(rune('0'+passes)), func(t *testing.T) {
			p, c, w := testPlan(3)
			c.passes = passes
			var log []string
			w.Prepare = func() error { log = append(log, "prepare"); return nil }
			w.Synchronize = func() error { log = append(log, "sync"); return nil }
			w.Run = func(*Scope) error { log = append(log, "run"); return nil }
			got, err := p.Collect(w)
			if err != nil {
				t.Fatal(err)
			}
			if got.Operations != 3 || got.Ranges[0].Metrics[0].Value != 12 || got.Quality.PassesStopped != passes || got.Timing != nil {
				t.Fatalf("result %+v", got)
			}
			var want []string
			for i := uint(0); i < passes; i++ {
				want = append(want, "sync", "prepare", "sync", "run", "run", "run", "sync")
			}
			if !reflect.DeepEqual(log, want) {
				t.Fatalf("sequence %v, want %v", log, want)
			}
		})
	}
}
func TestCollectRejectsInvalidCapture(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*cuptirange.Capture)
	}{
		{"incomplete", func(c *cuptirange.Capture) { c.Quality.AllPassesSubmitted = false }},
		{"undecoded", func(c *cuptirange.Capture) { c.Quality.Decoded = false }},
		{"dropped", func(c *cuptirange.Capture) { c.Quality.DroppedRanges = 1 }},
		{"unbalanced", func(c *cuptirange.Capture) { c.Quality.PassesStopped-- }},
		{"identity", func(c *cuptirange.Capture) { c.RangeInfo = nil }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, c, w := testPlan(1)
			c.mutate = tc.mutate
			got, err := p.Collect(w)
			if err == nil || got == nil || len(got.Ranges) != 0 {
				t.Fatalf("got %+v, %v", got, err)
			}
		})
	}
	for _, values := range [][]float64{nil, {math.NaN()}, {math.Inf(1)}} {
		p, c, w := testPlan(1)
		c.values = values
		got, err := p.Collect(w)
		if err == nil || len(got.Ranges) != 0 {
			t.Fatalf("accepted %v", values)
		}
	}
}
func TestCollectWorkloadFailureAndOwnership(t *testing.T) {
	p, c, w := testPlan(1)
	sentinel := errors.New("prepare failed")
	w.Prepare = func() error { return sentinel }
	got, err := p.Collect(w)
	if !errors.Is(err, sentinel) || len(got.Ranges) != 0 {
		t.Fatalf("%+v %v", got, err)
	}
	w.Prepare = nil
	w.Run = func(*Scope) error {
		if _, err := p.Collect(w); err == nil {
			t.Error("nested collect accepted")
		}
		if err := p.Close(); err == nil {
			t.Error("active close accepted")
		}
		return nil
	}
	if _, err = p.Collect(w); err != nil {
		t.Fatal(err)
	}
	c.errorClose = sentinel
	if err = p.Close(); !errors.Is(err, sentinel) {
		t.Fatal(err)
	}
	if p.backend == nil || p.context == nil {
		t.Fatal("failed close released owners")
	}
	c.errorClose = nil
	if err = p.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err = p.Collect(w); err == nil {
		t.Fatal("closed collect accepted")
	}
}
func TestScopeLifetimeAndPanic(t *testing.T) {
	var retained *Scope
	w := Workload{Run: func(s *Scope) error { retained = s; return s.Region("one", func() error { return nil }) }}
	if err := runWorkload(w, nil); err != nil {
		t.Fatal(err)
	}
	if err := retained.Region("late", func() error { return nil }); err == nil {
		t.Fatal("expired scope accepted")
	}
	w.Run = func(s *Scope) error { retained = s; panic("workload") }
	func() {
		defer func() {
			if recover() != "workload" {
				t.Error("panic not preserved")
			}
		}()
		_ = runWorkload(w, nil)
	}()
	if retained.active {
		t.Fatal("panic retained active scope")
	}
}
func TestConfigAndMetricSemantics(t *testing.T) {
	p, _, _ := testPlan(1)
	if err := p.config.validate(); err != nil {
		t.Fatal(err)
	}
	for _, metrics := range [][]string{nil, {""}, {"a", "a"}, {"a b"}, {"a\x00"}} {
		c := p.config
		c.Metrics = metrics
		if c.validate() == nil {
			t.Fatalf("accepted metrics %q", metrics)
		}
	}
	for _, tc := range []struct {
		name, unit string
		kind       cupti.MetricType
		want       bool
	}{
		{"smsp__inst_executed.sum", "inst", cupti.MetricTypeCounter, true},
		{"sm__ctas_launched.sum", "cta", cupti.MetricTypeCounter, true},
		{"lts__t_sectors_srcunit_tex_op_read.sum", "sector", cupti.MetricTypeCounter, true},
		{"lts__t_sectors_srcunit_tex_op_read.sum", "l2_sector", cupti.MetricTypeCounter, true},
		{"lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum", "l2_sector", cupti.MetricTypeCounter, true},
		{"lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum", "byte", cupti.MetricTypeCounter, false},
		{"lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum", "l2_sector", cupti.MetricTypeRatio, false},
		// A discovered counter rollup is additive: nothing curated expects a
		// particular dimension for it, but ".sum" still totals events.
		{"arbitrary.sum", "count", cupti.MetricTypeCounter, true},
		{"arbitrary.sum", "unknown", cupti.MetricTypeCounter, false},
		{"arbitrary.avg", "count", cupti.MetricTypeCounter, false},
		{"arbitrary.peak_sustained", "count", cupti.MetricTypeCounter, false},
		{"arbitrary.sum", "count", cupti.MetricTypeThroughput, false},
		// Time is not additive across ranges however the rollup is spelled.
		// gpu__time_start.sum is a timestamp; the cycle families are durations
		// summed over hardware units, so ranges that overlap count twice.
		{"gpu__time_start.sum", "nanosecond", cupti.MetricTypeCounter, false},
		{"gpu__time_active.sum", "nanosecond", cupti.MetricTypeCounter, false},
		{"fe__cycles_elapsed.sum", "sys_cycle", cupti.MetricTypeCounter, false},
		{"sm__warps_active.sum", "gpc_cyclewarp", cupti.MetricTypeCounter, false},
		{"tpc__l1tex_sram_bytes_mem_tagged_realtime.sum", "bytegpc_cycle", cupti.MetricTypeCounter, false},
		{"smsp__inst_executed.sum", "unknown", cupti.MetricTypeCounter, false},
		{"smsp__inst_executed.sum", "inst", cupti.MetricTypeRatio, false},
	} {
		if got := additiveMetric(tc.name, tc.unit, tc.kind); got != tc.want {
			t.Errorf("%s %s: %v", tc.name, tc.unit, got)
		}
	}
	if _, err := resolveSets([]string{"missing"}); err == nil {
		t.Fatal("unknown set accepted")
	}
	// A repeated selector contributes its metrics once: a duplicate metric is
	// rejected by Config.validate, so resolution must not produce one.
	names, err := resolveSets([]string{"instructions", "l2-read", "instructions"})
	want := []string{"smsp__inst_executed.sum", "smsp__thread_inst_executed.sum", "lts__t_sectors_srcunit_tex_op_read.sum"}
	if err != nil || !slices.Equal(names, want) {
		t.Fatalf("got %v %v, want %v", names, err, want)
	}
}

func TestContextMutationRejectsMeasurement(t *testing.T) {
	p, _, w := testPlan(1)
	current := true
	p.checkContext = func() error {
		if !current {
			current = true
			return errors.New("context changed")
		}
		return nil
	}
	w.Run = func(*Scope) error { current = false; return nil }
	result, err := p.Collect(w)
	if err == nil || len(result.Ranges) != 0 || !current {
		t.Fatalf("result=%+v err=%v current=%v", result, err, current)
	}
}
func TestWorkloadValidation(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*Workload)
	}{
		{"run", func(w *Workload) { w.Run = nil }}, {"sync", func(w *Workload) { w.Synchronize = nil }},
		{"policy", func(w *Workload) { w.Preparation = " " }}, {"streams", func(w *Workload) { w.Streams = nil }},
		{"duplicate", func(w *Workload) { w.Streams = append(w.Streams, w.Streams[0]) }},
		{"device", func(w *Workload) { w.Streams = []*cuda.Stream{new(cuda.Device).DefaultStream()} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, c, w := testPlan(1)
			tc.change(&w)
			if _, err := p.Collect(w); err == nil || c.calls != 0 {
				t.Fatalf("err=%v captures=%d", err, c.calls)
			}
		})
	}
}

func TestWorkloadView(t *testing.T) {
	for _, view := range []string{"", "full", "isolated", "unknown"} {
		p, _, w := testPlan(1)
		w.View = view
		r, err := p.Collect(w)
		if view == "unknown" {
			if err == nil {
				t.Fatal("unknown view accepted")
			}
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		want := view
		if want == "" {
			want = "unspecified"
		}
		if r.Metadata["view"] != want {
			t.Fatalf("view %q: %+v", view, r.Metadata)
		}
	}
}

func TestScopeIgnoredError(t *testing.T) {
	w := Workload{Run: func(s *Scope) error { _ = s.Region("invalid/name", func() error { return nil }); return nil }}
	if err := runWorkload(w, nil); err == nil {
		t.Fatal("ignored scope error admitted")
	}
}

func TestCollectPanicRequiresClose(t *testing.T) {
	p, _, w := testPlan(1)
	w.Run = func(*Scope) error { panic("run") }
	func() {
		defer func() {
			if recover() != "run" {
				t.Fatal("panic changed")
			}
		}()
		_, _ = p.Collect(w)
	}()
	if _, err := p.Collect(w); err == nil {
		t.Fatal("panicked plan reused")
	}
	if err := p.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestNegativeEventCounter(t *testing.T) {
	p, c, w := testPlan(1)
	p.additive = []bool{true}
	c.values = []float64{-1}
	r, err := p.Collect(w)
	if err == nil || len(r.Ranges) != 0 {
		t.Fatal("negative event counter admitted")
	}
}
