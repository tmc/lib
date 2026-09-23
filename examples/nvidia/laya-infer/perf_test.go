package main

// GPU measurements of the forward pass. They need a real model, so they skip
// unless LAYA_MODEL names a directory holding one:
//
//	LAYA_MODEL=$HOME/laya go test -run '^$' -bench . -benchtime 20x
//
// BenchmarkPredict times the whole request and BenchmarkPredictExact the same
// request in float32. BenchmarkForwardWhole times only the device work,
// BenchmarkForwardWholeExact times it in float32 for comparison, and
// BenchmarkForwardRegions attributes that time to the named phases of launch.
//
// The two ForwardWhole benchmarks differ only in the compute type, so run them
// together or not at all: absolute device time drifts between runs, and a
// speedup is only meaningful against a control measured alongside it.

import (
	"context"
	"fmt"
	"os"
	"sort"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cuptibench"
	"github.com/tmc/lib/nvidia/x/cuptirange"
)

// benchModel loads the model named by LAYA_MODEL at prec, skipping if it is
// unset.
func benchModel(tb testing.TB, prec precision) *model {
	tb.Helper()
	dir := os.Getenv("LAYA_MODEL")
	if dir == "" {
		tb.Skip("set LAYA_MODEL to a directory holding model.safetensors")
	}
	m, err := newModel(dir, prec)
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() {
		if err := m.Close(); err != nil {
			tb.Error(err)
		}
	})
	return m
}

// benchRequest is the fixture both forward benchmarks measure: four questions
// of all three types over one state, padded into a single batch.
func benchRequest(tb testing.TB) []byte {
	tb.Helper()
	req, err := os.ReadFile("testdata/bench.json")
	if err != nil {
		tb.Fatal(err)
	}
	return req
}

// benchBatch tokenizes request the way Predict does, without running anything
// on the device.
func benchBatch(tb testing.TB, m *model, request []byte) *batch {
	tb.Helper()
	root, err := parseJSON(request)
	if err != nil {
		tb.Fatal(err)
	}
	state, qs := root.get("state"), root.get("questions")
	if state == nil || qs == nil || qs.kind != jObject {
		tb.Fatal("request must be an object with 'state' and 'questions'")
	}
	st := state.str
	if state.kind != jString {
		st = state.dumpString(false)
	}
	n := qs.length()
	seqs, markers, types := make([][]int32, n), make([][]int32, n), make([]int32, n)
	for i := range n {
		q, err := parseQuestion(qs.keys[i], qs.vals[i])
		if err != nil {
			tb.Fatal(err)
		}
		if seqs[i], markers[i], err = buildSequence(m.cfg, m.tok, st, q); err != nil {
			tb.Fatal(err)
		}
		types[i] = int32(q.typ)
	}
	return newBatch(seqs, types, markers)
}

// benchPass prepares one forward pass and reports its shape, so the numbers
// below can be read against the work they describe.
func benchPass(tb testing.TB, m *model) *pass {
	tb.Helper()
	b := benchBatch(tb, m, benchRequest(tb))
	var p *pass
	if err := m.dev.WithContext(func() error {
		var err error
		p, err = m.prepare(b)
		return err
	}); err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() {
		if err := m.dev.WithContext(p.Free); err != nil {
			tb.Error(err)
		}
	})
	real := 0
	for _, n := range b.lens {
		real += int(n)
	}
	tb.Logf("batch: %d sequences, T=%d, %d padded rows for %d tokens", len(b.lens), b.T, len(b.lens)*b.T, real)
	return p
}

// workload describes one replay of p for the profiler. Nothing in launch
// mutates its inputs, so a replay recomputes the same activations from the
// weights and token ids uploaded by prepare and needs no reset.
func (m *model) workload(p *pass) cuptibench.Workload {
	return cuptibench.Workload{
		View:        "full",
		Streams:     []*cuda.Stream{m.stream},
		Run:         func(s *cuptibench.Scope) error { return m.launch(p, s.Region) },
		Synchronize: func() error { return m.stream.Synchronize(context.Background()) },
		Preparation: "weights and token ids are uploaded once by prepare; launch only reads them " +
			"and overwrites its own scratch, so each pass recomputes the same activations. " +
			"Cache state is uncontrolled and warm after the first pass.",
	}
}

// benchPlan configures a counter plan for one mode. Limits are sized for the
// ten region names launch uses, not for per-kernel attribution.
func benchPlan(tb testing.TB, dev *cuda.Device, mode cuptibench.Mode, selectors ...string) *cuptibench.Plan {
	tb.Helper()
	metrics, err := cuptibench.Resolve(selectors...)
	if err != nil {
		tb.Fatal(err)
	}
	depth := uint16(0) // only Regions may nest
	if mode == cuptibench.Regions {
		depth = 4
	}
	plan, err := cuptibench.New(dev, cuptibench.Config{
		CachePolicy:    cuptirange.CachePreserve,
		Metrics:        metrics,
		Mode:           mode,
		Operations:     1,
		MaxConfigBytes: 16 << 20,
		MaxDepth:       depth,
		Limits: cuptirange.Limits{
			MaxRanges:         256,
			MaxRangeTreeNodes: 512,
			MaxCounterBytes:   64 << 20,
			MaxPasses:         64,
		},
	})
	if plan != nil {
		tb.Cleanup(func() {
			if err := plan.Close(); err != nil {
				tb.Error(err)
			}
		})
	}
	if err != nil {
		tb.Fatal(err)
	}
	return plan
}

func BenchmarkPredict(b *testing.B) {
	m := benchModel(b, fast)
	req := benchRequest(b)
	if _, err := m.Predict(req); err != nil { // warm up: JIT, cuBLAS handles, first-touch
		b.Fatal(err)
	}
	b.ResetTimer()
	for b.Loop() {
		if _, err := m.Predict(req); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkPredictExact is the same request in float32. It is the whole-request
// counterpart to BenchmarkForwardWholeExact, and exists so that a comparison
// against another implementation has a matched-precision pair to use rather
// than only the device-time one.
func BenchmarkPredictExact(b *testing.B) {
	m := benchModel(b, exact)
	req := benchRequest(b)
	if _, err := m.Predict(req); err != nil { // warm up: JIT, cuBLAS handles, first-touch
		b.Fatal(err)
	}
	b.ResetTimer()
	for b.Loop() {
		if _, err := m.Predict(req); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkForwardWhole(b *testing.B) {
	m := benchModel(b, fast)
	p := benchPass(b, m)
	cuptibench.Benchmark(b, benchPlan(b, m.dev, cuptibench.Whole, "time"), m.workload(p))
}

// BenchmarkForwardWholeExact is the same pass in float32, which is what the
// tf32 and fused-attention numbers are a speedup over. Without it the default
// path has no control and the comparison cannot be reproduced from the tree.
func BenchmarkForwardWholeExact(b *testing.B) {
	m := benchModel(b, exact)
	p := benchPass(b, m)
	cuptibench.Benchmark(b, benchPlan(b, m.dev, cuptibench.Whole, "time"), m.workload(p))
}

func BenchmarkForwardRegions(b *testing.B) {
	m := benchModel(b, fast)
	p := benchPass(b, m)
	r := cuptibench.Benchmark(b, benchPlan(b, m.dev, cuptibench.Regions, "time"), m.workload(p))
	reportRegions(b, r, "gpu__time_duration.sum")
}

// reportRegions logs each region name's share of the operation root, largest
// first. Regions mode records one range per invocation, so the phases inside
// the layer loop are summed by name and counted. Region metrics are inclusive,
// so the root is not the sum of its children: the shortfall is time inside the
// operation but outside every named phase.
func reportRegions(tb testing.TB, r *cuptibench.Result, metric string) {
	tb.Helper()
	if r == nil {
		return
	}
	value := func(rg cuptibench.Range) (float64, bool) {
		for _, m := range rg.Metrics {
			if m.Name == metric {
				return m.Value, true
			}
		}
		return 0, false
	}
	type row struct {
		name   string
		v      float64
		calls  int
		nested bool // inside another named region, so not part of (named)
	}
	var root float64
	var rootID string
	var rows []row
	index := map[string]int{}
	for _, rg := range r.Ranges {
		v, ok := value(rg)
		if !ok {
			continue
		}
		if rg.ParentID == "" {
			root, rootID = v, rg.ID
			continue
		}
		i, seen := index[rg.Name]
		if !seen {
			i = len(rows)
			index[rg.Name], rows = i, append(rows, row{name: rg.Name})
		}
		rows[i].v += v
		rows[i].calls++
		rows[i].nested = rg.ParentID != rootID
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].v > rows[j].v })
	var named float64
	for _, x := range rows {
		if !x.nested {
			named += x.v
		}
	}
	tb.Logf("%-10s %6s %12s %8s", "region", "calls", "ns", "share")
	for _, x := range rows {
		name := x.name
		if x.nested {
			name = "  " + name
		}
		tb.Logf("%-10s %6d %12.0f %7s", name, x.calls, x.v, pct(x.v, root))
	}
	tb.Logf("%-10s %6s %12.0f %7s", "(named)", "", named, pct(named, root))
	tb.Logf("%-10s %6s %12.0f %7s", "operation", "", root, pct(root, root))
}

func pct(v, total float64) string {
	if total == 0 {
		return "n/a"
	}
	return fmt.Sprintf("%.1f%%", 100*v/total)
}

// BenchmarkForwardNVTX runs the same launch sequence with each phase bracketed
// by an NVTX range, for a timeline that names its kernels' callers:
//
//	LAYA_MODEL=$HOME/laya nsys profile -t cuda,nvtx -o /tmp/laya \
//	    go test -run '^$' -bench ForwardNVTX -benchtime 10x
//	nsys stats --report cuda_gpu_kern_sum /tmp/laya.nsys-rep
func BenchmarkForwardNVTX(b *testing.B) {
	m := benchModel(b, fast)
	p := benchPass(b, m)
	sync := func() error { return m.stream.Synchronize(context.Background()) }
	if err := m.dev.WithContext(func() error {
		if err := m.launch(p, nil); err != nil { // warm up outside the timer
			return err
		}
		return sync()
	}); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	if err := m.dev.WithContext(func() error {
		for b.Loop() {
			if err := m.launch(p, nvtxRegion); err != nil {
				return err
			}
			if err := sync(); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		b.Fatal(err)
	}
}

// nvtxRegion brackets a phase with an NVTX range. cudaprof.Do is a sink for
// caller-supplied observations; the range is what nsys records.
func nvtxRegion(name string, run func() error) error {
	var err error
	cudaprof.Do(context.Background(), cudaprof.Labels("phase", name), func(context.Context) {
		err = run()
	})
	return err
}
