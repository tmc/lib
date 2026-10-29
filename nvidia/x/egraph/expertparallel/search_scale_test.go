package expertparallel

import (
	"context"
	"errors"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// stackSpec is a decoder stack for e-graph search scaling: layer counts,
// widths and MoE shape. Only the graph's structure matters for search;
// widths go into leaf shapes.
type stackSpec struct {
	name                string
	layers, denseLayers int // denseLayers leading layers use a dense FFN
	hidden, ffn         int
	experts, topK       int
	expertFFN           int
	unrolled            bool // one MatMul path per expert instead of grouped terms
}

func (s stackSpec) build() (*hopper.Builder, egraph.Id) {
	const tokens = 64
	b := hopper.NewBuilder()
	H := s.hidden
	x := b.Input("x", hopper.BF16, tokens, H)
	attnCfg := hopper.PagedAttention{Hq: 64, Hkv: 8, QLen: 1, Page: 16}
	kc := b.Input("k_cache", hopper.E4M3, 4096, 8, 64)
	vc := b.Input("v_cache", hopper.E4M3, 4096, 8, 64)
	pt := b.Input("page_table", hopper.I32, tokens, 256)
	sl := b.Input("seq_lens", hopper.I32, tokens)
	ks := b.Input("k_scale", hopper.F32, 1)
	vs := b.Input("v_scale", hopper.F32, 1)
	ls := b.Input("local_start", hopper.I32, 1)
	rms := func(x egraph.Id, name string) egraph.Id {
		w := b.Input(name, hopper.BF16, H)
		ss := b.ReduceSum(b.Map(hopper.MustLambda("(mul $0 $0)"), x))
		r := b.Map(hopper.MustLambda(fmt.Sprintf("(rsqrt (add (mul $0 %g) 0.00001))", 1/float64(H))), ss)
		return b.Map(hopper.MustLambda("(mul (mul $0 $1) $2)"), x, r, w)
	}
	add := hopper.MustLambda("(add $0 $1)")
	for l := range s.layers {
		p := func(n string) string { return fmt.Sprintf("l%d_%s", l, n) }
		h := rms(x, p("attn_norm"))
		qkv := b.MatMul(h, b.Input(p("wqkv"), hopper.BF16, 80*64, H))
		q := b.Reshape(b.Slice(qkv, 0, 64*64), tokens, 64, 64)
		a := b.PagedAttention(attnCfg, q, kc, vc, pt, sl, ks, vs)
		o := b.MatMul(b.Reshape(a, tokens, 64*64), b.Input(p("wo"), hopper.BF16, H, 64*64))
		x = b.Map(add, x, o)
		h = rms(x, p("ffn_norm"))
		var f egraph.Id
		switch {
		case s.experts == 0 || l < s.denseLayers:
			gu := b.MatMul(h, b.Input(p("w_gate_up"), hopper.BF16, 2*s.ffn, H))
			f = b.MatMul(b.SwiGLU(gu), b.Input(p("w_down"), hopper.BF16, H, s.ffn))
		case s.unrolled:
			logits := b.MatMul(h, b.Input(p("w_router"), hopper.BF16, s.experts, H))
			_, weights := b.RowTopK(hopper.RowTopK{D: s.experts, K: s.topK, X: hopper.F32, Normalize: true}, logits, 0)
			for e := range s.experts {
				pe := func(n string) string { return fmt.Sprintf("l%d_e%d_%s", l, e, n) }
				gu := b.MatMul(h, b.Input(pe("w13"), hopper.BF16, 2*s.expertFFN, H))
				y := b.MatMul(b.SwiGLU(gu), b.Input(pe("w2"), hopper.BF16, H, s.expertFFN))
				g := b.Slice(weights, e, 1)
				ye := b.Map(hopper.MustLambda("(mul $0 $1)"), y, g)
				if e == 0 {
					f = ye
				} else {
					f = b.Map(add, f, ye)
				}
			}
		default:
			E, I := s.experts, s.expertFFN
			logits := b.MatMul(h, b.Input(p("w_router"), hopper.BF16, E, H))
			ids, weights := b.RowTopK(hopper.RowTopK{D: E, K: s.topK, X: hopper.F32, Normalize: true}, logits, 0)
			offsets, rows, slots := b.ExpertAlign(hopper.ExpertAlign{K: s.topK, Experts: E, Local: min(E, 32)}, ids, ls)
			w13 := b.NVFP4(b.Input(p("w13_q"), hopper.U8, E, 2*I, H/2), b.Input(p("w13_sf"), hopper.U8, E, 2*I, H/16), b.Input(p("w13_gs"), hopper.F32, E))
			w2 := b.NVFP4(b.Input(p("w2_q"), hopper.U8, E, H, I/2), b.Input(p("w2_sf"), hopper.U8, E, H, I/16), b.Input(p("w2_gs"), hopper.F32, E))
			inter := b.SwiGLU(b.GroupedMatMul(h, w13, offsets, rows))
			norm := b.PostExpertNorm(b.GroupedMatMul(inter, w2, offsets), b.Input(p("post_w"), hopper.BF16, H), 1e-8)
			routed := b.WeightedTopKSum(norm, slots, weights)
			gu := b.MatMul(h, b.Input(p("w_shared_gu"), hopper.BF16, 2*I, H))
			shared := b.MatMul(b.SwiGLU(gu), b.Input(p("w_shared_down"), hopper.BF16, H, I))
			f = b.Map(add, routed, shared)
		}
		x = b.Map(add, x, f)
	}
	return b, x
}

// scaleResult is one measurement of hopper's saturate-fuse loop followed
// by three extractions of the same graph.
type scaleResult struct {
	layers           int
	classes0, nodes0 int // graph before saturation
	classes, nodes   int // after
	stop             string
	saturate         time.Duration // rewrite saturation plus epilogue fusion
	satAllocMB       float64
	tree             extraction // ExtractBounded, default limits
	dag              extraction // ExtractDAG
	segmented        extraction // ExtractSegmented
	perLayer         extraction // each layer saturated and extracted on its own
	segments         int        // ExtractSegmented's segments
	distinct         int        // and those it extracted
	err              error
}

// An extraction is one extraction's time, allocation and result size.
type extraction struct {
	time     time.Duration
	allocMB  float64
	nodes    int     // distinct nodes of the result
	treeSize float64 // nodes of the result expanded to a tree
	launches int     // distinct kernel-shaped nodes
	work     int     // ExtractDAG's work units
	err      error
}

// launchOps are the operators that lower to a kernel launch.
var launchOps = map[string]bool{
	hopper.OpGEMM: true, hopper.OpMatMul: true, hopper.OpMap: true, hopper.OpReduceSum: true,
	hopper.OpPagedAttn: true, hopper.OpPagedAttnGated: true, hopper.OpRowTopK: true, hopper.OpExpertAlign: true,
	hopper.OpGroupedMatMul: true, hopper.OpGroupedGEMM: true, hopper.OpGroupedGEMMGLU: true, hopper.OpGroupedGEMMSumSq: true,
	hopper.OpSwiGLU: true, hopper.OpPostExpertNorm: true, hopper.OpWeightedTopKSum: true, hopper.OpExpertSum: true,
}

func extract(f func() (*egraph.Expr, int, error)) extraction {
	var x extraction
	runtime.GC()
	var m0, m1 runtime.MemStats
	runtime.ReadMemStats(&m0)
	t0 := time.Now()
	e, work, err := f()
	x.time = time.Since(t0)
	x.work = work
	runtime.ReadMemStats(&m1)
	x.allocMB = float64(m1.TotalAlloc-m0.TotalAlloc) / (1 << 20)
	if err != nil {
		x.err = err
		return x
	}
	x.nodes, x.treeSize, x.launches = exprSize(e)
	return x
}

func measure(s stackSpec) scaleResult {
	r := scaleResult{layers: s.layers}
	runtime.GC()
	var m0, m1 runtime.MemStats
	runtime.ReadMemStats(&m0)
	b, root := s.build()
	r.classes0, r.nodes0 = b.G.Size()
	t0 := time.Now()
	for range 8 {
		res, err := egraph.SaturateChecked(b.G, hopper.Rules(), egraph.SaturationOptions{MaxIterations: 16})
		if err != nil {
			r.err = err
			return r
		}
		r.stop = fmt.Sprint(res.StopReason)
		n, err := hopper.FuseEpilogues(b.G)
		if err != nil {
			r.err = err
			return r
		}
		if n == 0 {
			break
		}
	}
	r.saturate = time.Since(t0)
	runtime.ReadMemStats(&m1)
	r.satAllocMB = float64(m1.TotalAlloc-m0.TotalAlloc) / (1 << 20)
	r.classes, r.nodes = b.G.Size()
	if err := b.G.RebuildChecked(); err != nil {
		r.err = err
		return r
	}
	r.tree = extract(func() (*egraph.Expr, int, error) {
		e, _, stats, err := egraph.ExtractBoundedWithStats(context.Background(), b.G, root, hopper.Cost{}, egraph.DefaultExtractionLimits())
		return e, stats.Work, err
	})
	r.dag = extract(func() (*egraph.Expr, int, error) {
		e, _, stats, err := egraph.ExtractDAG(context.Background(), b.G, root, hopper.Cost{},
			egraph.ExtractionLimits{MaxWork: 1 << 40, MaxNodes: 1 << 24, MaxDepth: 1 << 20})
		return e, stats.Work, err
	})
	r.segmented = extract(func() (*egraph.Expr, int, error) {
		e, _, stats, err := egraph.ExtractSegmented(context.Background(), b.G, root, hopper.Cost{},
			egraph.ExtractionLimits{MaxWork: 1 << 40, MaxNodes: 1 << 24, MaxDepth: 1 << 20})
		r.segments, r.distinct = stats.Segments, stats.Distinct
		return e, stats.Work, err
	})
	r.perLayer = measurePerLayer(s)
	return r
}

// measurePerLayer saturates and extracts each layer of s on its own.
func measurePerLayer(s stackSpec) extraction {
	return extract(func() (*egraph.Expr, int, error) {
		var work int
		all := &egraph.Expr{Op: "layers"}
		for l := range s.layers {
			one := s
			one.layers = 1
			one.denseLayers = 0
			if l < s.denseLayers {
				one.denseLayers = 1
			}
			b, root := one.build()
			if _, err := hopper.Optimize(b.G, root); err != nil {
				return nil, 0, err
			}
			e, _, stats, err := egraph.ExtractDAG(context.Background(), b.G, root, hopper.Cost{},
				egraph.ExtractionLimits{MaxWork: 1 << 40, MaxNodes: 1 << 24, MaxDepth: 1 << 20})
			if err != nil {
				return nil, 0, err
			}
			all.Children = append(all.Children, e)
			work += stats.Work
		}
		return all, work, nil
	})
}

func exprSize(e *egraph.Expr) (dag int, tree float64, launches int) {
	seen := make(map[*egraph.Expr]float64)
	var walk func(*egraph.Expr) float64
	walk = func(e *egraph.Expr) float64 {
		if n, ok := seen[e]; ok {
			return n
		}
		n := 1.0
		for _, c := range e.Children {
			n += walk(c)
		}
		seen[e] = n
		if launchOps[e.Op] {
			launches++
		}
		return n
	}
	tree = walk(e)
	return len(seen), tree, launches
}

func (x extraction) String() string {
	if x.err != nil {
		msg := x.err.Error()
		if errors.Is(x.err, egraph.ErrExtractionLimit) {
			msg = "limit"
		}
		return fmt.Sprintf("%9.1fms %9.1fMB  %-28s", ms(x.time), x.allocMB, msg)
	}
	return fmt.Sprintf("%9.1fms %9.1fMB  %7d nodes %5d launches %.3g work", ms(x.time), x.allocMB, x.nodes, x.launches, float64(x.work))
}

func (r scaleResult) String() string {
	if r.err != nil {
		return "error: " + r.err.Error()
	}
	return fmt.Sprintf("%6d/%7d -> %6d/%7d  sat %7.1fms %7.1fMB  tree %s  dag %s  dagtree %.3g  segmented %s %d/%d segments  per-layer %9.1fms %9.1fMB %5d launches",
		r.classes0, r.nodes0, r.classes, r.nodes, ms(r.saturate), r.satAllocMB, r.tree, r.dag, r.dag.treeSize,
		r.segmented, r.distinct, r.segments, ms(r.perLayer.time), r.perLayer.allocMB, r.perLayer.launches)
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

// TestSearchScale reports saturation and extraction time and memory for
// Llama-70B-shaped and DeepSeek-V3-shaped stacks at growing layer and
// expert counts, and checks that whole-graph DAG extraction succeeds and
// that segmented extraction of the whole graph and per-layer extraction
// choose the same number of launches, one layer's worth per layer.
// Whole-graph DAG extraction agrees up to 8 layers; at 16 DeepSeek-V3
// layers it chooses 6 more launches. By default it runs the smallest points (a few
// seconds). Set EGRAPH_SEARCH_SCALE=1 for the full curve up to 80 layers
// and 256 unrolled experts: whole-graph ExtractDAG grows superlinearly
// (about 10 minutes and 85 GB allocated at 80 DeepSeek-V3 layers), so
// that run takes about 20 minutes.
func TestSearchScale(t *testing.T) {
	if testing.Short() {
		t.Skip("e-graph search scaling is slow")
	}
	full := os.Getenv("EGRAPH_SEARCH_SCALE") == "1"
	layers := []int{1, 2, 4}
	experts := []int{8, 32}
	if full {
		layers = []int{1, 2, 4, 8, 16, 32, 61, 80}
		experts = []int{8, 16, 32, 64, 128, 256}
	}
	if v := os.Getenv("EGRAPH_SEARCH_SCALE_MAX_LAYERS"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			t.Fatalf("EGRAPH_SEARCH_SCALE_MAX_LAYERS: %v", err)
		}
		layers = slicesUpTo(layers, n)
	}
	var lines []string
	check := func(name string, s stackSpec) scaleResult {
		r := measure(s)
		line := fmt.Sprintf("%-26s %s", name, r)
		lines = append(lines, line)
		t.Log(line)
		switch {
		case r.err != nil:
			t.Errorf("%s: %v", name, r.err)
		case r.dag.err != nil:
			t.Errorf("%s: ExtractDAG: %v", name, r.dag.err)
		case r.segmented.err != nil:
			t.Errorf("%s: ExtractSegmented: %v", name, r.segmented.err)
		case r.perLayer.err != nil:
			t.Errorf("%s: per-layer extraction: %v", name, r.perLayer.err)
		case r.segmented.launches != r.perLayer.launches:
			t.Errorf("%s: segmented extraction chose %d launches, per-layer %d", name, r.segmented.launches, r.perLayer.launches)
		case r.dag.launches != r.perLayer.launches:
			// Tree costs reach 1e28 at 16 DeepSeek-V3 layers, where
			// float64 cannot resolve a 2000-unit launch.
			t.Logf("%s: whole-graph DAG chose %d launches, per-layer %d", name, r.dag.launches, r.perLayer.launches)
		}
		if r.tree.err != nil && !errors.Is(r.tree.err, egraph.ErrExtractionLimit) {
			t.Errorf("%s: ExtractBounded: %v", name, r.tree.err)
		}
		return r
	}
	// Every Llama layer is the same, so each must extract to the same
	// launches.
	perLayer := 0
	for _, n := range layers {
		r := check(fmt.Sprintf("llama-70b L=%d", n), stackSpec{name: "llama70b", layers: n, hidden: 8192, ffn: 28672})
		if r.perLayer.err != nil {
			continue
		}
		if perLayer == 0 {
			perLayer = r.perLayer.launches / n
		}
		if r.perLayer.launches != n*perLayer {
			t.Errorf("llama-70b L=%d: %d launches, want %d per layer", n, r.perLayer.launches, perLayer)
		}
	}
	for _, n := range layers {
		s := stackSpec{name: "dsv3", layers: n, denseLayers: min(3, n/2), hidden: 7168, ffn: 18432, experts: 256, topK: 8, expertFFN: 2048}
		check(fmt.Sprintf("dsv3 grouped E=256 L=%d", n), s)
	}
	for _, e := range experts {
		s := stackSpec{name: "dsv3u", layers: 1, hidden: 7168, experts: e, topK: 8, expertFFN: 2048, unrolled: true}
		check(fmt.Sprintf("dsv3 unrolled E=%d L=1", e), s)
	}
	if out := os.Getenv("EGRAPH_SEARCH_SCALE_OUT"); out != "" {
		if err := os.WriteFile(out, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// saturated builds s and saturates it as [hopper.Optimize] does.
func saturated(t testing.TB, s stackSpec) (*egraph.EGraph, egraph.Id) {
	b, root := s.build()
	for range 8 {
		if _, err := egraph.SaturateChecked(b.G, hopper.Rules(), egraph.SaturationOptions{MaxIterations: 16}); err != nil {
			t.Fatal(err)
		}
		n, err := hopper.FuseEpilogues(b.G)
		if err != nil {
			t.Fatal(err)
		}
		if n == 0 {
			break
		}
	}
	if err := b.G.RebuildChecked(); err != nil {
		t.Fatal(err)
	}
	return b.G, root
}

// TestSegmentedMatchesWholeGraph checks that segmented extraction, which
// [hopper.Optimize] uses, chooses the same term as whole-graph ExtractDAG
// for every model shape small enough to extract whole, and that it
// extracts each distinct layer once.
func TestSegmentedMatchesWholeGraph(t *testing.T) {
	llama := func(n, hidden, ffn int) stackSpec {
		return stackSpec{layers: n, hidden: hidden, ffn: ffn}
	}
	dsv3 := func(n, dense int) stackSpec {
		return stackSpec{layers: n, denseLayers: dense, hidden: 7168, ffn: 18432, experts: 256, topK: 8, expertFFN: 2048}
	}
	unrolled := func(e int) stackSpec {
		return stackSpec{layers: 1, hidden: 7168, experts: e, topK: 8, expertFFN: 2048, unrolled: true}
	}
	tests := []struct {
		name     string
		spec     stackSpec
		distinct int // segments extracted rather than reused
	}{
		{"llama-8b L=1", llama(1, 4096, 14336), 2},
		{"llama-8b L=4", llama(4, 4096, 14336), 3},
		{"llama-70b L=2", llama(2, 8192, 28672), 3},
		{"llama-70b L=8", llama(8, 8192, 28672), 3},
		{"dsv3 L=1 dense", dsv3(1, 1), 2},
		{"dsv3 L=1 moe", dsv3(1, 0), 2},
		{"dsv3 L=2 dense+moe", dsv3(2, 1), 4},
		{"dsv3 L=4", dsv3(4, 2), 4},
		{"dsv3 L=8", dsv3(8, 3), 4},
		{"dsv3 unrolled E=8", unrolled(8), 2},
		{"dsv3 unrolled E=32", unrolled(32), 2},
	}
	limits := egraph.ExtractionLimits{MaxWork: 1 << 40, MaxNodes: 1 << 24, MaxDepth: 1 << 20}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g, root := saturated(t, tt.spec)
			want, _, _, err := egraph.ExtractDAG(context.Background(), g, root, hopper.Cost{}, limits)
			if err != nil {
				t.Fatal(err)
			}
			got, _, stats, err := egraph.ExtractSegmented(context.Background(), g, root, hopper.Cost{}, limits)
			if err != nil {
				t.Fatal(err)
			}
			if !got.Equal(want) {
				_, _, wl := exprSize(want)
				_, _, gl := exprSize(got)
				t.Errorf("segmented term differs from whole-graph: %d launches, whole-graph %d", gl, wl)
			}
			if stats.Distinct != tt.distinct {
				t.Errorf("extracted %d of %d segments, want %d", stats.Distinct, stats.Segments, tt.distinct)
			}
		})
	}
}

// TestOptimizeScale fails if [hopper.Optimize] stops extracting layer by
// layer: on a 61-layer DeepSeek-V3 stack it takes about 25 ms on an M4
// Max, and whole-graph extraction about 5 minutes and 43 GB.
func TestOptimizeScale(t *testing.T) {
	if testing.Short() {
		t.Skip("builds a 61-layer stack")
	}
	const limit = 60 * time.Second
	s := stackSpec{layers: 61, denseLayers: 3, hidden: 7168, ffn: 18432, experts: 256, topK: 8, expertFFN: 2048}
	b, root := s.build()
	done := make(chan error, 1)
	var e *egraph.Expr
	t0 := time.Now()
	go func() {
		var err error
		e, err = hopper.Optimize(b.G, root)
		done <- err
	}()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(limit):
		t.Fatalf("hopper.Optimize of 61 layers took over %v", limit)
	}
	t.Logf("hopper.Optimize of 61 layers: %v", time.Since(t0))
	want := measurePerLayer(s)
	if want.err != nil {
		t.Fatal(want.err)
	}
	if _, _, got := exprSize(e); got != want.launches {
		t.Errorf("%d launches, want %d (each layer optimized alone)", got, want.launches)
	}
}

// BenchmarkSearchScale times e-graph search on NVL72-scale MoE stacks: a
// DeepSeek-V3-shaped decoder saturated and extracted per layer at its full
// 61 layers, and as one graph at 8 layers, beyond which whole-graph
// ExtractDAG grows superlinearly (see TestSearchScale).
func BenchmarkSearchScale(b *testing.B) {
	dsv3 := func(n int) stackSpec {
		return stackSpec{name: "dsv3", layers: n, denseLayers: min(3, n/2), hidden: 7168, ffn: 18432, experts: 256, topK: 8, expertFFN: 2048}
	}
	b.Run("dsv3/L=61/per-layer", func(b *testing.B) {
		s := dsv3(61)
		for b.Loop() {
			if x := measurePerLayer(s); x.err != nil {
				b.Fatal(x.err)
			}
		}
	})
	b.Run("dsv3/L=61/optimize", func(b *testing.B) {
		s := dsv3(61)
		for b.Loop() {
			g, root := s.build()
			if _, err := hopper.Optimize(g.G, root); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("dsv3/L=8/dag", func(b *testing.B) {
		s := dsv3(8)
		for b.Loop() {
			if r := measure(s); r.err != nil || r.dag.err != nil {
				b.Fatal(r.err, r.dag.err)
			}
		}
	})
}

func slicesUpTo(v []int, n int) []int {
	var out []int
	for _, x := range v {
		if x <= n {
			out = append(out, x)
		}
	}
	return out
}
