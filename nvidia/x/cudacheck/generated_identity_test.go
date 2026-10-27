package cudacheck

import (
	"math"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
)

func TestGeneratedTraceNodeIdentity(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("shared", []int{2}, cudair.Float32)
	y := g.AddInput("shared", []int{2}, cudair.Float32)
	a := g.AddBinary("add", x, y)
	b := g.AddBinary("mul", a, a)
	a.Name, b.Name = "", ""
	if err := g.SetOutputs(a, b, b); err != nil {
		t.Fatal(err)
	}
	src, _, err := synthesizeGraphKernel(g)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"const float *__restrict__ in_n0", "const float *__restrict__ in_n1",
		"float var_n2 = (in_n0[idx] + in_n1[idx]);",
		"float var_n3 = (var_n2 * var_n2);",
		"trace_ordered_float(var_n2, 0U + (unsigned int)idx, trace_fold_limit)",
		"trace_ordered_float(var_n3, 2U + (unsigned int)idx, trace_fold_limit)",
		"out_0[idx] = var_n2;", "out_1[idx] = var_n3;", "out_2[idx] = var_n3;",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("generated source missing %q", want)
		}
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	f := newCachedTraceFixture(t)
	cfg := &CompareConfig{artifactCache: &perturbationCacheConfig{adapter: f.adapter, target: "compute_80"}}
	request, err := makeTraceCacheRequest(cfg, src, "eval_graph_kernel", snap, true)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"num_elements", "trace_fold_limit", "in_n0", "in_n1", "out_0", "out_1", "out_2"}
	if len(request.schema.Params) != len(want) {
		t.Fatalf("cache schema parameters = %d, want %d", len(request.schema.Params), len(want))
	}
	for i, p := range request.schema.Params {
		if p.Name != want[i] {
			t.Errorf("parameter[%d] = %q, want %q", i, p.Name, want[i])
		}
	}
	acc, err := EvaluateCPUGraph(g, map[string][]float32{"shared": {1, 2}}, PolicyRawBits, true)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range []float64{2, 4, 4, 16} {
		if got := acc.Events()[i].Value; got != v {
			t.Errorf("fold[%d] = %v, want %v", i, got, v)
		}
	}
}

func TestGeneratedTraceSignedNodeIdentity(t *testing.T) {
	seen := make(map[string]int)
	for _, id := range []int{math.MinInt, -3, -2, -1, 0, 1, 2, 3, math.MaxInt} {
		symbol := traceNodeSymbol(&cudair.Node{ID: id})
		if prior, ok := seen[symbol]; ok {
			t.Fatalf("IDs %d and %d share symbol %q", prior, id, symbol)
		}
		seen[symbol] = id
	}
	minimum := [3]string{"n_neg9223372036854775808", "n_neg9223372036854775807", "n_neg9223372036854775806"}
	if strconv.IntSize == 32 {
		minimum = [3]string{"n_neg2147483648", "n_neg2147483647", "n_neg2147483646"}
	}
	for _, tc := range []struct {
		name    string
		ids     [3]int
		symbols [3]string
	}{
		{"negative", [3]int{-1, -2, -3}, [3]string{"n_neg1", "n_neg2", "n_neg3"}},
		{"minimum", [3]int{math.MinInt, math.MinInt + 1, math.MinInt + 2}, minimum},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := cudair.NewGraph()
			x := g.AddInput("x", []int{1}, cudair.Float32)
			a := g.AddUnary("neg", x)
			b := g.AddBinary("mul", a, a)
			x.ID, a.ID, b.ID = tc.ids[0], tc.ids[1], tc.ids[2]
			a.Name, b.Name = "same", "same"
			if err := g.SetOutputs(a, b); err != nil {
				t.Fatal(err)
			}
			src, _, err := synthesizeGraphKernel(g)
			if err != nil {
				t.Fatal(err)
			}
			for _, want := range []string{
				"const float *__restrict__ in_" + tc.symbols[0],
				"float var_" + tc.symbols[1] + " = (-in_" + tc.symbols[0] + "[idx]);",
				"float var_" + tc.symbols[2] + " = (var_" + tc.symbols[1] + " * var_" + tc.symbols[1] + ");",
				"trace_ordered_float(var_" + tc.symbols[1] + ", 0U + (unsigned int)idx, trace_fold_limit)",
				"trace_ordered_float(var_" + tc.symbols[2] + ", 1U + (unsigned int)idx, trace_fold_limit)",
				"out_0[idx] = var_" + tc.symbols[1] + ";", "out_1[idx] = var_" + tc.symbols[2] + ";",
			} {
				if !strings.Contains(src, want) {
					t.Errorf("signed-ID source missing %q", want)
				}
			}
			snap, err := g.Snapshot()
			if err != nil {
				t.Fatal(err)
			}
			f := newCachedTraceFixture(t)
			cfg := &CompareConfig{artifactCache: &perturbationCacheConfig{adapter: f.adapter, target: "compute_80"}}
			request, err := makeTraceCacheRequest(cfg, src, "eval_graph_kernel", snap, true)
			if err != nil {
				t.Fatal(err)
			}
			if got := request.schema.Params[2].Name; got != "in_"+tc.symbols[0] {
				t.Errorf("input schema symbol = %q", got)
			}
			for i, want := range []string{"out_0", "out_1"} {
				if got := request.schema.Params[3+i].Name; got != want {
					t.Errorf("output schema symbol = %q, want %q", got, want)
				}
			}
			acc, err := EvaluateCPUGraph(g, map[string][]float32{"x": {-2}}, PolicyRawBits, true)
			if err != nil {
				t.Fatal(err)
			}
			if events := acc.Events(); len(events) != 2 || events[0].Value != 2 || events[1].Value != 4 {
				t.Fatalf("signed-ID CPU folds = %+v, want 2,4", events)
			}
		})
	}
}
