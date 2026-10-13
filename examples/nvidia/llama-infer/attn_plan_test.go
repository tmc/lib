package main

import (
	"math"
	"reflect"
	"testing"
)

func TestAttnPlanRefusals(t *testing.T) {
	for _, tt := range []struct {
		name                                     string
		hidden, qHeads, kvHeads, headDim, maxSeq int
		eps, theta                               float32
		budget                                   int
	}{
		{"zero hidden", 0, 2, 2, 4, 9, 1e-5, 1e4, 1000},
		{"zero query heads", 8, 0, 2, 4, 9, 1e-5, 1e4, 1000},
		{"negative kv heads", 8, 2, -1, 4, 9, 1e-5, 1e4, 1000},
		{"zero head dim", 8, 2, 2, 0, 9, 1e-5, 1e4, 1000},
		{"zero max seq", 8, 2, 2, 4, 0, 1e-5, 1e4, 1000},
		{"heads not grouped", 8, 3, 2, 4, 9, 1e-5, 1e4, 1000},
		{"odd head dim", 8, 2, 2, 3, 9, 1e-5, 1e4, 1000},
		{"zero epsilon", 8, 2, 2, 4, 9, 0, 1e4, 1000},
		{"nan epsilon", 8, 2, 2, 4, 9, float32(math.NaN()), 1e4, 1000},
		{"infinite epsilon", 8, 2, 2, 4, 9, float32(math.Inf(1)), 1e4, 1000},
		{"zero theta", 8, 2, 2, 4, 9, 1e-5, 0, 1000},
		{"negative theta", 8, 2, 2, 4, 9, 1e-5, -1e4, 1000},
		{"nan theta", 8, 2, 2, 4, 9, 1e-5, float32(math.NaN()), 1000},
		{"infinite theta", 8, 2, 2, 4, 9, 1e-5, float32(math.Inf(1)), 1000},
		{"query bytes", 8, 1 << 32, 1, 1 << 32, 9, 1e-5, 1e4, math.MaxInt},
		{"matrix bytes", 1 << 32, 1, 1, 1 << 32, 9, 1e-5, 1e4, math.MaxInt},
		{"cache bytes", 1, 1, 1, 1 << 20, 1 << 42, 1e-5, 1e4, math.MaxInt},
		{"scratch bytes", math.MaxInt/32 + 1, 1, 1, 2, 9, 1e-5, 1e4, math.MaxInt},
		{"short budget", 8, 2, 2, 4, 9, 1e-5, 1e4, 227},
		{"negative budget", 8, 2, 2, 4, 9, 1e-5, 1e4, -1},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := newAttnPlan(tt.hidden, tt.qHeads, tt.kvHeads, tt.headDim, tt.maxSeq, tt.eps, tt.theta, tt.budget); err == nil {
				t.Fatal("accepted invalid plan")
			}
		})
	}
	p, err := newAttnPlan(8, 2, 2, 4, 9, 1e-5, 1e4, 228)
	if err != nil {
		t.Fatal(err)
	}
	if p.qDim != 8 || p.kvDim != 8 || p.scratchWords != 57 {
		t.Fatalf("counts: %+v", p)
	}
	p, err = newAttnPlan(7, 2, 1, 6, 5, 1e-5, 1e4, 4*(2*7+2*12+2*6+5))
	if err != nil {
		t.Fatal(err)
	}
	if p.qDim != 12 || p.kvDim != 6 {
		t.Fatalf("query width independent of hidden: %+v", p)
	}
}

func TestAttnPlanLengths(t *testing.T) {
	p, err := newAttnPlan(7, 4, 2, 4, 5, 1e-5, 1e4, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	valid := func() (attnBuffers, attnCache) {
		return attnBuffers{x: make([]float32, 7), norm: make([]float32, 7),
				wq: make([]float32, 16*7), wk: make([]float32, 8*7), wv: make([]float32, 8*7), wo: make([]float32, 7*16),
				qBias: make([]float32, 16), kBias: make([]float32, 8), vBias: make([]float32, 8)},
			attnCache{k: make([]float32, 5*8), v: make([]float32, 5*8)}
	}
	b, c := valid()
	if _, err := p.reference(b, c, 4); err != nil {
		t.Fatalf("valid buffers refused: %v", err)
	}
	b.qBias, b.kBias, b.vBias = nil, nil, nil
	if _, err := p.reference(b, c, 0); err != nil {
		t.Fatalf("nil biases refused: %v", err)
	}
	for _, name := range []string{"x", "norm", "wq", "wk", "wv", "wo", "qBias", "kBias", "vBias", "k", "v"} {
		for _, delta := range []int{-1, 1} {
			b, c := valid()
			v := map[string]*[]float32{"x": &b.x, "norm": &b.norm, "wq": &b.wq, "wk": &b.wk, "wv": &b.wv, "wo": &b.wo,
				"qBias": &b.qBias, "kBias": &b.kBias, "vBias": &b.vBias, "k": &c.k, "v": &c.v}[name]
			*v = make([]float32, len(*v)+delta)
			if out, err := p.reference(b, c, 0); err == nil || out != nil {
				t.Fatalf("%s delta %d accepted", name, delta)
			}
		}
	}
	for _, pos := range []int{-1, 5} {
		b, c := valid()
		if out, err := p.reference(b, c, pos); err == nil || out != nil {
			t.Fatalf("position %d accepted", pos)
		}
	}
	b, c = valid()
	if out, err := (attnPlan{}).reference(b, c, 0); err == nil || out != nil {
		t.Fatal("zero plan accepted")
	}
	p.scratchWords--
	if out, err := p.reference(b, c, 0); err == nil || out != nil {
		t.Fatal("corrupt plan accepted")
	}
}

func TestAttnPlanDependencies(t *testing.T) {
	p, err := newAttnPlan(8, 2, 2, 4, 9, 1e-5, 1e4, 228)
	if err != nil {
		t.Fatal(err)
	}
	want := []attnStage{
		{name: "normalize"},
		{name: "q", after: []string{"normalize"}},
		{name: "k", after: []string{"normalize"}},
		{name: "v", after: []string{"normalize"}},
		{name: "rope", after: []string{"q", "k"}},
		{name: "append", after: []string{"rope", "v"}},
		{name: "attend", after: []string{"append"}},
		{name: "project", after: []string{"attend"}},
		{name: "residual", after: []string{"project"}},
	}
	got := p.stages()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("stages: %#v", got)
	}
	got[5].after[0] = "changed"
	if !reflect.DeepEqual(p.stages(), want) {
		t.Fatal("stage dependencies alias caller storage")
	}
}
