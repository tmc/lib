package cudaoptim_test

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaoptim"
	"github.com/tmc/lib/nvidia/x/cudaref"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

// shape is one tensor of the test model: its length, group and decay flag.
type shape struct {
	n       int
	group   int
	noDecay bool
}

// shapes spans a tensor larger than two chunks, a bias-sized tensor, a
// single element, and tensors in both groups with and without decay.
var shapes = []shape{
	{2*cudaoptim.ChunkSize + 1234, 0, false},
	{768, 0, true},
	{1, 0, false},
	{5000, 1, false},
	{300, 1, true},
}

var groupHyper = []struct{ lrScale, wd float32 }{{1, 0.1}, {0.5, 0.01}}

var cfg = cudaoptim.Config{LR: 1e-3, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8, MaxNorm: 50}

// model is the device and host copy of every tensor of the test model.
type model struct {
	arena   *cudamem.Arena
	tensors []cudaoptim.Tensor
	params  [][]float32
}

func newModel(t *testing.T, dev *cuda.Device, seed uint64) *model {
	t.Helper()
	r := rand.New(rand.NewPCG(seed, 1))
	m := &model{arena: cudamem.NewArena(dev)}
	t.Cleanup(func() { m.arena.Free() })
	for i, s := range shapes {
		ts := cudaoptim.Tensor{
			Param:   cudamem.AllocIn[float32](m.arena, "param", s.n),
			Grad:    cudamem.AllocIn[float32](m.arena, "grad", s.n),
			M:       cudamem.AllocIn[float32](m.arena, "m", s.n),
			V:       cudamem.AllocIn[float32](m.arena, "v", s.n),
			NoDecay: s.noDecay,
		}
		if err := m.arena.Err(); err != nil {
			t.Fatal(err)
		}
		p := make([]float32, s.n)
		for j := range p {
			p[j] = float32(r.NormFloat64())
		}
		zero := make([]float32, s.n)
		for _, x := range []struct {
			dst cudamem.Slice[float32]
			src []float32
		}{{ts.Param, p}, {ts.M, zero}, {ts.V, zero}, {ts.Grad, zero}} {
			if err := x.dst.CopyFromHost(x.src); err != nil {
				t.Fatalf("tensor %d: %v", i, err)
			}
		}
		m.tensors = append(m.tensors, ts)
		m.params = append(m.params, p)
	}
	return m
}

func (m *model) groups() []cudaoptim.Group {
	groups := make([]cudaoptim.Group, len(groupHyper))
	for i, h := range groupHyper {
		groups[i].LRScale, groups[i].WeightDecay = h.lrScale, h.wd
	}
	for i, s := range shapes {
		groups[s.group].Tensors = append(groups[s.group].Tensors, m.tensors[i])
	}
	return groups
}

// grads returns the gradients for one step. Their scale varies so that some
// steps are clipped and some are not.
func grads(step int) [][]float32 {
	r := rand.New(rand.NewPCG(uint64(step), 7))
	scale := 0.05
	if step%3 == 0 {
		scale = 1
	}
	out := make([][]float32, len(shapes))
	for i, s := range shapes {
		g := make([]float32, s.n)
		for j := range g {
			g[j] = float32(scale * r.NormFloat64())
		}
		out[i] = g
	}
	return out
}

func (m *model) upload(t *testing.T, g [][]float32) {
	t.Helper()
	for i := range g {
		if err := m.tensors[i].Grad.CopyFromHost(g[i]); err != nil {
			t.Fatal(err)
		}
	}
}

func download(t *testing.T, s cudamem.Slice[float32]) []float32 {
	t.Helper()
	out := make([]float32, s.Len())
	if err := s.CopyToHost(out); err != nil {
		t.Fatal(err)
	}
	return out
}

func newOptimizer(t *testing.T, dev *cuda.Device, m *model) *cudaoptim.AdamW {
	t.Helper()
	opt, err := cudaoptim.NewAdamW(dev, m.groups(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { opt.Close() })
	return opt
}

func sync(t *testing.T, s *cuda.Stream) {
	t.Helper()
	if err := s.Synchronize(context.Background()); err != nil {
		t.Fatal(err)
	}
}

// TestAdamWMatchesReference runs 100 steps and compares every parameter,
// and every step's gradient norm, with the float64 host reference.
//
// Two references run side by side. One is pure float64. The other computes
// each step in float64 but rounds p, m and v to float32 after it, as the
// device stores them. Storage rounding is a random walk of about half an ulp
// per step, and over 100 steps and 150k elements the worst element drifts
// about 1e-6 relative from pure float64, so 1e-6 is required only against
// the rounded reference.
func TestAdamWMatchesReference(t *testing.T) {
	dev := cudatest.Require(t)
	stream := dev.DefaultStream()
	m := newModel(t, dev, 1)
	opt := newOptimizer(t, dev, m)

	type ref struct{ p, m, v [][]float64 }
	var exact, rounded ref
	for _, r := range []*ref{&exact, &rounded} {
		for _, p := range m.params {
			p64 := make([]float64, len(p))
			for j, x := range p {
				p64[j] = float64(x)
			}
			r.p = append(r.p, p64)
			r.m = append(r.m, make([]float64, len(p)))
			r.v = append(r.v, make([]float64, len(p)))
		}
	}
	round := func(xs []float64) {
		for i, x := range xs {
			xs[i] = float64(float32(x))
		}
	}
	const steps = 100
	clipped := 0
	for step := 1; step <= steps; step++ {
		g := grads(step)
		m.upload(t, g)
		if err := opt.Step(stream); err != nil {
			t.Fatal(err)
		}
		g64 := make([][]float64, len(g))
		for i := range g {
			g64[i] = make([]float64, len(g[i]))
			for j, x := range g[i] {
				g64[i][j] = float64(x)
			}
		}
		norm := cudaref.GlobalNorm(g64...)
		clip := cudaref.ClipScale(norm, float64(cfg.MaxNorm))
		if clip < 1 {
			clipped++
		}
		for i, s := range shapes {
			h := groupHyper[s.group]
			c := cudaref.AdamW{
				LR:    float64(cfg.LR) * float64(h.lrScale),
				Beta1: float64(cfg.Beta1), Beta2: float64(cfg.Beta2), Eps: float64(cfg.Eps),
				WeightDecay: float64(h.wd),
			}
			if s.noDecay {
				c.WeightDecay = 0
			}
			c.Step(exact.p[i], exact.m[i], exact.v[i], g64[i], step, clip)
			c.Step(rounded.p[i], rounded.m[i], rounded.v[i], g64[i], step, clip)
			round(rounded.p[i])
			round(rounded.m[i])
			round(rounded.v[i])
		}
		sync(t, stream)
		st, err := opt.ReadState()
		if err != nil {
			t.Fatal(err)
		}
		if st.Step != int32(step) || st.Nonfinite != 0 {
			t.Fatalf("step %d: state %+v", step, st)
		}
		if d := math.Abs(float64(st.Norm)-norm) / norm; d > 1e-6 {
			t.Fatalf("step %d: norm %v, want %v (rel %.2g)", step, st.Norm, norm, d)
		}
	}
	if clipped == 0 || clipped == steps {
		t.Fatalf("clipping engaged on %d of %d steps; the test must cover both", clipped, steps)
	}
	var worstRounded, worstExact float64
	for i := range shapes {
		got := download(t, m.tensors[i].Param)
		for j := range got {
			// rtol=atol=1e-6, as torch.allclose weighs them.
			d := math.Abs(float64(got[j])-rounded.p[i][j]) / (1 + math.Abs(rounded.p[i][j]))
			worstRounded = max(worstRounded, d)
			if d > 1e-6 {
				t.Fatalf("tensor %d element %d after %d steps: %v, want %v", i, j, steps, got[j], rounded.p[i][j])
			}
			d = math.Abs(float64(got[j])-exact.p[i][j]) / (1 + math.Abs(exact.p[i][j]))
			worstExact = max(worstExact, d)
			if d > 1e-5 {
				t.Fatalf("tensor %d element %d after %d steps: %v, want %v in pure float64", i, j, steps, got[j], exact.p[i][j])
			}
		}
	}
	t.Logf("worst |gpu-ref|/(1+|ref|) after %d steps: %.3g against float32-state reference, %.3g against pure float64", steps, worstRounded, worstExact)
}

// TestAdamWReproducible runs the same steps on two identical models and
// requires bitwise-identical parameters and moments.
func TestAdamWReproducible(t *testing.T) {
	dev := cudatest.Require(t)
	stream := dev.DefaultStream()
	var runs [2][][]float32
	for run := range runs {
		m := newModel(t, dev, 2)
		opt := newOptimizer(t, dev, m)
		for step := 1; step <= 20; step++ {
			m.upload(t, grads(step))
			if err := opt.Step(stream); err != nil {
				t.Fatal(err)
			}
		}
		sync(t, stream)
		for _, ts := range m.tensors {
			runs[run] = append(runs[run], download(t, ts.Param), download(t, ts.M), download(t, ts.V))
		}
	}
	for i := range runs[0] {
		for j := range runs[0][i] {
			if math.Float32bits(runs[0][i][j]) != math.Float32bits(runs[1][i][j]) {
				t.Fatalf("buffer %d element %d differs between runs: %v vs %v", i, j, runs[0][i][j], runs[1][i][j])
			}
		}
	}
}

// TestAdamWNonfinite injects a NaN or an infinity into one element of one
// gradient and requires the flag to be set and the step to change nothing.
func TestAdamWNonfinite(t *testing.T) {
	dev := cudatest.Require(t)
	stream := dev.DefaultStream()
	for _, tt := range []struct {
		name   string
		tensor int
		value  float32
	}{
		{"nan in large tensor", 0, float32(math.NaN())},
		{"inf in bias", 1, float32(math.Inf(-1))},
	} {
		t.Run(tt.name, func(t *testing.T) {
			m := newModel(t, dev, 3)
			opt := newOptimizer(t, dev, m)
			for step := 1; step <= 3; step++ {
				m.upload(t, grads(step))
				if err := opt.Step(stream); err != nil {
					t.Fatal(err)
				}
			}
			sync(t, stream)
			before := download(t, m.tensors[3].Param)
			g := grads(4)
			idx := 17
			if tt.tensor == 0 {
				idx += cudaoptim.ChunkSize // a chunk other than the first
			}
			g[tt.tensor][idx] = tt.value
			m.upload(t, g)
			if err := opt.Step(stream); err != nil {
				t.Fatal(err)
			}
			sync(t, stream)
			st, err := opt.ReadState()
			if err != nil {
				t.Fatal(err)
			}
			if st.Nonfinite == 0 || st.Step != 3 {
				t.Fatalf("state after bad step = %+v, want Nonfinite set and Step 3", st)
			}
			after := download(t, m.tensors[3].Param)
			for j := range before {
				if math.Float32bits(before[j]) != math.Float32bits(after[j]) {
					t.Fatalf("parameter %d changed on a non-finite step", j)
				}
			}
			// Clearing the flag lets the next finite step through.
			m.upload(t, grads(5))
			if err := opt.ClearNonfinite(stream); err != nil {
				t.Fatal(err)
			}
			if err := opt.Step(stream); err != nil {
				t.Fatal(err)
			}
			sync(t, stream)
			if st, err = opt.ReadState(); err != nil || st.Nonfinite != 0 || st.Step != 4 {
				t.Fatalf("state after recovery = %+v, %v", st, err)
			}
		})
	}
}

// TestAdamWSetLR checks that SetLR reaches the device state.
func TestAdamWSetLR(t *testing.T) {
	dev := cudatest.Require(t)
	stream := dev.DefaultStream()
	m := newModel(t, dev, 4)
	opt := newOptimizer(t, dev, m)
	if err := opt.SetLR(stream, 0.125); err != nil {
		t.Fatal(err)
	}
	sync(t, stream)
	st, err := opt.ReadState()
	if err != nil {
		t.Fatal(err)
	}
	if st.LR != 0.125 {
		t.Fatalf("LR = %v, want 0.125", st.LR)
	}
}
