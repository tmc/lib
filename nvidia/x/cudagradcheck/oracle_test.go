package cudagradcheck

import (
	"math"
	"math/rand/v2"
	"os"
	"regexp"
	"slices"
	"strings"
	"testing"
)

func randn(r *rand.Rand, n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = r.NormFloat64()
	}
	return x
}

// gradCase is one input of one oracle: f is the forward reduced to a scalar
// by a fixed upstream gradient, and grad is the oracle's backward for that
// input. Both close over the other inputs.
type gradCase struct {
	name string
	x    []float64
	f    func(x []float64) float64
	grad func(x []float64) []float64
}

func gradCases() []gradCase {
	r := rand.New(rand.NewPCG(1, 2))
	var cases []gradCase

	const rows, n, eps = 3, 5, 1e-5
	x, w, b := randn(r, rows*n), randn(r, n), randn(r, n)
	dy := randn(r, rows*n)
	for _, bias := range []bool{true, false} {
		bb, name := b, "layernorm"
		if !bias {
			bb, name = nil, "layernorm no bias"
		}
		cases = append(cases,
			gradCase{name + " x", x,
				func(x []float64) float64 { y, _, _ := LayerNorm(x, w, bb, n, eps); return Dot(y, dy) },
				func(x []float64) []float64 {
					_, mu, rs := LayerNorm(x, w, bb, n, eps)
					dx, _, _ := LayerNormBackward(dy, x, w, mu, rs, n)
					return dx
				}},
			gradCase{name + " w", w,
				func(w []float64) float64 { y, _, _ := LayerNorm(x, w, bb, n, eps); return Dot(y, dy) },
				func(w []float64) []float64 {
					_, mu, rs := LayerNorm(x, w, bb, n, eps)
					_, dw, _ := LayerNormBackward(dy, x, w, mu, rs, n)
					return dw
				}},
		)
	}
	cases = append(cases, gradCase{"layernorm b", b,
		func(b []float64) float64 { y, _, _ := LayerNorm(x, w, b, n, eps); return Dot(y, dy) },
		func(b []float64) []float64 {
			_, mu, rs := LayerNorm(x, w, b, n, eps)
			_, _, db := LayerNormBackward(dy, x, w, mu, rs, n)
			return db
		}})

	cases = append(cases, gradCase{"softmax", randn(r, rows*n),
		func(x []float64) float64 { return Dot(Softmax(x, n), dy) },
		func(x []float64) []float64 { return SoftmaxBackward(Softmax(x, n), dy, n) }})

	gx := []float64{-4, -1.5, -0.3, 0, 0.2, 1, 2.5, 6}
	gdy := randn(r, len(gx))
	cases = append(cases, gradCase{"gelu", gx,
		func(x []float64) float64 { return Dot(GELU(x), gdy) },
		func(x []float64) []float64 { return GELUBackward(x, gdy) }})

	gg := randn(r, rows*2*n)
	cases = append(cases, gradCase{"geglu", gg,
		func(x []float64) float64 { return Dot(GeGLU(x, n), dy) },
		func(x []float64) []float64 { return GeGLUBackward(x, dy, n) }})

	pos := []int{0, 1, 7, 100}
	const heads, dim = 2, 8
	rdy := randn(r, len(pos)*heads*dim)
	cases = append(cases, gradCase{"rope", randn(r, len(pos)*heads*dim),
		func(x []float64) float64 { return Dot(RoPE(x, pos, heads, dim, 10000), rdy) },
		func(x []float64) []float64 { return RoPEBackward(rdy, pos, heads, dim, 10000) }})

	for _, a := range []struct {
		name   string
		tq, tk int
		m      Mask
	}{
		{"full", 5, 6, Full},
		{"causal", 6, 6, Causal},
		{"window", 6, 6, SlidingWindow(1)},
		{"ragged", 7, 7, Segments([]int{4, 2}, Causal)}, // row 6 is padding
		{"masked row", 4, 5, func(i, j int) bool { return i != 2 && j != 4 }},
	} {
		const d = 4
		scale := 1 / math.Sqrt(d)
		q, k, v := randn(r, a.tq*d), randn(r, a.tk*d), randn(r, a.tk*d)
		dout := randn(r, a.tq*d)
		fwd := func(q, k, v []float64) float64 {
			out, _ := Attention(q, k, v, a.tq, a.tk, d, scale, a.m)
			return Dot(out, dout)
		}
		bwd := func(q, k, v []float64) (dq, dk, dv []float64) {
			return AttentionBackward(q, k, v, dout, a.tq, a.tk, d, scale, a.m)
		}
		cases = append(cases,
			gradCase{"attention " + a.name + " q", q,
				func(q []float64) float64 { return fwd(q, k, v) },
				func(q []float64) []float64 { dq, _, _ := bwd(q, k, v); return dq }},
			gradCase{"attention " + a.name + " k", k,
				func(k []float64) float64 { return fwd(q, k, v) },
				func(k []float64) []float64 { _, dk, _ := bwd(q, k, v); return dk }},
			gradCase{"attention " + a.name + " v", v,
				func(v []float64) float64 { return fwd(q, k, v) },
				func(v []float64) []float64 { _, _, dv := bwd(q, k, v); return dv }},
		)
	}
	return cases
}

// TestOracleGradients checks every backward oracle against central
// differences of its own forward at the default tolerance.
func TestOracleGradients(t *testing.T) {
	for _, c := range gradCases() {
		t.Run(c.name, func(t *testing.T) {
			r, err := CheckGrad(c.f, c.x, c.grad(c.x), GradOptions{})
			if err != nil {
				t.Fatal(err)
			}
			if r.Ratio > 1e-2 {
				t.Errorf("worst %v; expected far inside tolerance in float64", r)
			}
		})
	}
}

// TestMutations removes one term at a time from a backward oracle and
// requires CheckGrad to fail on the named case.
func TestMutations(t *testing.T) {
	mutations := []struct{ term, fails string }{
		{"layernorm.mean", "layernorm x"},
		{"layernorm.var", "layernorm x"},
		{"layernorm.mean", "layernorm no bias x"},
		{"softmax.sum", "softmax"},
		{"gelu.pdf", "gelu"},
		{"geglu.gate", "geglu"},
		{"rope.sign", "rope"},
		{"attention.delta", "attention full q"},
		{"attention.delta", "attention causal k"},
		{"attention.dk-scale", "attention full k"},
		{"attention.mask", "attention window q"},
		{"attention.mask", "attention ragged v"},
		{"attention.mask", "attention masked row q"},
	}
	cases := gradCases()
	for _, m := range mutations {
		t.Run(m.term+"/"+m.fails, func(t *testing.T) {
			i := slices.IndexFunc(cases, func(c gradCase) bool { return c.name == m.fails })
			if i < 0 {
				t.Fatalf("no case %q", m.fails)
			}
			c := cases[i]
			dropped = m.term
			defer func() { dropped = "" }()
			r, err := CheckGrad(c.f, c.x, c.grad(c.x), GradOptions{})
			if err == nil {
				t.Fatalf("dropping %s passed: worst %v", m.term, r)
			}
			t.Logf("dropping %s: worst %v", m.term, r)
		})
	}

	// Every droppable term in the oracles must have a mutation above.
	re := regexp.MustCompile(`(?:keep\(|dropped == )"([a-z.-]+)"`)
	var terms []string
	for _, name := range []string{"oracle.go", "attention.go"} {
		src, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		for _, m := range re.FindAllStringSubmatch(string(src), -1) {
			terms = append(terms, m[1])
		}
	}
	if len(terms) == 0 {
		t.Fatal("found no droppable terms in the oracle sources")
	}
	for _, term := range terms {
		if !slices.ContainsFunc(mutations, func(m struct{ term, fails string }) bool { return m.term == term }) {
			t.Errorf("term %q has no mutation test", term)
		}
	}
}

// TestForwardValues pins each forward to values computed independently, since
// CheckGrad only shows that a backward matches its own forward.
func TestForwardValues(t *testing.T) {
	near := func(name string, got, want []float64) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s: got %d values, want %d", name, len(got), len(want))
		}
		for i := range got {
			if math.Abs(got[i]-want[i]) > 1e-12*(1+math.Abs(want[i])) {
				t.Errorf("%s[%d] = %.17g, want %.17g", name, i, got[i], want[i])
			}
		}
	}
	y, mu, rs := LayerNorm([]float64{1, 2, 3}, []float64{1, 2, 1}, []float64{0, 0, 1}, 3, 0)
	s := math.Sqrt(1.5)
	near("layernorm", y, []float64{-s, 0, 1 + s})
	near("layernorm stats", []float64{mu[0], rs[0]}, []float64{2, s})

	near("softmax", Softmax([]float64{0, math.Ln2, 5, 5}, 2), []float64{1. / 3, 2. / 3, 0.5, 0.5})
	// torch.nn.functional.gelu(torch.tensor([1., -2.], dtype=torch.float64))
	near("gelu", GELU([]float64{1, -2}), []float64{0.8413447460685429, -0.04550026389635842})
	near("geglu", GeGLU([]float64{1, -2, 3, 0.5}, 2), []float64{3 * 0.8413447460685429, 0.5 * -0.04550026389635842})

	// Position 1 of a 2-element head rotates by one radian; position 0 not at all.
	sin1, cos1 := math.Sincos(1)
	near("rope", RoPE([]float64{1, 2, 1, 2}, []int{0, 1}, 1, 2, 10000), []float64{1, 2, cos1 - 2*sin1, 2*cos1 + sin1})

	// One visible key returns its value; equal scores average the values.
	out, lse := Attention([]float64{1, 0, 0, 1}, []float64{1, 0, 0, 1}, []float64{3, 4, 5, 6}, 2, 2, 2, 0, Causal)
	near("attention", out, []float64{3, 4, 4, 5})
	near("attention lse", lse, []float64{0, math.Ln2})
	out, lse = Attention([]float64{1, 1}, []float64{1, 1}, []float64{7, 8}, 1, 1, 2, 1, func(i, j int) bool { return false })
	near("attention masked", out, []float64{0, 0})
	if !math.IsInf(lse[0], -1) {
		t.Errorf("masked lse = %v, want -Inf", lse[0])
	}
}

func TestSegments(t *testing.T) {
	m := Segments([]int{3, 2}, Causal)
	var b strings.Builder
	for i := range 6 {
		for j := range 6 {
			if m(i, j) {
				b.WriteByte('x')
			} else {
				b.WriteByte('.')
			}
		}
		b.WriteByte('\n')
	}
	want := "x.....\nxx....\nxxx...\n...x..\n...xx.\n......\n"
	if b.String() != want {
		t.Errorf("mask:\n%swant:\n%s", b.String(), want)
	}
}
