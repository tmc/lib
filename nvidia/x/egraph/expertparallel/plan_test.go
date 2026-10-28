package expertparallel

import (
	"encoding/binary"
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// testConfig is a small DeepSeek-style layer: sigmoid scores with a
// selection-only bias, normalized top-8 of 32 experts.
func testConfig(ranks, tokens, hidden, inter, shared int) Config {
	return Config{
		Ranks: ranks, Tokens: tokens, Hidden: hidden, Inter: inter,
		Experts: 32, TopK: 8, SharedInter: shared,
		Route: hopper.RowTopK{
			Bias:      hopper.F32,
			Score:     hopper.MustLambda("(sigmoid (mul $0 0.1767766952966369))"),
			Key:       hopper.MustLambda("(add $0 $1)"),
			Normalize: true, Eps: 1e-15, Scale: 1,
		},
	}
}

// testEnv returns random values for cfg's inputs: BF16 activations,
// NVFP4 weights with E4M3 block scales, and logits that skew some tokens
// toward one rank's experts so ranks receive unequal loads.
func testEnv(cfg Config, seed uint64) map[string]hopper.Value {
	r := rand.New(rand.NewPCG(seed, seed+1))
	env := make(map[string]hopper.Value)
	for _, t := range cfg.Inputs() {
		v := make([]float64, size(t.Shape))
		switch {
		case t.Name == "logits":
			for i := range v {
				v[i] = float64(float32(r.NormFloat64() * 4))
			}
			for tok := 0; tok < cfg.Tokens; tok += 3 {
				for e := range 8 {
					v[tok*cfg.Experts+e] += 20
				}
			}
		case t.Name == "bias":
			for i := range v {
				v[i] = float64(float32(r.NormFloat64() * 0.1))
			}
		case strings.HasSuffix(t.Name, "_q"):
			for i := range v {
				v[i] = float64(r.IntN(256))
			}
		case strings.HasSuffix(t.Name, "_sf"):
			for i := range v {
				v[i] = float64(hopper.E4M3Bits(float32(8 + r.Float64()*400)))
			}
		case strings.HasSuffix(t.Name, "_gs"):
			for i := range v {
				v[i] = float64(float32(0.002 + r.Float64()*0.002))
			}
		case t.Name == "post_w":
			for i := range v {
				v[i] = float64(hopper.BF16.Round(float32(r.NormFloat64() * 0.1)))
			}
		default:
			for i := range v {
				v[i] = float64(hopper.BF16.Round(float32(r.NormFloat64())))
			}
		}
		env[t.Name] = hopper.Value{Shape: t.Shape, Data: v}
	}
	return env
}

func TestLayoutReference(t *testing.T) {
	tests := []struct {
		name         string
		ids          []int32
		ranks, local int
		want         Layout
	}{
		{"one rank", []int32{3, 1, 2}, 1, 4, Layout{[]int32{0, 3}, []int32{0, 1, 2}, []int32{0, 1, 2}}},
		{"stable", []int32{5, 0, 4, 1}, 2, 4, Layout{[]int32{0, 2, 4}, []int32{2, 0, 3, 1}, []int32{1, 3, 0, 2}}},
		{"empty rank", []int32{0, 1, 2, 3}, 3, 2, Layout{[]int32{0, 2, 4, 4}, []int32{0, 1, 2, 3}, []int32{0, 1, 2, 3}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := LayoutReference(tt.ids, tt.ranks, tt.local)
			if !slices.Equal(got.SendOff, tt.want.SendOff) || !slices.Equal(got.SendPos, tt.want.SendPos) || !slices.Equal(got.SendRoute, tt.want.SendRoute) {
				t.Errorf("LayoutReference(%v) = %+v, want %+v", tt.ids, got, tt.want)
			}
		})
	}
}

func TestNewErrors(t *testing.T) {
	tests := []struct {
		name string
		cfg  Config
	}{
		{"tokens", testConfig(3, 8, 128, 64, 0)},
		{"experts per rank", func() Config { c := testConfig(1, 8, 128, 64, 0); c.Experts = 64; return c }()},
		{"hidden", testConfig(1, 8, 100, 64, 0)},
		{"bias dtype", func() Config { c := testConfig(1, 8, 128, 64, 0); c.Route.Bias = hopper.BF16; return c }()},
	}
	for _, tt := range tests {
		if _, err := New(tt.cfg, "ep"); err == nil {
			t.Errorf("%s: New succeeded, want error", tt.name)
		}
	}
	if _, err := New(testConfig(1, 8, 128, 64, 0), "e p"); err == nil {
		t.Error("New with a bad name succeeded")
	}
}

func TestStagesAndCollectives(t *testing.T) {
	p, err := New(testConfig(4, 8, 128, 64, 64), "ep")
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, s := range p.Stages() {
		names = append(names, s.Name)
	}
	if got, want := strings.Join(names, ","), "route,layout,shared,dispatch,expert,combine"; got != want {
		t.Errorf("stages %s, want %s", got, want)
	}
	// The expert stage fuses to align, W13+SwiGLU, W2+sumsq and expert sum.
	if n := len(p.Expert.Kernels); n != 4 {
		t.Errorf("expert stage has %d kernels, want 4", n)
	}
	cs, err := p.Collectives("ep", 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range cs {
		d, err := c.OperationDescriptor()
		if err != nil {
			t.Fatal(err)
		}
		if d.Kind.String() != "AllToAll" || c.WorldSize != 4 || c.Count != uint64(p.Capacity()*p.Hidden) {
			t.Errorf("collective %+v (%v)", c, d.Kind)
		}
	}
	if egraph.CanRewrite(cs[0], cs[1]) {
		t.Error("dispatch and combine may be reordered; want distinct effects")
	}
}

// TestReferenceRankInvariant checks that the CPU reference of EP=2 and
// EP=4 reproduces EP=1 bit for bit, and that EP=1 agrees with the
// single-rank dense term.
func TestReferenceRankInvariant(t *testing.T) {
	for _, shared := range []int{0, 64} {
		var want []float32
		for _, ranks := range []int{1, 2, 4} {
			cfg := testConfig(ranks, 16, 128, 64, shared)
			p, err := New(cfg, "ep")
			if err != nil {
				t.Fatal(err)
			}
			got, err := p.Reference(testEnv(cfg, 7))
			if err != nil {
				t.Fatal(err)
			}
			if want == nil {
				want = got
				continue
			}
			if i := firstDiff(got, want); i >= 0 {
				t.Errorf("shared %d: EP=%d differs from EP=1 at %d: %g vs %g", shared, ranks, i, got[i], want[i])
			}
		}
		if shared != 0 {
			continue
		}
		cfg := testConfig(1, 16, 128, 64, 0)
		b, root, err := cfg.Dense()
		if err != nil {
			t.Fatal(err)
		}
		e, err := hopper.Optimize(b.G, root)
		if err != nil {
			t.Fatal(err)
		}
		env := testEnv(cfg, 7)
		env["local_start"] = hopper.Value{Shape: []int{1}, Data: []float64{0}}
		dense, err := hopper.Eval(e, env)
		if err != nil {
			t.Fatal(err)
		}
		if e := relErr(want, roundBF16(dense.Data)); e > 1e-3 {
			t.Errorf("EP=1 vs dense term: relative L2 error %.2e > 1e-3", e)
		}
	}
}

// TestEmulated runs the emitted multi-rank source under cudahostemu for
// EP=1, 2 and 4 and checks that the outputs are identical and match the
// CPU reference.
func TestEmulated(t *testing.T) {
	if testing.Short() {
		t.Skip("emulated multi-rank MoE is slow")
	}
	for _, shared := range []int{0, 64} {
		var want []float32
		for _, ranks := range []int{1, 2, 4} {
			cfg := testConfig(ranks, 8, 128, 64, shared)
			p, err := New(cfg, "ep")
			if err != nil {
				t.Fatal(err)
			}
			env := testEnv(cfg, 3)
			got := runEmulated(t, p, env)
			ref, err := p.Reference(env)
			if err != nil {
				t.Fatal(err)
			}
			if e := relErr(got, ref); e > 5e-3 || math.IsNaN(e) {
				t.Errorf("shared %d EP=%d: emulated vs reference relative L2 error %.2e > 5e-3", shared, ranks, e)
			}
			if want == nil {
				want = got
				continue
			}
			if i := firstDiff(got, want); i >= 0 {
				t.Errorf("shared %d: emulated EP=%d differs from EP=1 at %d: %g vs %g", shared, ranks, i, got[i], want[i])
			}
		}
	}
}

func runEmulated(t *testing.T, p *Plan, env map[string]hopper.Value) []float32 {
	t.Helper()
	src, err := p.Source()
	if err != nil {
		t.Fatal(err)
	}
	var bufs []cudahostemu.Buffer
	var names []string
	for _, in := range p.Inputs() {
		bufs = append(bufs, cudahostemu.Buffer{Name: in.Name, CType: ctype(in.DType), Data: encode(in.DType, env[in.Name].Data)})
		names = append(names, in.Name)
	}
	bufs = append(bufs,
		cudahostemu.Buffer{Name: "out", CType: "uint16_t", Data: cudahostemu.Poison(p.Tokens * p.Hidden * 2)},
		cudahostemu.Buffer{Name: "ws", CType: "uint8_t", Data: cudahostemu.Poison(p.WorkspaceBytes())})
	res, err := cudahostemu.Run(t.Context(), t.TempDir(), cudahostemu.Program{
		Source:  src,
		Launch:  p.Name + "_run(" + strings.Join(append(names, "out", "ws"), ", ") + ")",
		Buffers: bufs,
	})
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip("no host C++ compiler")
	}
	if err != nil {
		t.Fatal(err)
	}
	return decodeBF16(res["out"])
}

// encode returns v in d's storage: integer codes for U8 and I32, values
// for F32 and BF16.
func encode(d hopper.DType, v []float64) []byte {
	var b []byte
	for _, f := range v {
		switch d {
		case hopper.F32:
			b = binary.LittleEndian.AppendUint32(b, math.Float32bits(float32(f)))
		case hopper.BF16:
			b = binary.LittleEndian.AppendUint16(b, uint16(math.Float32bits(hopper.BF16.Round(float32(f)))>>16))
		case hopper.I32:
			b = binary.LittleEndian.AppendUint32(b, uint32(int32(f)))
		default:
			b = append(b, uint8(f))
		}
	}
	return b
}

func decodeBF16(b []byte) []float32 {
	out := make([]float32, len(b)/2)
	for i := range out {
		out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(b[2*i:])) << 16)
	}
	return out
}

func roundBF16(v []float64) []float32 {
	out := make([]float32, len(v))
	for i, f := range v {
		out[i] = hopper.BF16.Round(float32(f))
	}
	return out
}

func firstDiff(a, b []float32) int {
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return i
		}
	}
	if len(a) != len(b) {
		return min(len(a), len(b))
	}
	return -1
}

func relErr(got, want []float32) float64 {
	var num, den float64
	for i := range want {
		d := float64(got[i]) - float64(want[i])
		num += d * d
		den += float64(want[i]) * float64(want[i])
	}
	if den == 0 {
		return math.Sqrt(num)
	}
	return math.Sqrt(num / den)
}
