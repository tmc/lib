package hopper

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/tcgen05"
)

func ssmEnv(r *rand.Rand, heads int, cfg tcgen05.SSMChunkSchedule) map[string]Value {
	t := cfg.SeqLen()
	val := func(shape []int, f func() float64) Value {
		v := Value{Shape: shape, Data: make([]float64, size(shape))}
		for i := range v.Data {
			v.Data[i] = float64(float32(f()))
		}
		return v
	}
	norm := func(s float64) func() float64 { return func() float64 { return r.NormFloat64() * s } }
	return map[string]Value{
		"q":     val([]int{heads, t, cfg.HeadDimK}, norm(0.2)),
		"k":     val([]int{heads, t, cfg.HeadDimK}, norm(0.2)),
		"v":     val([]int{heads, t, cfg.HeadDimV}, norm(1)),
		"alpha": val([]int{heads, t}, func() float64 { return 0.9 + 0.1*r.Float64() }),
		"beta":  val([]int{heads, t}, func() float64 { return 0.5 * r.Float64() }),
		"gate":  val([]int{heads, t, cfg.HeadDimV}, norm(1)),
	}
}

func buildSSM(b *Builder, v tcgen05.SSMVariant, heads int, cfg tcgen05.SSMChunkSchedule) egraph.Id {
	t := cfg.SeqLen()
	q := b.Input("q", F32, heads, t, cfg.HeadDimK)
	k := b.Input("k", F32, heads, t, cfg.HeadDimK)
	vv := b.Input("v", F32, heads, t, cfg.HeadDimV)
	a := b.Input("alpha", F32, heads, t)
	if v == tcgen05.SSMGatedDeltaNet {
		return b.GatedDeltaNet(cfg, q, k, vv, a, b.Input("beta", F32, heads, t))
	}
	return b.SSMChunkScan(cfg, q, k, vv, a)
}

func TestSSMTermEval(t *testing.T) {
	cfg := tcgen05.SSMChunkSchedule{HeadDimK: 32, HeadDimV: 32, ChunkLen: 32, NumChunks: 2}
	const heads = 2
	for _, v := range []tcgen05.SSMVariant{tcgen05.SSMMamba2SSD, tcgen05.SSMGatedDeltaNet} {
		r := rand.New(rand.NewPCG(1, uint64(v)))
		env := ssmEnv(r, heads, cfg)
		b := NewBuilder()
		root := buildSSM(b, v, heads, cfg)
		expr, _, err := egraph.ExtractBest(b.G, root, Cost{})
		if err != nil {
			t.Fatal(err)
		}
		got, err := Eval(expr, env)
		if err != nil {
			t.Fatalf("%v: Eval: %v", v, err)
		}
		// Head 1 against the tcgen05 reference directly.
		tl := cfg.SeqLen()
		rows := func(name string, d int) [][]float32 {
			m := make([][]float32, tl)
			for i := range m {
				m[i] = make([]float32, d)
				for j := range m[i] {
					m[i][j] = float32(env[name].Data[(tl+i)*d+j])
				}
			}
			return m
		}
		vec := func(name string) []float32 {
			x := make([]float32, tl)
			for i := range x {
				x[i] = float32(env[name].Data[tl+i])
			}
			return x
		}
		c := cfg
		c.Variant = v
		want, _, err := c.Reference(tcgen05.SSMInput{Q: rows("q", 32), K: rows("k", 32), V: rows("v", 32), Alpha: vec("alpha"), Beta: vec("beta")})
		if err != nil {
			t.Fatal(err)
		}
		for i := range want {
			for j := range want[i] {
				if g := got.Data[(tl+i)*32+j]; g != float64(want[i][j]) {
					t.Fatalf("%v: out[1][%d][%d] = %g, want %g", v, i, j, g, want[i][j])
				}
			}
		}
	}
}

// TestSSMGateRewrite checks that Optimize fuses RMSNormGate into each scan
// and that the fused term evaluates to the same values.
func TestSSMGateRewrite(t *testing.T) {
	cfg := tcgen05.SSMChunkSchedule{HeadDimK: 32, HeadDimV: 32, ChunkLen: 32, NumChunks: 2}
	const heads = 2
	for _, tt := range []struct {
		v     tcgen05.SSMVariant
		fused string
	}{
		{tcgen05.SSMMamba2SSD, OpSSMChunkScanGated},
		{tcgen05.SSMGatedDeltaNet, OpGatedDeltaNetGated},
	} {
		env := ssmEnv(rand.New(rand.NewPCG(3, uint64(tt.v))), heads, cfg)
		b := NewBuilder()
		scan := buildSSM(b, tt.v, heads, cfg)
		root := b.RMSNormGate(scan, b.Input("gate", F32, heads, cfg.SeqLen(), cfg.HeadDimV), 1e-5)
		plain, _, err := egraph.ExtractBest(b.G, root, Cost{})
		if err != nil {
			t.Fatal(err)
		}
		want, err := Eval(plain, env)
		if err != nil {
			t.Fatal(err)
		}
		opt, err := Optimize(b.G, root)
		if err != nil {
			t.Fatal(err)
		}
		if opt.Op != tt.fused {
			t.Fatalf("%v: optimized root %q, want %q", tt.v, opt.Op, tt.fused)
		}
		got, err := Eval(opt, env)
		if err != nil {
			t.Fatal(err)
		}
		for i := range want.Data {
			if got.Data[i] != want.Data[i] {
				t.Fatalf("%v: fused[%d] = %g, unfused %g", tt.v, i, got.Data[i], want.Data[i])
			}
		}
	}
}

func TestSSMTermRejectsConv(t *testing.T) {
	cfg := tcgen05.SSMChunkSchedule{HeadDimK: 32, HeadDimV: 32, ChunkLen: 32, NumChunks: 1, ConvKernel: 4}
	b := NewBuilder()
	root := buildSSM(b, tcgen05.SSMMamba2SSD, 1, cfg)
	expr, _, err := egraph.ExtractBest(b.G, root, Cost{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Eval(expr, ssmEnv(rand.New(rand.NewPCG(4, 4)), 1, cfg)); err == nil {
		t.Fatal("Eval with ConvKernel 4 succeeded, want error")
	}
}

func TestPlanHybrid(t *testing.T) {
	ssm := tcgen05.SSMChunkSchedule{Variant: tcgen05.SSMGatedDeltaNet, HeadDimK: 128, HeadDimV: 128, ChunkLen: 64, NumChunks: 64, ConvKernel: 4}
	var layers []HybridLayer
	for i := range 16 {
		if i%8 == 7 {
			layers = append(layers, HybridLayer{Kind: HybridAttention, Heads: 32, AttnTMEMCol: 256})
			continue
		}
		layers = append(layers, HybridLayer{Kind: HybridSSM, Heads: 32, SSM: ssm})
	}
	p, err := PlanHybrid("sm_100a", 148, layers)
	if err != nil {
		t.Fatal(err)
	}
	if p.SSMPerAttention != 7 || len(p.Stages) != 16 {
		t.Fatalf("plan: %v SSM per attention, %d stages", p.SSMPerAttention, len(p.Stages))
	}
	st := p.Stages[0]
	if st.TMEMColumns != 512 || st.HeadsPerCTA != 1 || st.StateBytes != 32*128*128*4 {
		t.Fatalf("stage 0 = %+v", st)
	}
	if got, want := p.RoundTripBytes(), int64(14)*2*32*128*128*4*63; got != want {
		t.Fatalf("RoundTripBytes = %d, want %d", got, want)
	}
	if !math.IsInf(mustPlan(t, []HybridLayer{{Kind: HybridSSM, Heads: 1, SSM: ssm}}).SSMPerAttention, 1) {
		t.Fatal("SSM-only plan ratio is finite")
	}

	for _, tt := range []struct {
		name   string
		target string
		ctas   int
		layers []HybridLayer
	}{
		{"sm_121a has no TMEM", "sm_121a", 1, layers},
		{"no ctas", "sm_100a", 0, layers},
		{"no layers", "sm_100a", 1, nil},
		{"no heads", "sm_100a", 1, []HybridLayer{{Kind: HybridSSM, SSM: ssm}}},
		{"bad ssm", "sm_100a", 1, []HybridLayer{{Kind: HybridSSM, Heads: 1, SSM: tcgen05.SSMChunkSchedule{HeadDimK: 128, HeadDimV: 128, ChunkLen: 128, NumChunks: 1}}}},
		{"bad attention cols", "sm_100a", 1, []HybridLayer{{Kind: HybridAttention, Heads: 1, AttnTMEMCol: 96}}},
		{"bad kind", "sm_100a", 1, []HybridLayer{{Heads: 1}}},
	} {
		if _, err := PlanHybrid(tt.target, tt.ctas, tt.layers); err == nil {
			t.Errorf("%s: PlanHybrid succeeded, want error", tt.name)
		}
	}
}

func mustPlan(t *testing.T, layers []HybridLayer) HybridPlan {
	t.Helper()
	p, err := PlanHybrid("sm_107a", 4, layers)
	if err != nil {
		t.Fatal(err)
	}
	return p
}
