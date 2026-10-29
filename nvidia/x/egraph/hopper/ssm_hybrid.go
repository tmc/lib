package hopper

import (
	"fmt"
	"math"
	"slices"

	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/tcgen05"
)

// Linear-recurrence term operators for hybrid SSM/attention stacks. The
// scans run the [tcgen05.SSMChunkSchedule] recurrence per head; the gated
// forms fuse a following RMSNormGate into the scan's epilogue.
const (
	OpSSMChunkScan       = "ssm_chunk_scan"
	OpGatedDeltaNet      = "gated_delta_net"
	OpSSMChunkScanGated  = "ssm_chunk_scan_gated"
	OpGatedDeltaNetGated = "gated_delta_net_gated"
)

func (b *Builder) ssmConfig(cfg tcgen05.SSMChunkSchedule, v tcgen05.SSMVariant) egraph.Id {
	cfg.Name = ""
	cfg.Variant = v
	return b.G.Add(egraph.ENode{
		Op:   fmt.Sprintf("ssm:%v/k%d/v%d/l%d/c%d/w%d", v, cfg.HeadDimK, cfg.HeadDimV, cfg.ChunkLen, cfg.NumChunks, cfg.ConvKernel),
		Data: cfg,
	})
}

// SSMChunkScan adds a Mamba-2 SSD scan term over q, k [heads, T, d_k],
// v [heads, T, d_v] and decays alpha [heads, T], with T = cfg.SeqLen(). The
// result is [heads, T, d_v]. cfg.ConvKernel must be 0: the term takes
// already-convolved inputs.
func (b *Builder) SSMChunkScan(cfg tcgen05.SSMChunkSchedule, q, k, v, alpha egraph.Id) egraph.Id {
	return b.node(OpSSMChunkScan, b.ssmConfig(cfg, tcgen05.SSMMamba2SSD), q, k, v, alpha)
}

// GatedDeltaNet adds a gated delta-rule scan term; beta is [heads, T].
// Shapes and the cfg.ConvKernel rule are as for [Builder.SSMChunkScan].
func (b *Builder) GatedDeltaNet(cfg tcgen05.SSMChunkSchedule, q, k, v, alpha, beta egraph.Id) egraph.Id {
	return b.node(OpGatedDeltaNet, b.ssmConfig(cfg, tcgen05.SSMGatedDeltaNet), q, k, v, alpha, beta)
}

// ssmRules fuse rms_norm(scan) * sigmoid(gate) into the scan.
func ssmRules() []*egraph.Rewrite {
	v := egraph.Var
	return []*egraph.Rewrite{
		{
			Name: "ssm-scan-gate",
			LHS:  egraph.Pat(OpRMSNormGate, egraph.Pat(OpSSMChunkScan, v("cfg"), v("q"), v("k"), v("v"), v("a")), v("gate"), v("eps")),
			RHS:  egraph.Pat(OpSSMChunkScanGated, v("cfg"), v("q"), v("k"), v("v"), v("a"), v("gate"), v("eps")),
		},
		{
			Name: "gated-deltanet-gate",
			LHS:  egraph.Pat(OpRMSNormGate, egraph.Pat(OpGatedDeltaNet, v("cfg"), v("q"), v("k"), v("v"), v("a"), v("b")), v("gate"), v("eps")),
			RHS:  egraph.Pat(OpGatedDeltaNetGated, v("cfg"), v("q"), v("k"), v("v"), v("a"), v("b"), v("gate"), v("eps")),
		},
	}
}

func isSSMOp(op string) bool {
	switch op {
	case OpSSMChunkScan, OpGatedDeltaNet, OpSSMChunkScanGated, OpGatedDeltaNetGated:
		return true
	}
	return false
}

// evalSSM evaluates the scan operators with [tcgen05.SSMChunkSchedule.Reference].
func evalSSM(e *egraph.Expr, env map[string]Value, memo map[*egraph.Expr]Value) (Value, error) {
	cfg, ok := e.Children[0].Data.(tcgen05.SSMChunkSchedule)
	if !ok {
		return Value{}, fmt.Errorf("eval %s: missing SSMChunkSchedule config", e.Op)
	}
	if cfg.ConvKernel != 0 {
		return Value{}, fmt.Errorf("eval %s: conv width %d: the term takes convolved inputs", e.Op, cfg.ConvKernel)
	}
	nin := 4
	if cfg.Variant == tcgen05.SSMGatedDeltaNet {
		nin = 5
	}
	in := make([]Value, nin)
	for i := range in {
		x, err := eval(e.Children[1+i], env, memo)
		if err != nil {
			return Value{}, err
		}
		in[i] = x
	}
	t, dk, dv := cfg.SeqLen(), cfg.HeadDimK, cfg.HeadDimV
	heads := 0
	if len(in[0].Shape) == 3 {
		heads = in[0].Shape[0]
	}
	want := [][]int{{heads, t, dk}, {heads, t, dk}, {heads, t, dv}, {heads, t}, {heads, t}}
	for i, x := range in {
		if !slices.Equal(x.Shape, want[i]) {
			return Value{}, fmt.Errorf("eval %s: operand %d shape %v, want %v", e.Op, i, x.Shape, want[i])
		}
	}
	rows := func(x Value, h, d int) [][]float32 {
		m := make([][]float32, t)
		for i := range m {
			m[i] = make([]float32, d)
			for j := range m[i] {
				m[i][j] = float32(x.Data[(h*t+i)*d+j])
			}
		}
		return m
	}
	vec := func(x Value, h int) []float32 {
		r := make([]float32, t)
		for i := range r {
			r[i] = float32(x.Data[h*t+i])
		}
		return r
	}
	out := Value{Shape: []int{heads, t, dv}, Data: make([]float64, heads*t*dv)}
	for h := range heads {
		si := tcgen05.SSMInput{Q: rows(in[0], h, dk), K: rows(in[1], h, dk), V: rows(in[2], h, dv), Alpha: vec(in[3], h)}
		if nin == 5 {
			si.Beta = vec(in[4], h)
		}
		o, _, err := cfg.Reference(si)
		if err != nil {
			return Value{}, fmt.Errorf("eval %s: %w", e.Op, err)
		}
		for i, r := range o {
			for j, f := range r {
				out.Data[(h*t+i)*dv+j] = float64(f)
			}
		}
	}
	if e.Op == OpSSMChunkScan || e.Op == OpGatedDeltaNet {
		return out, nil
	}
	gate, err := eval(e.Children[1+nin], env, memo)
	if err != nil {
		return Value{}, err
	}
	if !slices.Equal(gate.Shape, out.Shape) {
		return Value{}, fmt.Errorf("eval %s: gate shape %v, want %v", e.Op, gate.Shape, out.Shape)
	}
	eps, _ := e.Children[2+nin].Data.(float32)
	return rmsNormGateValue(out, gate, eps), nil
}

// rmsNormGateValue computes rms_norm(x) * sigmoid(bf16(gate)) over the last
// axis, as [Builder.RMSNormGate] evaluates.
func rmsNormGateValue(x, g Value, eps float32) Value {
	if eps == 0 {
		eps = 1e-5
	}
	d := x.Shape[len(x.Shape)-1]
	out := Value{Shape: slices.Clone(x.Shape), Data: make([]float64, len(x.Data))}
	for r := 0; r < len(x.Data)/d; r++ {
		row := x.Data[r*d : (r+1)*d]
		var ss float64
		for _, v := range row {
			ss += v * v
		}
		inv := 1 / math.Sqrt(ss/float64(d)+float64(eps))
		for c, v := range row {
			gf := float64(BF16.Round(float32(g.Data[r*d+c])))
			out.Data[r*d+c] = v * inv / (1 + math.Exp(-gf))
		}
	}
	return out
}

// A HybridLayerKind is the mixer of one layer in a hybrid stack.
type HybridLayerKind uint8

const (
	HybridSSM       HybridLayerKind = iota + 1 // linear recurrence ([tcgen05.SSMChunkSchedule])
	HybridAttention                            // softmax attention (GQA or MLA)
)

func (k HybridLayerKind) String() string {
	switch k {
	case HybridSSM:
		return "ssm"
	case HybridAttention:
		return "attention"
	}
	return fmt.Sprintf("HybridLayerKind(%d)", uint8(k))
}

// A HybridLayer is one mixer layer of a hybrid stack.
type HybridLayer struct {
	Kind        HybridLayerKind
	Heads       int
	SSM         tcgen05.SSMChunkSchedule // HybridSSM layers
	AttnTMEMCol uint32                   // TMEM columns an attention stage allocates (HybridAttention layers)
}

// A HybridStage is the slice of a persistent hybrid megakernel that runs
// one layer. Stage i waits on grid barrier epoch i before it starts.
type HybridStage struct {
	Layer       int
	Kind        HybridLayerKind
	TMEMColumns uint32 // per-CTA allocation, freed before the next stage
	HeadsPerCTA int    // heads each CTA iterates over
	StateBytes  int64  // FP32 recurrent state held in TMEM across the layer's chunks
	// RoundTripBytes is the global-memory traffic a kernel that stores and
	// reloads S at every chunk boundary would add: 2 * StateBytes * (chunks-1).
	RoundTripBytes int64
}

// A HybridPlan assigns the layers of a hybrid stack to the stages of one
// persistent cooperative launch. It is a static plan: the stages' kernels
// are emitted separately ([tcgen05.SSMChunkSchedule.CUDA] for SSM layers).
type HybridPlan struct {
	Target          string
	CTAs            int
	Stages          []HybridStage
	SSMPerAttention float64 // SSM layers per attention layer (+Inf without attention)
}

// PlanHybrid plans layers on ctas resident CTAs for target.
func PlanHybrid(target string, ctas int, layers []HybridLayer) (HybridPlan, error) {
	switch target {
	case "sm_100a", "sm_103a", "sm_107a", "sm_100f", "sm_103f", "sm_107f":
	default:
		return HybridPlan{}, fmt.Errorf("plan hybrid: target %q has no tcgen05 TMEM", target)
	}
	if ctas < 1 {
		return HybridPlan{}, fmt.Errorf("plan hybrid: %d CTAs", ctas)
	}
	if len(layers) == 0 {
		return HybridPlan{}, fmt.Errorf("plan hybrid: no layers")
	}
	p := HybridPlan{Target: target, CTAs: ctas}
	var nssm, nattn int
	for i, l := range layers {
		if l.Heads < 1 {
			return HybridPlan{}, fmt.Errorf("plan hybrid: layer %d has %d heads", i, l.Heads)
		}
		st := HybridStage{Layer: i, Kind: l.Kind, HeadsPerCTA: (l.Heads + ctas - 1) / ctas}
		switch l.Kind {
		case HybridSSM:
			s := l.SSM
			s.Target = target
			if err := s.Validate(); err != nil {
				return HybridPlan{}, fmt.Errorf("plan hybrid: layer %d: %w", i, err)
			}
			st.TMEMColumns = s.TMEMColumns()
			st.StateBytes = int64(l.Heads) * int64(s.HeadDimK*s.HeadDimV) * 4
			st.RoundTripBytes = 2 * st.StateBytes * int64(s.NumChunks-1)
			nssm++
		case HybridAttention:
			c := l.AttnTMEMCol
			if c < 32 || c > 512 || c&(c-1) != 0 {
				return HybridPlan{}, fmt.Errorf("plan hybrid: layer %d: attention TMEM columns %d not a power of two in [32, 512]", i, c)
			}
			st.TMEMColumns = c
			nattn++
		default:
			return HybridPlan{}, fmt.Errorf("plan hybrid: layer %d: unknown kind %v", i, l.Kind)
		}
		p.Stages = append(p.Stages, st)
	}
	p.SSMPerAttention = math.Inf(1)
	if nattn > 0 {
		p.SSMPerAttention = float64(nssm) / float64(nattn)
	}
	return p, nil
}

// RoundTripBytes sums the stages' avoided state round trips.
func (p HybridPlan) RoundTripBytes() int64 {
	var n int64
	for _, s := range p.Stages {
		n += s.RoundTripBytes
	}
	return n
}
