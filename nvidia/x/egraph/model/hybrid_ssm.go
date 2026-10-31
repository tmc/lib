package model

import (
	"errors"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// HybridLayerKind identifies the operator type of a layer in a hybrid
// Mamba-2 SSM + Attention + SwiGLU MLP decoder stack (such as Jamba,
// Nemotron-H, or Zamba).
type HybridLayerKind string

const (
	// HybridLayerMamba2SSM represents a Mamba-2 selective state-space model
	// recurrence layer:
	//   h_t = A_t * h_{t-1} + B_t * x_t
	//   y_t = C_t * h_t + D * x_t
	// bracketed by in_proj and out_proj GEMMs and post-layer RMSNorm.
	HybridLayerMamba2SSM HybridLayerKind = "mamba2_ssm"

	// HybridLayerGQA represents a grouped-query attention projection layer
	// (qkv_proj -> attention/context projection -> o_proj -> post-layer RMSNorm).
	HybridLayerGQA HybridLayerKind = "gqa"

	// HybridLayerSwiGLUMLP represents a SwiGLU feed-forward network layer
	// (gate_up_proj -> SwiGLU -> down_proj -> post-layer RMSNorm).
	HybridLayerSwiGLUMLP HybridLayerKind = "swiglu_mlp"
)

// HybridStackConfig configures a multi-layer hybrid Mamba-2 SSM + GQA + SwiGLU
// MLP stack lowered into a single cross-layer [hopper.Megakernel].
type HybridStackConfig struct {
	Name        string
	Target      string
	BatchM      int
	HiddenDim   int
	SSMStateDim int
	QKVDim      int
	InterDim    int
	Eps         float32
	Layers      []HybridLayerKind
}

// Norm validates and populates default dimensions on c.
func (c HybridStackConfig) Norm() (HybridStackConfig, error) {
	if strings.TrimSpace(c.Name) == "" {
		c.Name = "hybrid_ssm_stack"
	}
	if strings.TrimSpace(c.Target) == "" {
		c.Target = "sm_121a"
	}
	if c.BatchM <= 0 {
		c.BatchM = 1
	}
	if c.HiddenDim <= 0 {
		c.HiddenDim = 256
	}
	if c.SSMStateDim <= 0 {
		c.SSMStateDim = c.HiddenDim
	}
	if c.QKVDim <= 0 {
		c.QKVDim = c.HiddenDim
	}
	if c.InterDim <= 0 {
		c.InterDim = 2 * c.HiddenDim
	}
	if c.Eps <= 0 {
		c.Eps = 1e-5
	}
	if len(c.Layers) == 0 {
		return c, errors.New("hybrid ssm stack: at least one layer is required")
	}
	for i, kind := range c.Layers {
		switch kind {
		case HybridLayerMamba2SSM, HybridLayerGQA, HybridLayerSwiGLUMLP:
		default:
			return c, fmt.Errorf("hybrid ssm stack: layer %d has unsupported kind %q", i, kind)
		}
	}
	return c, nil
}

// BuildHybridLayerPlan lowers a single [HybridLayerKind] at layerIdx into a
// [hopper.Plan] whose bridge input is "xn" ([BatchM, HiddenDim] BF16) and whose
// output has shape [BatchM, HiddenDim] BF16.
func BuildHybridLayerPlan(cfg HybridStackConfig, layerIdx int, kind HybridLayerKind) (*hopper.Plan, error) {
	normCfg, err := cfg.Norm()
	if err != nil {
		return nil, err
	}
	m := normCfg.BatchM
	h := normCfg.HiddenDim
	eps := normCfg.Eps
	prefix := fmt.Sprintf("l%d", layerIdx)

	b := hopper.NewBuilder()
	xn := b.Input("xn", hopper.BF16, m, h)
	xRes := b.Input(prefix+"_res_in", hopper.F32, m, h)

	var outNode egraph.Id
	var planName string

	switch kind {
	case HybridLayerMamba2SSM:
		dState := normCfg.SSMStateDim
		wIn := b.Input(prefix+"_ssm_in_proj", hopper.BF16, dState, h)
		xProj := b.MatMul(xn, wIn)

		// Selective state-space recurrence:
		//   h_t = A_t * h_{t-1} + B_t * x_t
		//   y_t = C_t * h_t + D * x_t
		aDecay := b.Input(prefix+"_ssm_a", hopper.F32, m, dState)
		hPrev := b.Input(prefix+"_ssm_h_prev", hopper.F32, m, dState)
		bGate := b.Input(prefix+"_ssm_b", hopper.F32, m, dState)
		cProj := b.Input(prefix+"_ssm_c", hopper.F32, m, dState)
		dSkip := b.Input(prefix+"_ssm_d", hopper.F32, m, dState)

		hNew := b.Map(hopper.MustLambda("(add (mul $0 $1) (mul $2 $3))"), aDecay, hPrev, bGate, xProj)
		yScan := b.Map(hopper.MustLambda("(add (mul $0 $1) (mul $2 $3))"), cProj, hNew, dSkip, xProj)
		yScanBF16 := b.Cast(hopper.BF16, yScan)

		wOut := b.Input(prefix+"_ssm_out_proj", hopper.BF16, h, dState)
		ssmOut := b.MatMul(yScanBF16, wOut)
		hid := b.Map(hopper.MustLambda("(add $0 $1)"), ssmOut, xRes)
		normW := b.Input(prefix+"_ssm_norm", hopper.F32, h)
		outNode = b.Cast(hopper.BF16, b.RMSNorm(hid, normW, h, eps))
		planName = prefix + "_mamba2_ssm"

	case HybridLayerGQA:
		qkvDim := normCfg.QKVDim
		wQKV := b.Input(prefix+"_gqa_qkv_proj", hopper.BF16, qkvDim, h)
		qkv := b.MatMul(xn, wQKV)
		attnGate := b.Input(prefix+"_gqa_ctx_scale", hopper.F32, m, qkvDim)
		attnCtx := b.Cast(hopper.BF16, b.Map(hopper.MustLambda("(mul $0 $1)"), qkv, attnGate))

		wO := b.Input(prefix+"_gqa_o_proj", hopper.BF16, h, qkvDim)
		attnOut := b.MatMul(attnCtx, wO)
		hid := b.Map(hopper.MustLambda("(add $0 $1)"), attnOut, xRes)
		normW := b.Input(prefix+"_gqa_norm", hopper.F32, h)
		outNode = b.Cast(hopper.BF16, b.RMSNorm(hid, normW, h, eps))
		planName = prefix + "_gqa"

	case HybridLayerSwiGLUMLP:
		inter := normCfg.InterDim
		wGU := b.Input(prefix+"_mlp_gate_up", hopper.BF16, 2*inter, h)
		act := b.SwiGLU(b.MatMul(xn, wGU))
		wDown := b.Input(prefix+"_mlp_down", hopper.BF16, h, inter)
		down := b.MatMul(act, wDown)
		hid := b.Map(hopper.MustLambda("(add $0 $1)"), down, xRes)
		normW := b.Input(prefix+"_mlp_norm", hopper.F32, h)
		outNode = b.Cast(hopper.BF16, b.RMSNorm(hid, normW, h, eps))
		planName = prefix + "_swiglu_mlp"

	default:
		return nil, fmt.Errorf("hybrid ssm layer %d: unsupported kind %q", layerIdx, kind)
	}

	expr, err := hopper.Optimize(b.G, outNode)
	if err != nil {
		return nil, fmt.Errorf("hybrid ssm optimize %s: %w", planName, err)
	}
	plan, err := hopper.CompileFor(expr, planName, hopper.BF16, normCfg.Target)
	if err != nil {
		return nil, fmt.Errorf("hybrid ssm compile %s: %w", planName, err)
	}
	normalizeMegakernelGEMMRows(plan, m)
	return plan, nil
}

// BuildHybridSSMMegakernelStack lowers each configured Mamba-2 SSM, GQA, and
// SwiGLU MLP layer into a [hopper.Plan], fuses all layers across their
// inter-layer bridge boundaries via [hopper.FuseCrossLayerPlans], and lowers
// the fused stack into a single persistent cross-layer [hopper.Megakernel].
func BuildHybridSSMMegakernelStack(cfg HybridStackConfig) (*hopper.Plan, *hopper.Megakernel, error) {
	normCfg, err := cfg.Norm()
	if err != nil {
		return nil, nil, err
	}

	var fused *hopper.Plan
	for i, kind := range normCfg.Layers {
		layerPlan, err := BuildHybridLayerPlan(normCfg, i, kind)
		if err != nil {
			return nil, nil, err
		}
		if fused == nil {
			fused = layerPlan
			continue
		}
		glue := hopper.CrossLayerGlue{
			Name:               fmt.Sprintf("%s_upto_l%d", normCfg.Name, i),
			LayerLPrefix:       fmt.Sprintf("s%d_", i-1),
			LayerNextPrefix:    fmt.Sprintf("l%d_", i),
			BridgeInput:        "xn",
			RetainBridgeOutput: true,
		}
		fused, err = hopper.FuseCrossLayerPlans(fused, layerPlan, glue)
		if err != nil {
			return nil, nil, fmt.Errorf("hybrid ssm stack fuse layer %d (%s): %w", i, kind, err)
		}
		normalizeMegakernelGEMMRows(fused, normCfg.BatchM)
	}

	fused.Name = normCfg.Name
	megaName := normCfg.Name + "_mega"
	mega, err := fused.Megakernel(megaName)
	if err != nil {
		return nil, nil, fmt.Errorf("hybrid ssm stack megakernel: %w", err)
	}
	return fused, &mega, nil
}

func normalizeMegakernelGEMMRows(pl *hopper.Plan, bucketM int) {
	if pl == nil {
		return
	}
	for i := range pl.Kernels {
		if g := pl.Kernels[i].GEMM; g != nil {
			for _, r := range []int{32, 16, 8, 4, 2, 1} {
				if g.GLU && r%2 != 0 {
					continue
				}
				if g.N%r == 0 && r*max(bucketM, g.MaxM, 1) <= 256 {
					g.RowsPerBlock = r
					break
				}
			}
		}
	}
}
