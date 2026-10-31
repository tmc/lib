package hopper

import (
	"math"
	"strings"
	"testing"
)

func TestDeltaNetFastWeightLinearAttnRecurrence(t *testing.T) {
	rules := []struct {
		name     string
		rule     FastWeightRuleKind
		useDecay bool
		alpha    float32
		beta     float32
	}{
		{name: "DeltaNet_NoDecay", rule: RuleDeltaNet, useDecay: false, alpha: 1.0, beta: 0.75},
		{name: "GatedDeltaNet", rule: RuleGatedDeltaNet, useDecay: true, alpha: 0.9, beta: 0.8},
		{name: "RWKV7_Goose", rule: RuleRWKV7, useDecay: true, alpha: 0.85, beta: 0.65},
		{name: "TTTLinear", rule: RuleTTTLinear, useDecay: true, alpha: 0.95, beta: 0.5},
	}

	const (
		batch    = 2
		numHeads = 2
		headDim  = 64
		steps    = 3
	)
	bh := batch * numHeads
	stepElems := bh * headDim
	stateElems := bh * headDim * headDim

	for _, tc := range rules {
		t.Run(tc.name, func(t *testing.T) {
			fw := FastWeightLinearAttn{
				Name:     "fw_" + string(tc.rule),
				Rule:     tc.rule,
				Batch:    batch,
				NumHeads: numHeads,
				HeadDim:  headDim,
				UseDecay: tc.useDecay,
				Arch:     ArchSM90A,
			}

			q := make([]float32, steps*stepElems)
			k := make([]float32, steps*stepElems)
			v := make([]float32, steps*stepElems)
			beta := make([]float32, steps*bh)
			alpha := make([]float32, steps*bh)
			stateIn := make([]float32, stateElems)

			for i := range stateIn {
				stateIn[i] = float32((i%13)-6) * 0.01
			}
			for s := range steps {
				for bhIdx := range bh {
					beta[s*bh+bhIdx] = tc.beta - 0.05*float32(s)
					alpha[s*bh+bhIdx] = tc.alpha - 0.02*float32(bhIdx)
					base := (s*bh + bhIdx) * headDim
					for d := range headDim {
						q[base+d] = float32(((s+d+bhIdx)%11)-5) * 0.1
						k[base+d] = float32(((2*s+d+bhIdx)%9)-4) * 0.08
						v[base+d] = float32(((3*s+d+2*bhIdx)%15)-7) * 0.12
					}
				}
			}

			out, stateOut, err := EvalFastWeightLinearAttnHost(fw, q, k, v, beta, alpha, stateIn)
			if err != nil {
				t.Fatalf("EvalFastWeightLinearAttnHost(%s): %v", tc.name, err)
			}
			if len(out) != steps*stepElems {
				t.Fatalf("len(out) = %d, want %d", len(out), steps*stepElems)
			}
			if len(stateOut) != stateElems {
				t.Fatalf("len(stateOut) = %d, want %d", len(stateOut), stateElems)
			}

			// Verify step-by-step recurrence matches single-call multi-step execution
			// and explicit Householder matrix equation:
			//   S_t = alpha_t * S_{t-1} * (I - beta_t * k_t * k_t^T) + beta_t * v_t * k_t^T
			//   o_t = S_t * q_t
			curState := make([]float32, len(stateIn))
			copy(curState, stateIn)
			for s := range steps {
				qStep := q[s*stepElems : (s+1)*stepElems]
				kStep := k[s*stepElems : (s+1)*stepElems]
				vStep := v[s*stepElems : (s+1)*stepElems]
				bStep := beta[s*bh : (s+1)*bh]
				aStep := alpha[s*bh : (s+1)*bh]

				// Compute reference S_t and o_t explicitly per (batch, head).
				wantOut := make([]float32, stepElems)
				wantState := make([]float32, stateElems)
				for bhIdx := range bh {
					bVal := float64(bStep[bhIdx])
					aVal := 1.0
					if tc.useDecay || tc.rule == RuleGatedDeltaNet || tc.rule == RuleRWKV7 {
						aVal = float64(aStep[bhIdx])
					}
					stOff := bhIdx * headDim * headDim
					vOff := bhIdx * headDim

					for i := range headDim {
						var sk float64
						for j := range headDim {
							sk += float64(curState[stOff+i*headDim+j]) * float64(kStep[vOff+j])
						}
						var oAcc float64
						for j := range headDim {
							sPrev := float64(curState[stOff+i*headDim+j])
							kj := float64(kStep[vOff+j])
							vi := float64(vStep[vOff+i])
							sNew := aVal*(sPrev-bVal*sk*kj) + bVal*vi*kj
							wantState[stOff+i*headDim+j] = float32(sNew)
							oAcc += sNew * float64(qStep[vOff+j])
						}
						wantOut[vOff+i] = float32(oAcc)
					}
				}

				stepOut, nextState, err := EvalFastWeightLinearAttnHost(fw, qStep, kStep, vStep, bStep, aStep, curState)
				if err != nil {
					t.Fatalf("step %d EvalFastWeightLinearAttnHost: %v", s, err)
				}
				for i := range stepElems {
					if diff := math.Abs(float64(stepOut[i] - wantOut[i])); diff > 1e-5 {
						t.Fatalf("step %d out[%d] = %g, want %g (diff %g)", s, i, stepOut[i], wantOut[i], diff)
					}
					if diff := math.Abs(float64(out[s*stepElems+i] - wantOut[i])); diff > 1e-5 {
						t.Fatalf("multi-step out[%d] = %g, want %g (diff %g)", s*stepElems+i, out[s*stepElems+i], wantOut[i], diff)
					}
				}
				curState = nextState
			}

			for i := range stateElems {
				if diff := math.Abs(float64(stateOut[i] - curState[i])); diff > 1e-5 {
					t.Fatalf("final stateOut[%d] = %g, want %g (diff %g)", i, stateOut[i], curState[i], diff)
				}
			}

			cudaSrc := fw.CUDA()
			for _, token := range []string{
				fw.Name,
				string(tc.rule),
				"__shared__ float s_q[D]",
				"__shared__ float s_k[D]",
				"__shared__ float s_v[D]",
				"float s_row[D]",
				"alpha_t * (s_prev - beta_t * proj_k * k_j) + beta_t * v_i * k_j",
			} {
				if !strings.Contains(cudaSrc, token) {
					t.Fatalf("CUDA() for %s missing %q:\n%s", tc.name, token, cudaSrc)
				}
			}
		})
	}
}

func TestDeltaNetExactHouseholderAssociativeRecall(t *testing.T) {
	// Verify the exact Householder erasure-and-write property of DeltaNet:
	// when alpha = 1, beta = 1, and ||k||_2 = 1, writing (k, v_new) into S_{t-1}
	// erases the previous association (S_{t-1} k) along k and replaces it with
	// v_new so that querying with q = k yields o_t = S_t k == v_new exactly.
	fw := FastWeightLinearAttn{
		Name:     "deltanet_householder",
		Rule:     RuleDeltaNet,
		Batch:    1,
		NumHeads: 1,
		HeadDim:  64,
		UseDecay: false,
		Arch:     ArchSM100A,
	}

	kUnit := make([]float32, 64)
	vOld := make([]float32, 64)
	vNew := make([]float32, 64)
	var normSq float64
	for i := range 64 {
		val := float32((i%7)-3) + 0.5
		kUnit[i] = val
		normSq += float64(val * val)
		vOld[i] = float32(i) * 0.25
		vNew[i] = -float32(i)*0.5 + 1.25
	}
	invNorm := float32(1.0 / math.Sqrt(normSq))
	for i := range 64 {
		kUnit[i] *= invNorm
	}

	beta := []float32{1.0}
	alpha := []float32{1.0}

	// Step 1: Write (kUnit, vOld) into zero state and query with q = kUnit.
	out1, state1, err := EvalFastWeightLinearAttnHost(fw, kUnit, kUnit, vOld, beta, alpha, nil)
	if err != nil {
		t.Fatalf("step 1: %v", err)
	}
	for i := range 64 {
		if diff := math.Abs(float64(out1[i] - vOld[i])); diff > 1e-5 {
			t.Fatalf("step 1 recall out1[%d] = %g, want vOld[%d] = %g (diff %g)", i, out1[i], i, vOld[i], diff)
		}
	}

	// Step 2: Overwrite the same unit key kUnit with vNew (beta=1 erases vOld along kUnit).
	out2, _, err := EvalFastWeightLinearAttnHost(fw, kUnit, kUnit, vNew, beta, alpha, state1)
	if err != nil {
		t.Fatalf("step 2: %v", err)
	}
	for i := range 64 {
		if diff := math.Abs(float64(out2[i] - vNew[i])); diff > 1e-5 {
			t.Fatalf("step 2 Householder overwrite out2[%d] = %g, want vNew[%d] = %g (diff %g)", i, out2[i], i, vNew[i], diff)
		}
	}
}

func TestDeltaNetHybridAttentionMegakernel(t *testing.T) {
	// Model a 4-layer 3:1 Gated-DeltaNet : GQA hybrid stack (Qwen3-Next / Kimi-Linear ratio).
	fw := FastWeightLinearAttn{
		Name:     "qwen3_next_gated_deltanet",
		Rule:     RuleGatedDeltaNet,
		Batch:    2,
		NumHeads: 4,
		HeadDim:  64,
		UseDecay: true,
		Arch:     ArchSM100A,
	}

	plan, mega, err := BuildHybridDeltaNetAttentionMegakernel(
		"qwen3_next_hybrid_4l",
		4,   // 4 layers total
		4,   // every 4th layer (layer 3) is GQA -> 3:1 Gated-DeltaNet : GQA
		fw,
		256, // hiddenDim
		512, // intermDim
	)
	if err != nil {
		t.Fatalf("BuildHybridDeltaNetAttentionMegakernel: %v", err)
	}
	if plan == nil || mega == nil {
		t.Fatalf("expected non-nil plan and megakernel")
	}
	if len(mega.Stages) < 16 {
		t.Fatalf("len(mega.Stages) = %d, want >= 16 stages across 4 hybrid layers", len(mega.Stages))
	}

	// Verify 3 layers have Gated-DeltaNet inputs and 1 layer (l3) has GQA inputs.
	var dnLayers, gqaLayers int
	for _, in := range plan.Inputs {
		if strings.HasSuffix(in.Name, "_dn_in_w") {
			dnLayers++
		}
		if strings.HasSuffix(in.Name, "_gqa_qkv_w") {
			gqaLayers++
		}
	}
	if dnLayers != 3 || gqaLayers != 1 {
		t.Fatalf("hybrid layer ratio: got %d DeltaNet layers and %d GQA layers, want 3:1", dnLayers, gqaLayers)
	}

	src, err := mega.Source()
	if err != nil {
		t.Fatalf("mega.Source(): %v", err)
	}
	for _, want := range []string{
		"qwen3_next_hybrid_4l_mega",
		"eg_ready_wait",
		"eg_ready_signal",
	} {
		if !strings.Contains(src, want) {
			t.Fatalf("mega.Source() missing %q", want)
		}
	}
}
