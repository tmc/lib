package hopper

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
)

func buildTestCrossLayerPlans(t *testing.T, m, h, qDim, kvDim, inter int, qkvBias, fuseTails bool) (post, nextAttnIn *Plan) {
	t.Helper()
	const eps = 1e-5

	// Layer L post: o_proj -> residual_rmsnorm -> gate_up -> down -> post_residual_rmsnorm -> xn
	bPost := NewBuilder()
	attn := bPost.Input("attn", BF16, m, qDim)
	x := bPost.Input("x", F32, m, h)
	wo := bPost.Input("w_o", BF16, h, qDim)
	hid := bPost.Map(MustLambda("(add $0 $1)"), bPost.MatMul(attn, wo), x)
	ffnNorm := bPost.Cast(BF16, bPost.RMSNorm(hid, bPost.Input("ffn_norm", F32, h), h, eps))
	act := bPost.SwiGLU(bPost.MatMul(ffnNorm, bPost.Input("w_gu", BF16, 2*inter, h)))
	down := bPost.Map(MustLambda("(add $0 $1)"), bPost.MatMul(act, bPost.Input("w_down", BF16, h, inter)), hid)
	xnOut := bPost.Cast(BF16, bPost.RMSNorm(down, bPost.Input("next_norm", F32, h), h, eps))

	postExpr, err := Optimize(bPost.G, xnOut)
	if err != nil {
		t.Fatalf("optimize post: %v", err)
	}
	post, err = Compile(postExpr, "l0_post", BF16)
	if err != nil {
		t.Fatalf("compile post: %v", err)
	}
	if len(post.Kernels) != 5 {
		t.Fatalf("post has %d kernels before FuseTails, want 5", len(post.Kernels))
	}
	// Give descriptive stage names matching the canonical 5-stage post pipeline.
	stageNames := []string{
		"l0_o_proj",
		"l0_residual_rmsnorm",
		"l0_gate_up",
		"l0_down",
		"l0_post_residual_rmsnorm",
	}
	for i, name := range stageNames {
		setPlanKernelName(&post.Kernels[i], name)
		if post.Kernels[i].GEMM != nil {
			post.Kernels[i].GEMM.RowsPerBlock = 32
		}
	}
	if fuseTails {
		if fused := post.FuseTails(); fused != 2 {
			t.Fatalf("post.FuseTails() = %d, want 2", fused)
		}
	}

	// Layer L+1 attnIn: qkv_proj over xn
	bAttn := NewBuilder()
	xnIn := bAttn.Input("xn", BF16, m, h)
	wqkv := bAttn.Input("w_qkv", BF16, qDim+2*kvDim, h)
	qkv := bAttn.MatMul(xnIn, wqkv)
	if qkvBias {
		qkv = bAttn.Map(MustLambda("(add $0 $1)"), qkv, bAttn.Input("b_qkv", F32, qDim+2*kvDim))
	}
	attnExpr, err := Optimize(bAttn.G, qkv)
	if err != nil {
		t.Fatalf("optimize nextAttnIn: %v", err)
	}
	nextAttnIn, err = Compile(attnExpr, "l1_attn_in", F32)
	if err != nil {
		t.Fatalf("compile nextAttnIn: %v", err)
	}
	if len(nextAttnIn.Kernels) != 1 {
		t.Fatalf("nextAttnIn has %d kernels, want 1", len(nextAttnIn.Kernels))
	}
	setPlanKernelName(&nextAttnIn.Kernels[0], "l1_qkv_proj")
	if nextAttnIn.Kernels[0].GEMM != nil {
		nextAttnIn.Kernels[0].GEMM.RowsPerBlock = 32
	}
	return post, nextAttnIn
}

func runSequentialCrossLayer(t *testing.T, post, nextAttnIn *Plan, env map[string]Value, bridgeInput string) []float64 {
	t.Helper()
	if bridgeInput == "" {
		bridgeInput = "xn"
	}
	postOut := runPlan(t, post, env)
	nextEnv := make(map[string]Value, len(env)+1)
	for k, v := range env {
		nextEnv[k] = v
	}
	nextEnv[bridgeInput] = Value{
		Shape: slices.Clone(post.Output.Shape),
		Data:  postOut,
	}
	return runPlan(t, nextAttnIn, nextEnv)
}

func TestCrossLayerMegakernelEmulated(t *testing.T) {
	const (
		tokens = 1
		h      = 256
		qDim   = 256
		kvDim  = 64
		inter  = 256
	)
	post, nextAttnIn := buildTestCrossLayerPlans(t, tokens, h, qDim, kvDim, inter, true, false)

	glue := CrossLayerGlue{
		Name:               "cross_l0_l1_mega",
		LayerLPrefix:       "l0_",
		LayerNextPrefix:    "l1_",
		BridgeInput:        "xn",
		RetainBridgeOutput: true,
	}
	fusedPlan, mega, err := BuildCrossLayerMegakernel(post, nextAttnIn, glue)
	if err != nil {
		t.Fatalf("BuildCrossLayerMegakernel: %v", err)
	}

	wantStages := []string{
		"l0_o_proj",
		"l0_residual_rmsnorm",
		"l0_gate_up",
		"l0_down",
		"l0_post_residual_rmsnorm",
		"l1_qkv_proj",
	}
	if len(mega.Stages) != len(wantStages) {
		t.Fatalf("len(mega.Stages) = %d, want %d", len(mega.Stages), len(wantStages))
	}
	for i, want := range wantStages {
		if got := mega.Stages[i].Name; got != want {
			t.Fatalf("stage[%d].Name = %q, want %q", i, got, want)
		}
	}

	// Verify the bridge tensor (post.Output) is retained in Temps and wired into stage 5's wait slot.
	hasBridgeTemp := false
	for _, tmp := range fusedPlan.Temps {
		if tmp.Name == post.Output.Name {
			hasBridgeTemp = true
			break
		}
	}
	if !hasBridgeTemp {
		t.Fatalf("bridge tensor %q not found in fusedPlan.Temps %v", post.Output.Name, fusedPlan.Temps)
	}
	if !slices.Contains(mega.Stages[5].WaitSlots, 4) {
		t.Fatalf("stage 5 (%s) WaitSlots = %v, want to contain stage 4", mega.Stages[5].Name, mega.Stages[5].WaitSlots)
	}

	// Verify StageAttributionMap() covers all 6 stages across both layers with valid #line directives.
	attrMap := mega.StageAttributionMap()
	if len(attrMap) != len(wantStages) {
		t.Fatalf("len(StageAttributionMap) = %d, want %d", len(attrMap), len(wantStages))
	}
	lineSrc, spans, err := mega.LineInfoSource()
	if err != nil {
		t.Fatalf("LineInfoSource: %v", err)
	}
	lines := strings.Split(lineSrc, "\n")
	for i, name := range wantStages {
		sp, ok := attrMap[name]
		if !ok {
			t.Fatalf("StageAttributionMap missing stage %q", name)
		}
		if sp != spans[i] {
			t.Fatalf("StageAttributionMap[%q] = %+v, want %+v", name, sp, spans[i])
		}
		wantDirective := fmt.Sprintf("#line 1 %q", name+".cu")
		if got := lines[sp.StartLine-2]; got != wantDirective {
			t.Errorf("stage %s line before StartLine = %q, want %q", name, got, wantDirective)
		}
		wantFn := "EG_DEV void " + name + "_dev("
		if got := lines[sp.StartLine-1]; !strings.HasPrefix(got, wantFn) {
			t.Errorf("stage %s StartLine = %q, want prefix %q", name, got, wantFn)
		}
		if got := lines[sp.EndLine-1]; got != "}" {
			t.Errorf("stage %s EndLine = %q, want %q", name, got, "}")
		}
	}

	// Execute the fused cross-layer Megakernel via cudahostemu across 2 epochs
	// and verify numerical agreement against running post and nextAttnIn sequentially.
	r := rand.New(rand.NewPCG(81, 82))
	envs := make([]map[string]Value, 2)
	for i := range envs {
		envs[i] = randEnv(r, fusedPlan.Inputs...)
	}

	outs, readyWords, err := runMegakernelSourceEpochs(t, mega, lineSrc, envs)
	if err != nil {
		t.Fatalf("runMegakernelSourceEpochs: %v", err)
	}
	for ep, env := range envs {
		seqOut := runSequentialCrossLayer(t, post, nextAttnIn, env, "xn")
		agree(t, fmt.Sprintf("epoch_%d_sequential", ep+1), F32, outs[ep], seqOut)
		fusedPlanOut := runPlan(t, fusedPlan, env)
		agree(t, fmt.Sprintf("epoch_%d_fused_plan", ep+1), F32, outs[ep], fusedPlanOut)
	}
	for i := 0; i < len(mega.Stages)-1; i++ {
		wantWord := PackReadyWord(uint32(len(envs)), uint32(mega.Stages[i].Blocks))
		if readyWords[i] != wantWord {
			t.Fatalf("ready[%d] = %#x, want %#x", i, readyWords[i], wantWord)
		}
	}
}

func TestCrossLayerFuseTailsAndCollisions(t *testing.T) {
	const (
		tokens = 1
		h      = 256
		qDim   = 256
		kvDim  = 64
		inter  = 256
	)
	// Build with FuseTails enabled on post (as llama-infer's compilePlan does).
	post, nextAttnIn := buildTestCrossLayerPlans(t, tokens, h, qDim, kvDim, inter, false, true)
	if len(post.Kernels) != 3 {
		t.Fatalf("tail-fused post has %d kernels, want 3", len(post.Kernels))
	}

	// Force colliding kernel names on both plans to test prefix disambiguation.
	post.Kernels[0].GEMM.Name = "proj"
	nextAttnIn.Kernels[0].GEMM.Name = "proj"

	fusedPlan, mega, err := BuildCrossLayerMegakernel(post, nextAttnIn, CrossLayerGlue{
		Name:            "tail_unfused_cross",
		LayerLPrefix:    "l0_",
		LayerNextPrefix: "l1_",
	})
	if err != nil {
		t.Fatalf("BuildCrossLayerMegakernel with tail-fused post: %v", err)
	}
	// Unfusing the 2 tails in post restores 5 stages + 1 nextAttnIn stage = 6 stages.
	if len(mega.Stages) != 6 {
		t.Fatalf("len(mega.Stages) = %d, want 6 after tail unfusing", len(mega.Stages))
	}
	if mega.Stages[0].Name != "l0_proj" {
		t.Fatalf("stage 0 name = %q, want l0_proj", mega.Stages[0].Name)
	}
	if mega.Stages[5].Name != "l1_proj" {
		t.Fatalf("stage 5 name = %q, want l1_proj", mega.Stages[5].Name)
	}

	r := rand.New(rand.NewPCG(91, 92))
	env := randEnv(r, fusedPlan.Inputs...)
	outs, _, err := runMegakernelEpochs(t, mega, []map[string]Value{env})
	if err != nil {
		t.Fatalf("runMegakernelEpochs: %v", err)
	}
	seqOut := runSequentialCrossLayer(t, post, nextAttnIn, env, "xn")
	agree(t, "tail_unfused_sequential", F32, outs[0], seqOut)

	// Verify colliding input names and Cluster / WarpSpecialized flags are normalized.
	rk0 := RowKernel{
		Name:           "norm_stage",
		D:              256,
		Cluster:        true,
		ClusterMaxRows: 64,
		Inputs:         []RowTensor{{Name: "in1", Buffer: "x", DType: BF16, Kind: RowData}, {Name: "in2", Buffer: "w", DType: BF16, Kind: ColVector}},
		Values:         []RowValue{{Name: "v1", Body: MustLambda("(mul $0 $1)"), Args: []string{"in1", "in2"}}},
		Outputs:        []RowTensor{{Name: "out", DType: BF16, Kind: RowData, Value: "v1"}},
	}
	p0 := &Plan{
		Name:    "p",
		Inputs:  []Tensor{{"x", BF16, []int{64, 256}}, {"w", BF16, []int{256}}},
		Output:  Tensor{"out", BF16, []int{64, 256}},
		Kernels: []PlanKernel{{Row: &rk0, Rows: 64, Args: []string{"x", "w", "out"}}},
	}
	tg1 := TileGEMM{
		Name:            "norm_stage",
		N:               128,
		K:               256,
		BM:              64,
		BN:              128,
		Stages:          2,
		Weight:          WeightBF16,
		Out:             F32,
		WGMMA:           false,
		Swizzle128B:     true,
		WarpSpecialized: true,
	}
	p1 := &Plan{
		Name:    "p",
		Inputs:  []Tensor{{"xn", BF16, []int{64, 256}}, {"w", BF16, []int{128, 256}}},
		Output:  Tensor{"out", F32, []int{64, 128}},
		Kernels: []PlanKernel{{Tile: &tg1, M: 64, Args: []string{"xn", "w", "out"}}},
	}
	fusedCollide, megaCollide, err := BuildCrossLayerMegakernel(p0, p1, CrossLayerGlue{})
	if err != nil {
		t.Fatalf("BuildCrossLayerMegakernel colliding: %v", err)
	}
	if fusedCollide.Kernels[0].Row.Cluster {
		t.Fatal("expected Row.Cluster to be disabled")
	}
	if fusedCollide.Kernels[1].Tile.WarpSpecialized {
		t.Fatal("expected Tile.WarpSpecialized to be disabled")
	}
	var inNames []string
	for _, in := range fusedCollide.Inputs {
		inNames = append(inNames, in.Name)
	}
	if !slices.Contains(inNames, "l0_w") || !slices.Contains(inNames, "l1_w") {
		t.Fatalf("colliding input 'w' was not disambiguated in %v", inNames)
	}
	if len(megaCollide.StageAttributionMap()) != 2 {
		t.Fatalf("StageAttributionMap len = %d, want 2", len(megaCollide.StageAttributionMap()))
	}
}

func TestCrossLayerErrors(t *testing.T) {
	validRow := RowKernel{
		Name:    "rk",
		D:       64,
		Inputs:  []RowTensor{{Name: "in1", Buffer: "x", DType: BF16, Kind: RowData}},
		Values:  []RowValue{{Name: "v1", Body: Identity, Args: []string{"in1"}}},
		Outputs: []RowTensor{{Name: "xn", DType: BF16, Kind: RowData, Value: "v1"}},
	}
	p0 := &Plan{
		Name:    "p0",
		Inputs:  []Tensor{{"x", BF16, []int{1, 64}}},
		Output:  Tensor{"xn", BF16, []int{1, 64}},
		Kernels: []PlanKernel{{Row: &validRow, Rows: 1, Args: []string{"x", "xn"}}},
	}
	p1 := &Plan{
		Name:    "p1",
		Inputs:  []Tensor{{"xn", BF16, []int{1, 64}}},
		Output:  Tensor{"y", BF16, []int{1, 64}},
		Kernels: []PlanKernel{{Row: &validRow, Rows: 1, Args: []string{"xn", "y"}}},
	}

	if _, err := FuseCrossLayerPlans(nil, p1, CrossLayerGlue{}); err == nil {
		t.Error("nil post: expected error")
	}
	if _, err := FuseCrossLayerPlans(p0, nil, CrossLayerGlue{}); err == nil {
		t.Error("nil nextAttnIn: expected error")
	}
	if _, err := FuseCrossLayerPlans(p0, p1, CrossLayerGlue{BridgeInput: "missing"}); err == nil {
		t.Error("missing bridge input: expected error")
	}
	if _, err := FuseCrossLayerPlans(p0, p1, CrossLayerGlue{LayerLPrefix: "same_", LayerNextPrefix: "same_"}); err == nil {
		t.Error("identical prefixes: expected error")
	}
	if _, err := FuseCrossLayerPlans(p0, p1, CrossLayerGlue{Name: "eg_reserved"}); err == nil {
		t.Error("reserved name: expected error")
	}

	p1BadDType := *p1
	p1BadDType.Inputs = []Tensor{{"xn", F32, []int{1, 64}}}
	if _, err := FuseCrossLayerPlans(p0, &p1BadDType, CrossLayerGlue{}); err == nil {
		t.Error("mismatched bridge dtype: expected error")
	}

	p1BadShape := *p1
	p1BadShape.Inputs = []Tensor{{"xn", BF16, []int{2, 64}}}
	if _, err := FuseCrossLayerPlans(p0, &p1BadShape, CrossLayerGlue{}); err == nil {
		t.Error("mismatched bridge shape: expected error")
	}
}

func buildTestStackPlans(t *testing.T, numLayers, m, h, qDim, kvDim, inter, vocab int) (norm *Plan, layers []DecoderLayerPlans, head *Plan) {
	t.Helper()
	const eps = 1e-5

	bNorm := NewBuilder()
	xIn := bNorm.Input("x_in", F32, m, h)
	wNorm := bNorm.Input("emb_norm", F32, h)
	xn0 := bNorm.Cast(BF16, bNorm.RMSNorm(xIn, wNorm, h, eps))
	normExpr, err := Optimize(bNorm.G, xn0)
	if err != nil {
		t.Fatalf("optimize norm: %v", err)
	}
	norm, err = Compile(normExpr, "emb_rmsnorm", BF16)
	if err != nil {
		t.Fatalf("compile norm: %v", err)
	}

	postProto, attnInProto := buildTestCrossLayerPlans(t, m, h, qDim, kvDim, inter, true, true)
	layers = make([]DecoderLayerPlans, numLayers)
	for i := 0; i < numLayers; i++ {
		attnIn := clonePlan(attnInProto)
		attnIn.Name = fmt.Sprintf("l%d_attn_in", i)
		setPlanKernelName(&attnIn.Kernels[0], fmt.Sprintf("l%d_qkv_proj", i))

		post := clonePlan(postProto)
		post.Name = fmt.Sprintf("l%d_post", i)
		for k := range post.Kernels {
			oldName := planKernelName(post.Kernels[k])
			newName := strings.Replace(oldName, "l0_", fmt.Sprintf("l%d_", i), 1)
			setPlanKernelName(&post.Kernels[k], newName)
		}
		layers[i] = DecoderLayerPlans{
			LayerIndex: i,
			AttnIn:     attnIn,
			Post:       post,
		}
	}

	bHead := NewBuilder()
	xnFinal := bHead.Input("xn", BF16, m, h)
	wHead := bHead.Input("w_head", BF16, vocab, h)
	logits := bHead.MatMul(xnFinal, wHead)
	headExpr, err := Optimize(bHead.G, logits)
	if err != nil {
		t.Fatalf("optimize head: %v", err)
	}
	head, err = Compile(headExpr, "lm_head", F32)
	if err != nil {
		t.Fatalf("compile head: %v", err)
	}
	if head.Kernels[0].GEMM != nil {
		head.Kernels[0].GEMM.RowsPerBlock = 32
	}
	return norm, layers, head
}

func TestFuseDecoderLayerStack(t *testing.T) {
	tests := []struct {
		name          string
		numLayers     int
		withNorm      bool
		withHead      bool
		fuseFinalHead bool
		packArena     bool
		mega          bool
		wantUnfused   int
		wantFused     int
	}{
		{
			name:          "1_layer_basic",
			numLayers:     1,
			withNorm:      true,
			withHead:      true,
			fuseFinalHead: false,
			packArena:     true,
			mega:          true,
			wantUnfused:   4, // 2*1 + 2
			wantFused:     4, // 1 + 1 + 2
		},
		{
			name:          "1_layer_fuse_final_head",
			numLayers:     1,
			withNorm:      true,
			withHead:      true,
			fuseFinalHead: true,
			packArena:     true,
			mega:          true,
			wantUnfused:   4, // 2*1 + 2
			wantFused:     3, // Norm + FirstAttnIn + FinalPost(fused head)
		},
		{
			name:          "4_layer_stack_with_norm_and_head",
			numLayers:     4,
			withNorm:      true,
			withHead:      true,
			fuseFinalHead: false,
			packArena:     true,
			mega:          true,
			wantUnfused:   10, // 2*4 + 2
			wantFused:     7,  // 4 + 2 + 1
		},
		{
			name:          "4_layer_stack_fuse_final_head",
			numLayers:     4,
			withNorm:      true,
			withHead:      true,
			fuseFinalHead: true,
			packArena:     true,
			mega:          true,
			wantUnfused:   10, // 2*4 + 2
			wantFused:     6,  // 4 + 1 + 1
		},
		{
			name:          "8_layer_stack_full",
			numLayers:     8,
			withNorm:      true,
			withHead:      true,
			fuseFinalHead: false,
			packArena:     true,
			mega:          true,
			wantUnfused:   18, // 2*8 + 2
			wantFused:     11, // 8 + 2 + 1
		},
		{
			name:          "8_layer_stack_fuse_final_head",
			numLayers:     8,
			withNorm:      true,
			withHead:      true,
			fuseFinalHead: true,
			packArena:     true,
			mega:          true,
			wantUnfused:   18, // 2*8 + 2
			wantFused:     10, // 8 + 1 + 1
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			norm, layers, head := buildTestStackPlans(t, tc.numLayers, 1, 256, 256, 64, 256, 256)
			origFirstKernels := len(layers[0].Post.Kernels)

			cfg := CrossLayerStackConfig{
				NamePrefix:         "llama_cross",
				BridgeInput:        "xn",
				RetainBridgeOutput: true,
				FuseFinalHead:      tc.fuseFinalHead,
				PackArena:          tc.packArena,
				Mega:               tc.mega,
			}
			if tc.withNorm {
				cfg.Norm = norm
			}
			if tc.withHead {
				cfg.Head = head
			}

			stack, err := FuseDecoderLayerStack(layers, cfg)
			if err != nil {
				t.Fatalf("FuseDecoderLayerStack: %v", err)
			}

			// Input plans must not be mutated.
			if len(layers[0].Post.Kernels) != origFirstKernels {
				t.Fatalf("caller layer[0].Post was mutated: kernels=%d, want %d", len(layers[0].Post.Kernels), origFirstKernels)
			}

			if got := stack.UnfusedLaunchCount(); got != tc.wantUnfused {
				t.Fatalf("UnfusedLaunchCount() = %d, want %d", got, tc.wantUnfused)
			}
			if got := stack.FusedLaunchCount(); got != tc.wantFused {
				t.Fatalf("FusedLaunchCount() = %d, want %d", got, tc.wantFused)
			}
			if got := len(stack.CrossBoundaries); got != tc.numLayers-1 {
				t.Fatalf("len(CrossBoundaries) = %d, want %d", got, tc.numLayers-1)
			}
			if tc.packArena && stack.ArenaBytes == 0 {
				t.Fatal("expected non-zero ArenaBytes when PackArena=true")
			}

			if tc.mega {
				if got := len(stack.CrossMegakernels); got != tc.numLayers-1 {
					t.Fatalf("len(CrossMegakernels) = %d, want %d", got, tc.numLayers-1)
				}
				for i, mk := range stack.CrossMegakernels {
					wantName := fmt.Sprintf("llama_cross_l%d_l%d_mega", i, i+1)
					if mk.Name != wantName {
						t.Fatalf("CrossMegakernels[%d].Name = %q, want %q", i, mk.Name, wantName)
					}
					attr := mk.StageAttributionMap()
					if len(attr) != 6 {
						t.Fatalf("CrossMegakernels[%d] StageAttributionMap len = %d, want 6", i, len(attr))
					}
					qkvStage := fmt.Sprintf("l%d_qkv_proj", i+1)
					if _, ok := attr[qkvStage]; !ok {
						t.Fatalf("CrossMegakernels[%d] StageAttributionMap missing %q in %v", i, qkvStage, attr)
					}
				}
				if tc.fuseFinalHead {
					if stack.FinalMegakernel == nil {
						t.Fatal("expected FinalMegakernel when FuseFinalHead=true and Mega=true")
					}
					if stack.Head != nil {
						t.Fatal("expected Head=nil when FuseFinalHead=true")
					}
					if len(stack.FinalMegakernel.StageAttributionMap()) != 6 {
						t.Fatalf("FinalMegakernel StageAttributionMap len = %d, want 6", len(stack.FinalMegakernel.StageAttributionMap()))
					}
				}
			}
		})
	}
}

func TestFuseDecoderLayerStackErrors(t *testing.T) {
	if _, err := FuseDecoderLayerStack(nil, CrossLayerStackConfig{}); err == nil {
		t.Error("empty layers: expected error")
	}
	_, layers, _ := buildTestStackPlans(t, 2, 1, 256, 256, 64, 256, 256)
	if _, err := FuseDecoderLayerStack(layers, CrossLayerStackConfig{
		PackArena:     true,
		ArenaMaxBytes: 16, // too small
	}); err == nil {
		t.Error("insufficient ArenaMaxBytes: expected error")
	}
	badLayers := slices.Clone(layers)
	badLayers[1].AttnIn = nil
	if _, err := FuseDecoderLayerStack(badLayers, CrossLayerStackConfig{}); err == nil {
		t.Error("nil AttnIn: expected error")
	}
}
