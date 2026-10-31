package hopper

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
)

func populateFullLayerTestEnv(r *rand.Rand, p *Plan, batchM, maxPages int) map[string]Value {
	env := randEnv(r, p.Inputs...)
	setMeta := func(prefix string) {
		pos := make([]float64, batchM)
		seqIDs := make([]float64, batchM)
		seqLens := make([]float64, batchM)
		kvPos := make([]float64, batchM)
		pt := make([]float64, batchM*maxPages)
		for b := range batchM {
			pos[b] = 0
			seqIDs[b] = float64(b)
			seqLens[b] = 1
			kvPos[b] = 0
			for pg := range maxPages {
				pt[b*maxPages+pg] = float64(b*maxPages + pg)
			}
		}
		if _, ok := env[prefix+"positions"]; ok {
			env[prefix+"positions"] = Value{Shape: []int{batchM}, Data: pos}
		}
		if _, ok := env[prefix+"seq_ids"]; ok {
			env[prefix+"seq_ids"] = Value{Shape: []int{batchM}, Data: seqIDs}
		}
		if _, ok := env[prefix+"seqlens"]; ok {
			env[prefix+"seqlens"] = Value{Shape: []int{batchM}, Data: seqLens}
		}
		if _, ok := env[prefix+"kv_positions"]; ok {
			env[prefix+"kv_positions"] = Value{Shape: []int{batchM}, Data: kvPos}
		}
		if _, ok := env[prefix+"page_table"]; ok {
			env[prefix+"page_table"] = Value{Shape: []int{batchM, maxPages}, Data: pt}
		}
		if _, ok := env[prefix+"k_scale"]; ok {
			env[prefix+"k_scale"] = Value{Shape: []int{1}, Data: []float64{1.0}}
		}
		if _, ok := env[prefix+"v_scale"]; ok {
			env[prefix+"v_scale"] = Value{Shape: []int{1}, Data: []float64{1.0}}
		}
	}
	setMeta("")
	for l := range 8 {
		setMeta(fmt.Sprintf("l%d_", l))
	}
	return env
}

// evalFullLayerHostEmu executes the monolithic Megakernel in host emulation
// alongside sequential execution of each individual layer's Plan and returns
// both outputs and the final ready counter words.
func evalFullLayerHostEmu(t *testing.T, mega *Megakernel, layerPlans []*Plan, env map[string]Value) (megaOut, seqOut []float64, readyWords []uint64) {
	t.Helper()
	lineSrc, _, err := mega.LineInfoSource()
	if err != nil {
		t.Fatalf("LineInfoSource: %v", err)
	}
	emuSrc := strings.ReplaceAll(lineSrc, "__nanosleep(32);", "__nanosleep(32); std::this_thread::yield();")
	outs, ready, err := runMegakernelSourceEpochs(t, *mega, emuSrc, []map[string]Value{env})
	if err != nil {
		t.Fatalf("runMegakernelSourceEpochs: %v", err)
	}
	megaOut = outs[0]
	readyWords = ready

	curX := env["x"]
	for i, lp := range layerPlans {
		lPrefix := fmt.Sprintf("l%d_", i)
		layerEnv := make(map[string]Value, len(lp.Inputs))
		for _, in := range lp.Inputs {
			if in.Name == "x" {
				layerEnv["x"] = curX
				continue
			}
			if v, ok := env[lPrefix+in.Name]; ok {
				layerEnv[in.Name] = v
			} else if v, ok := env[in.Name]; ok {
				layerEnv[in.Name] = v
			} else {
				t.Fatalf("layer %d missing input %q (or %q)", i, in.Name, lPrefix+in.Name)
			}
		}
		outData := runPlanSource(t, lp, layerEnv)
		curX = Value{
			Shape: slices.Clone(lp.Output.Shape),
			Data:  outData,
		}
	}
	seqOut = curX.Data
	return megaOut, seqOut, readyWords
}

func assertFullLayerStageKinds(t *testing.T, stages []MegaStage, numLayers int) {
	t.Helper()
	wantSuffixes := []string{
		"attn_norm",
		"qkv_proj",
		"rope_kv",
		"paged_attn",
		"o_proj",
		"ffn_norm",
		"gate_proj",
		"up_proj",
		"down_proj",
	}
	if len(stages) != 9*numLayers {
		t.Fatalf("len(stages) = %d, want %d", len(stages), 9*numLayers)
	}
	for l := range numLayers {
		base := l * 9
		for s, suf := range wantSuffixes {
			st := stages[base+s]
			wantName := fmt.Sprintf("l%d_%s", l, suf)
			if st.Name != wantName {
				t.Fatalf("stage[%d].Name = %q, want %q", base+s, st.Name, wantName)
			}
			k := st.Kernel
			switch s {
			case 0, 5:
				if k.Row == nil {
					t.Fatalf("stage[%d] (%s): expected RowKernel", base+s, st.Name)
				}
			case 1, 4, 6, 7, 8:
				if k.GEMM == nil {
					t.Fatalf("stage[%d] (%s): expected SkinnyGEMM", base+s, st.Name)
				}
			case 2:
				if k.RoPE == nil {
					t.Fatalf("stage[%d] (%s): expected QKNormRoPEKVAppend", base+s, st.Name)
				}
				if st.Threads != 128 {
					t.Fatalf("stage[%d] (%s): Threads = %d, want 128", base+s, st.Name, st.Threads)
				}
			case 3:
				if k.Attention == nil {
					t.Fatalf("stage[%d] (%s): expected PagedAttention", base+s, st.Name)
				}
				if st.Threads != 128 {
					t.Fatalf("stage[%d] (%s): Threads = %d, want 128", base+s, st.Name, st.Threads)
				}
			}
		}
	}
}

func TestFullLayerMegakernel1Layer(t *testing.T) {
	spec := FullDecoderLayerSpec{
		LayerIndex: 0,
		BatchM:     1,
		HiddenDim:  16,
		IntermDim:  32,
		NumQHeads:  2,
		NumKVHeads: 1,
		HeadDim:    8,
		PageSize:   16,
		MaxPages:   4,
		RoPETheta:  10000.0,
		RMSEps:     1e-5,
		Arch:       TargetSM121A,
	}

	layerPlan, err := BuildFullDecoderLayerPlan(spec)
	if err != nil {
		t.Fatalf("BuildFullDecoderLayerPlan: %v", err)
	}
	if len(layerPlan.Kernels) != 9 {
		t.Fatalf("len(layerPlan.Kernels) = %d, want 9", len(layerPlan.Kernels))
	}

	fusedPlan, mega, err := BuildFullLayerMegakernel("tiny_full_1l_mega", []FullDecoderLayerSpec{spec})
	if err != nil {
		t.Fatalf("BuildFullLayerMegakernel: %v", err)
	}
	if len(fusedPlan.Kernels) != 9 {
		t.Fatalf("len(fusedPlan.Kernels) = %d, want 9", len(fusedPlan.Kernels))
	}
	assertFullLayerStageKinds(t, mega.Stages, 1)

	cudaSrc, err := mega.CUDA()
	if err != nil {
		t.Fatalf("mega.CUDA(): %v", err)
	}
	if !strings.Contains(cudaSrc, `asm volatile("bar.sync 1, 128;")`) {
		t.Fatal("missing bar.sync 1, 128; in CUDA source")
	}
	if strings.Count(cudaSrc, "if (threadIdx.x >= 128) return;") < 2 {
		t.Fatal("expected threadIdx.x >= 128 guard in both RoPE and Attention stages")
	}
	if !strings.Contains(cudaSrc, "eg_ready_resolve_epoch") {
		t.Fatal("missing eg_ready_resolve_epoch in CUDA source")
	}

	lineSrc, spans, err := mega.LineInfoSource()
	if err != nil {
		t.Fatalf("mega.LineInfoSource(): %v", err)
	}
	if len(spans) != 9 {
		t.Fatalf("len(spans) = %d, want 9", len(spans))
	}
	if !strings.Contains(lineSrc, "#line 1 \"tiny_full_1l_mega.cu\"") {
		t.Fatal("missing megakernel entry #line directive")
	}

	r := rand.New(rand.NewPCG(101, 102))
	env := populateFullLayerTestEnv(r, fusedPlan, spec.BatchM, spec.MaxPages)
	megaOut, seqOut, readyWords := evalFullLayerHostEmu(t, mega, []*Plan{layerPlan}, env)
	agree(t, "full_layer_1l_mega_vs_seq", F32, megaOut, seqOut)

	for i := 0; i < len(mega.Stages)-1; i++ {
		want := PackReadyWord(1, uint32(mega.Stages[i].Blocks))
		if readyWords[i] != want {
			t.Fatalf("ready[%d] = %#x, want %#x", i, readyWords[i], want)
		}
	}
}

func TestFullLayerMegakernel2Layer(t *testing.T) {
	specs := []FullDecoderLayerSpec{
		{
			LayerIndex: 0,
			BatchM:     1,
			HiddenDim:  16,
			IntermDim:  32,
			NumQHeads:  2,
			NumKVHeads: 1,
			HeadDim:    8,
			PageSize:   16,
			MaxPages:   4,
			RoPETheta:  10000.0,
			RMSEps:     1e-5,
			Arch:       TargetSM90A,
		},
		{
			LayerIndex: 1,
			BatchM:     1,
			HiddenDim:  16,
			IntermDim:  32,
			NumQHeads:  2,
			NumKVHeads: 1,
			HeadDim:    8,
			PageSize:   16,
			MaxPages:   4,
			RoPETheta:  10000.0,
			RMSEps:     1e-5,
			Arch:       TargetSM90A,
		},
	}

	l0Plan, err := BuildFullDecoderLayerPlan(specs[0])
	if err != nil {
		t.Fatalf("BuildFullDecoderLayerPlan(0): %v", err)
	}
	l1Plan, err := BuildFullDecoderLayerPlan(specs[1])
	if err != nil {
		t.Fatalf("BuildFullDecoderLayerPlan(1): %v", err)
	}

	fusedPlan, mega, err := BuildFullLayerMegakernel("tiny_full_2l_mega", specs)
	if err != nil {
		t.Fatalf("BuildFullLayerMegakernel(2L): %v", err)
	}
	if len(fusedPlan.Kernels) != 18 {
		t.Fatalf("len(fusedPlan.Kernels) = %d, want 18", len(fusedPlan.Kernels))
	}
	assertFullLayerStageKinds(t, mega.Stages, 2)

	// Verify cross-layer producer-consumer WaitSlots:
	// Stage 9 (l1_attn_norm) and Stage 13 (l1_o_proj residual add) must wait on Stage 8 (l0_down_proj).
	if !slices.Contains(mega.Stages[9].WaitSlots, 8) {
		t.Fatalf("stage 9 (%s) WaitSlots = %v, want to contain stage 8 (l0_down_proj)",
			mega.Stages[9].Name, mega.Stages[9].WaitSlots)
	}
	if !slices.Contains(mega.Stages[13].WaitSlots, 8) || !slices.Contains(mega.Stages[13].WaitSlots, 12) {
		t.Fatalf("stage 13 (%s) WaitSlots = %v, want to contain stages 8 and 12",
			mega.Stages[13].Name, mega.Stages[13].WaitSlots)
	}
	// Stage 7 (l0_up_proj) waits on Stage 5 (l0_ffn_norm) and Stage 6 (l0_gate_proj).
	if !slices.Contains(mega.Stages[7].WaitSlots, 5) || !slices.Contains(mega.Stages[7].WaitSlots, 6) {
		t.Fatalf("stage 7 (%s) WaitSlots = %v, want to contain stages 5 and 6",
			mega.Stages[7].Name, mega.Stages[7].WaitSlots)
	}
	// Stage 16 (l1_up_proj) waits on Stage 14 (l1_ffn_norm) and Stage 15 (l1_gate_proj).
	if !slices.Contains(mega.Stages[16].WaitSlots, 14) || !slices.Contains(mega.Stages[16].WaitSlots, 15) {
		t.Fatalf("stage 16 (%s) WaitSlots = %v, want to contain stages 14 and 15",
			mega.Stages[16].Name, mega.Stages[16].WaitSlots)
	}

	cudaSrc, err := mega.CUDA()
	if err != nil {
		t.Fatalf("mega.CUDA(): %v", err)
	}
	if !strings.Contains(cudaSrc, `asm volatile("bar.sync 1, 128;")`) {
		t.Fatal("missing bar.sync 1, 128; in 2-layer CUDA source")
	}
	if strings.Count(cudaSrc, "if (threadIdx.x >= 128) return;") < 4 {
		t.Fatal("expected threadIdx.x >= 128 guard in all 4 RoPE/Attention stages across 2 layers")
	}
	if !strings.Contains(cudaSrc, "eg_ready_resolve_epoch") {
		t.Fatal("missing eg_ready_resolve_epoch in 2-layer CUDA source")
	}

	lineSrc, spans, err := mega.LineInfoSource()
	if err != nil {
		t.Fatalf("mega.LineInfoSource(): %v", err)
	}
	if len(spans) != 18 {
		t.Fatalf("len(spans) = %d, want 18", len(spans))
	}
	attrMap := mega.StageAttributionMap()
	if len(attrMap) != 18 {
		t.Fatalf("len(StageAttributionMap) = %d, want 18", len(attrMap))
	}

	r := rand.New(rand.NewPCG(201, 202))
	env := populateFullLayerTestEnv(r, fusedPlan, specs[0].BatchM, specs[0].MaxPages)
	megaOut, seqOut, readyWords := evalFullLayerHostEmu(t, mega, []*Plan{l0Plan, l1Plan}, env)
	for i, v := range megaOut {
		if math.IsNaN(v) {
			t.Fatalf("megaOut[%d] is NaN", i)
		}
	}
	agree(t, "full_layer_2l_mega_vs_seq", F32, megaOut, seqOut)

	for i := 0; i < len(mega.Stages)-1; i++ {
		want := PackReadyWord(1, uint32(mega.Stages[i].Blocks))
		if readyWords[i] != want {
			t.Fatalf("ready[%d] = %#x, want %#x", i, readyWords[i], want)
		}
	}

	// Verify multi-epoch execution without resetting ready counters between epochs.
	env2 := populateFullLayerTestEnv(r, fusedPlan, specs[0].BatchM, specs[0].MaxPages)
	outs2, ready2, err := runMegakernelSourceEpochs(t, *mega, lineSrc, []map[string]Value{env, env2})
	if err != nil {
		t.Fatalf("2-epoch runMegakernelSourceEpochs: %v", err)
	}
	agree(t, "epoch1_repeat", F32, outs2[0], megaOut)
	_, seqOut2, _ := evalFullLayerHostEmu(t, mega, []*Plan{l0Plan, l1Plan}, env2)
	agree(t, "epoch2_vs_seq", F32, outs2[1], seqOut2)
	for i := 0; i < len(mega.Stages)-1; i++ {
		want := PackReadyWord(2, uint32(mega.Stages[i].Blocks))
		if ready2[i] != want {
			t.Fatalf("2-epoch ready[%d] = %#x, want %#x", i, ready2[i], want)
		}
	}
}

func TestFullLayerMegakernelErrors(t *testing.T) {
	if _, _, err := BuildFullLayerMegakernel("x", nil); err == nil {
		t.Error("empty specs: expected error")
	}
	valid := FullDecoderLayerSpec{
		LayerIndex: 0,
		BatchM:     1,
		HiddenDim:  16,
		IntermDim:  32,
		NumQHeads:  2,
		NumKVHeads: 1,
		HeadDim:    8,
		PageSize:   16,
		MaxPages:   4,
	}
	if _, _, err := BuildFullLayerMegakernel("eg_bad", []FullDecoderLayerSpec{valid}); err == nil {
		t.Error("bad megakernel name: expected error")
	}
	if _, _, err := BuildFullLayerMegakernel("dup_idx", []FullDecoderLayerSpec{
		{LayerIndex: 1, BatchM: 1, HiddenDim: 16, IntermDim: 32, NumQHeads: 2, NumKVHeads: 1, HeadDim: 8},
		{LayerIndex: 1, BatchM: 1, HiddenDim: 16, IntermDim: 32, NumQHeads: 2, NumKVHeads: 1, HeadDim: 8},
	}); err == nil {
		t.Error("duplicate LayerIndex: expected error")
	}
	for _, bad := range []FullDecoderLayerSpec{
		{LayerIndex: -1, BatchM: 1, HiddenDim: 16, IntermDim: 32, NumQHeads: 2, NumKVHeads: 1, HeadDim: 8},
		{LayerIndex: 0, BatchM: 32, HiddenDim: 16, IntermDim: 32, NumQHeads: 2, NumKVHeads: 1, HeadDim: 8},
		{LayerIndex: 0, BatchM: 1, HiddenDim: 0, IntermDim: 32, NumQHeads: 2, NumKVHeads: 1, HeadDim: 8},
		{LayerIndex: 0, BatchM: 1, HiddenDim: 16, IntermDim: 0, NumQHeads: 2, NumKVHeads: 1, HeadDim: 8},
		{LayerIndex: 0, BatchM: 1, HiddenDim: 16, IntermDim: 32, NumQHeads: 3, NumKVHeads: 2, HeadDim: 8},
		{LayerIndex: 0, BatchM: 1, HiddenDim: 16, IntermDim: 32, NumQHeads: 2, NumKVHeads: 1, HeadDim: 7},
	} {
		if _, err := BuildFullDecoderLayerPlan(bad); err == nil {
			t.Errorf("BuildFullDecoderLayerPlan(%+v): expected error", bad)
		}
	}
}
