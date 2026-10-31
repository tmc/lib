package specdecode

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudagrammar"
	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

func buildTestFinalPostAndHeadPlans(t *testing.T, m, h, qDim, inter, vocab int, fuseTails bool) (finalPost, head *hopper.Plan) {
	t.Helper()
	const eps = 1e-5

	// Final decoder layer post: o_proj -> rmsnorm -> gate_up -> swiglu -> down_proj -> final_rmsnorm.
	bPost := hopper.NewBuilder()
	attn := bPost.Input("attn", hopper.BF16, m, qDim)
	x := bPost.Input("x", hopper.F32, m, h)
	wo := bPost.Input("w_o", hopper.BF16, h, qDim)
	hid := bPost.Map(hopper.MustLambda("(add $0 $1)"), bPost.MatMul(attn, wo), x)
	ffnNorm := bPost.Cast(hopper.BF16, bPost.RMSNorm(hid, bPost.Input("ffn_norm", hopper.F32, h), h, eps))
	act := bPost.SwiGLU(bPost.MatMul(ffnNorm, bPost.Input("w_gu", hopper.BF16, 2*inter, h)))
	down := bPost.Map(hopper.MustLambda("(add $0 $1)"), bPost.MatMul(act, bPost.Input("w_down", hopper.BF16, h, inter)), hid)
	xnOut := bPost.Cast(hopper.BF16, bPost.RMSNorm(down, bPost.Input("final_norm", hopper.F32, h), h, eps))

	postExpr, err := hopper.Optimize(bPost.G, xnOut)
	if err != nil {
		t.Fatalf("optimize finalPost: %v", err)
	}
	finalPost, err = hopper.Compile(postExpr, "final_post", hopper.BF16)
	if err != nil {
		t.Fatalf("compile finalPost: %v", err)
	}
	if len(finalPost.Kernels) != 5 {
		t.Fatalf("finalPost has %d kernels before FuseTails, want 5", len(finalPost.Kernels))
	}
	stageNames := []string{
		"final_o_proj",
		"final_residual_rmsnorm",
		"final_gate_up",
		"final_down_proj",
		"final_rmsnorm",
	}
	for i, name := range stageNames {
		k := &finalPost.Kernels[i]
		switch {
		case k.GEMM != nil:
			k.GEMM.Name = name
			k.GEMM.RowsPerBlock = 32
		case k.Row != nil:
			k.Row.Name = name
		}
	}
	if fuseTails {
		if fused := finalPost.FuseTails(); fused != 2 {
			t.Fatalf("finalPost.FuseTails() = %d, want 2", fused)
		}
	}

	// lm_head projection: xn [M, H] * w_head [V, H] -> logits [M, V].
	bHead := hopper.NewBuilder()
	xnIn := bHead.Input("xn", hopper.BF16, m, h)
	wHead := bHead.Input("w_head", hopper.BF16, vocab, h)
	logits := bHead.MatMul(xnIn, wHead)
	headExpr, err := hopper.Optimize(bHead.G, logits)
	if err != nil {
		t.Fatalf("optimize head: %v", err)
	}
	head, err = hopper.Compile(headExpr, "lm_head", hopper.F32)
	if err != nil {
		t.Fatalf("compile head: %v", err)
	}
	if len(head.Kernels) != 1 {
		t.Fatalf("head has %d kernels, want 1", len(head.Kernels))
	}
	if head.Kernels[0].GEMM != nil {
		head.Kernels[0].GEMM.Name = "lm_head"
		head.Kernels[0].GEMM.RowsPerBlock = 32
	}
	return finalPost, head
}

func TestFinalHeadVerifySampleMegakernelBuildAndSource(t *testing.T) {
	const (
		mNodes = 4
		h      = 256
		qDim   = 256
		inter  = 256
		vocab  = 256
	)
	finalPost, head := buildTestFinalPostAndHeadPlans(t, mNodes, h, qDim, inter, vocab, true)

	mega, err := BuildFinalHeadVerifySampleMegakernel(finalPost, head, FusedVerifySampleConfig{
		Name:               "llama_final_head_verify_sample_mega",
		BridgeInput:        "xn",
		DraftNodes:         mNodes,
		VocabSize:          vocab,
		DefaultTemperature: 0.8,
	})
	if err != nil {
		t.Fatalf("BuildFinalHeadVerifySampleMegakernel: %v", err)
	}

	wantStages := []string{
		"final_o_proj",
		"final_residual_rmsnorm",
		"final_gate_up",
		"final_down_proj",
		"final_rmsnorm",
		"head_lm_head",
		"eg_spec_grammar_verify_sample",
	}
	if len(mega.Stages) != len(wantStages) {
		t.Fatalf("len(mega.Stages) = %d, want %d", len(mega.Stages), len(wantStages))
	}
	for i, want := range wantStages {
		if got := mega.Stages[i].Name; got != want {
			t.Fatalf("stage[%d].Name = %q, want %q", i, got, want)
		}
	}

	// Check ready synchronization wiring between head_lm_head and eg_spec_grammar_verify_sample.
	if mega.HeadStageIndex != 5 || mega.HeadReadySlot != 5 {
		t.Fatalf("HeadStageIndex=%d HeadReadySlot=%d, want 5, 5", mega.HeadStageIndex, mega.HeadReadySlot)
	}
	if mega.Stages[5].SignalSlot != 5 {
		t.Fatalf("head stage SignalSlot = %d, want 5", mega.Stages[5].SignalSlot)
	}
	if !slices.Equal(mega.Stages[6].WaitSlots, []int{5}) {
		t.Fatalf("verify stage WaitSlots = %v, want [5]", mega.Stages[6].WaitSlots)
	}
	if mega.Stages[6].SignalSlot != -1 {
		t.Fatalf("verify stage SignalSlot = %d, want -1", mega.Stages[6].SignalSlot)
	}

	// Verify StageAttributionMap and #line directives across all 7 stages.
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

	// Verify required device symbols and primitives in emitted CUDA source.
	wantSubstrings := []string{
		"EG_DEV void eg_spec_grammar_verify_sample_dev(",
		fmt.Sprintf("eg_ready_wait(&ready[%d], epoch, %d);", mega.HeadReadySlot, mega.HeadBlocks),
		fmt.Sprintf("eg_ready_signal(&ready[%d], epoch, 1);", mega.HeadReadySlot),
		"(vocab_bitmask[v >> 5] >> (v & 31)) & 1u",
		"-INFINITY",
		"eg_philox4x32(",
		"eg_philox_uniform(",
		"accepted_tokens",
		"num_accepted",
		"next_bonus_token",
	}
	for _, sub := range wantSubstrings {
		if !strings.Contains(lineSrc, sub) {
			t.Errorf("emitted source missing %q", sub)
		}
	}
}

func TestFinalHeadVerifySampleCPUReference(t *testing.T) {
	const vocab = 64

	t.Run("linear_chain_all_accepted_greedy_and_sampled", func(t *testing.T) {
		logits := make([][]float32, 4)
		for m := range logits {
			logits[m] = make([]float32, vocab)
			for v := range vocab {
				logits[m][v] = -10.0
			}
		}
		// High probability target tokens matching draftIDs[0..3] = {5, 12, 33, 47}.
		draftIDs := []int32{5, 12, 33, 47}
		draftProbs := []float32{0.8, 0.7, 0.9, 0.6}
		parentIdx := []int32{-1, 0, 1, 2}
		for m, tok := range draftIDs {
			logits[m][tok] = 20.0
		}

		for _, temp := range []float32{0.0, 0.7} {
			res, err := EvalFinalHeadVerifySample(logits, nil, draftIDs, draftProbs, parentIdx, temp, 0x12345678, 1)
			if err != nil {
				t.Fatalf("EvalFinalHeadVerifySample(temp=%v): %v", temp, err)
			}
			if res.NumAccepted != 4 || !slices.Equal(res.AcceptedTokens, draftIDs) {
				t.Fatalf("temp=%v: got NumAccepted=%d AcceptedTokens=%v, want 4 %v",
					temp, res.NumAccepted, res.AcceptedTokens, draftIDs)
			}
			if res.NextBonusToken != 47 {
				t.Fatalf("temp=%v: got NextBonusToken=%d, want 47", temp, res.NextBonusToken)
			}
		}
	})

	t.Run("branching_tree_fallback_to_sibling", func(t *testing.T) {
		// Tree of 4 nodes:
		//   node 0 (parent -1): token 3 (accepted)
		//   node 1 (parent  0): token 10 (rejected: low target logit, high draft prob)
		//   node 2 (parent  0): token 22 (accepted: sibling of node 1!)
		//   node 3 (parent  1): token 50 (descendant of rejected node 1: must be skipped!)
		logits := make([][]float32, 4)
		for m := range logits {
			logits[m] = make([]float32, vocab)
			for v := range vocab {
				logits[m][v] = -15.0
			}
		}
		logits[0][3] = 25.0  // node 0 accepts token 3
		logits[1][22] = 25.0 // node 1 prefers 22, so draftIDs[1]=10 has ~0 prob and is rejected
		logits[2][22] = 25.0 // node 2 (sibling with parent 0) proposes 22 and is accepted
		logits[3][50] = 25.0 // node 3 has parent 1 (which was rejected), so skipped

		draftIDs := []int32{3, 10, 22, 50}
		draftProbs := []float32{0.9, 0.95, 0.85, 0.9}
		parentIdx := []int32{-1, 0, 0, 1}

		res, err := EvalFinalHeadVerifySample(logits, nil, draftIDs, draftProbs, parentIdx, 0.5, 42, 7)
		if err != nil {
			t.Fatalf("EvalFinalHeadVerifySample: %v", err)
		}
		wantAccepted := []int32{3, 22}
		if res.NumAccepted != 2 || !slices.Equal(res.AcceptedTokens, wantAccepted) {
			t.Fatalf("got NumAccepted=%d AcceptedTokens=%v, want 2 %v",
				res.NumAccepted, res.AcceptedTokens, wantAccepted)
		}
		if !slices.Equal(res.AcceptedNodes, []int32{0, 2}) {
			t.Fatalf("AcceptedNodes = %v, want [0, 2]", res.AcceptedNodes)
		}
		if res.NextBonusToken != 22 {
			t.Fatalf("NextBonusToken = %d, want 22", res.NextBonusToken)
		}
	})

	t.Run("grammar_bitmask_rejects_disallowed_draft_and_constrains_bonus", func(t *testing.T) {
		// Build a cudagrammar.Bitmask that only allows tokens 17 and 41.
		bm := cudagrammar.NewBitmask(vocab)
		bm.Set(17, true)
		bm.Set(41, true)

		logits := make([][]float32, 3)
		for m := range logits {
			logits[m] = make([]float32, vocab)
			for v := range vocab {
				logits[m][v] = -5.0
			}
		}
		// Position 0 allows token 17 (which is drafted at node 0).
		logits[0][17] = 10.0
		// Position 1 has a huge raw logit on token 9 (disallowed by grammar!)
		// and moderate logits on allowed tokens 17 and 41, with 41 > 17.
		logits[1][9] = 100.0
		logits[1][17] = 2.0
		logits[1][41] = 8.0
		logits[2][41] = 10.0

		draftIDs := []int32{17, 9, 41}
		draftProbs := []float32{0.5, 0.5, 0.5}
		parentIdx := []int32{-1, 0, 1}

		// Without grammar mask, token 9 at position 1 is accepted.
		unmasked, err := EvalFinalHeadVerifySample(logits, nil, draftIDs, draftProbs, parentIdx, 0.5, 99, 1)
		if err != nil {
			t.Fatal(err)
		}
		if unmasked.NumAccepted != 3 {
			t.Fatalf("unmasked NumAccepted = %d, want 3", unmasked.NumAccepted)
		}

		// With grammar mask, token 9 at position 1 is masked to -INFINITY,
		// so node 1 is rejected and NextBonusToken must be an allowed token (41).
		masked, err := EvalFinalHeadVerifySample(logits, bm.Words, draftIDs, draftProbs, parentIdx, 0.5, 99, 1)
		if err != nil {
			t.Fatal(err)
		}
		if masked.NumAccepted != 1 || !slices.Equal(masked.AcceptedTokens, []int32{17}) {
			t.Fatalf("masked NumAccepted=%d AcceptedTokens=%v, want 1 [17]",
				masked.NumAccepted, masked.AcceptedTokens)
		}
		if !bm.IsAllowed(int(masked.NextBonusToken)) {
			t.Fatalf("masked NextBonusToken=%d is not allowed by grammar bitmask", masked.NextBonusToken)
		}
		if masked.NextBonusToken != 41 {
			t.Fatalf("masked NextBonusToken=%d, want 41", masked.NextBonusToken)
		}
	})

	t.Run("philox_deterministic_reproducibility", func(t *testing.T) {
		r := rand.New(rand.NewPCG(77, 88))
		logits := make([][]float32, 4)
		for m := range logits {
			logits[m] = make([]float32, vocab)
			for v := range vocab {
				logits[m][v] = float32(r.NormFloat64() * 1.5)
			}
		}
		draftIDs := []int32{4, 19, 27, 55}
		draftProbs := []float32{0.6, 0.6, 0.6, 0.6}
		parentIdx := []int32{-1, 0, 1, 2}

		res1, err := EvalFinalHeadVerifySample(logits, nil, draftIDs, draftProbs, parentIdx, 1.0, 0xdeadbeefcafe, 42)
		if err != nil {
			t.Fatal(err)
		}
		res2, err := EvalFinalHeadVerifySample(logits, nil, draftIDs, draftProbs, parentIdx, 1.0, 0xdeadbeefcafe, 42)
		if err != nil {
			t.Fatal(err)
		}
		if res1.NumAccepted != res2.NumAccepted ||
			res1.NextBonusToken != res2.NextBonusToken ||
			!slices.Equal(res1.AcceptedTokens, res2.AcceptedTokens) {
			t.Fatalf("non-deterministic result for identical (seed, offset): %+v vs %+v", res1, res2)
		}
	})
}

func encodePlanTensorData(dt hopper.DType, vals []float32) []byte {
	switch dt {
	case hopper.F32:
		return f32Bytes(vals)
	case hopper.BF16:
		u16 := make([]uint16, len(vals))
		for i, v := range vals {
			u16[i] = hopper.BF16Bits(v)
		}
		return u16Bytes(u16)
	default:
		panic(fmt.Sprintf("unsupported tensor dtype %v", dt))
	}
}

func tensorElemCount(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

func TestFinalHeadVerifySampleEmulatedMegakernel(t *testing.T) {
	if _, err := cudahostemu.Compiler(); errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip(err)
	}

	const (
		mNodes = 4
		h      = 256
		vocab  = 32
		eps    = 1e-5
	)
	bPost := hopper.NewBuilder()
	xIn := bPost.Input("x", hopper.F32, mNodes, h)
	wNorm := bPost.Input("final_norm", hopper.F32, h)
	xnOut := bPost.Cast(hopper.BF16, bPost.RMSNorm(xIn, wNorm, h, eps))
	postExpr, err := hopper.Optimize(bPost.G, xnOut)
	if err != nil {
		t.Fatalf("optimize finalPost: %v", err)
	}
	finalPost, err := hopper.Compile(postExpr, "final_post", hopper.BF16)
	if err != nil {
		t.Fatalf("compile finalPost: %v", err)
	}
	if finalPost.Kernels[0].Row != nil {
		finalPost.Kernels[0].Row.Name = "final_rmsnorm"
	}

	bHead := hopper.NewBuilder()
	xnIn := bHead.Input("xn", hopper.BF16, mNodes, h)
	wHead := bHead.Input("w_head", hopper.BF16, vocab, h)
	logits := bHead.MatMul(xnIn, wHead)
	headExpr, err := hopper.Optimize(bHead.G, logits)
	if err != nil {
		t.Fatalf("optimize head: %v", err)
	}
	head, err := hopper.Compile(headExpr, "lm_head", hopper.F32)
	if err != nil {
		t.Fatalf("compile head: %v", err)
	}
	if head.Kernels[0].GEMM != nil {
		head.Kernels[0].GEMM.Name = "lm_head"
		head.Kernels[0].GEMM.RowsPerBlock = 32
	}

	mega, err := BuildFinalHeadVerifySampleMegakernel(finalPost, head, FusedVerifySampleConfig{
		Name:               "emu_final_head_verify_mega",
		DraftNodes:         mNodes,
		VocabSize:          vocab,
		DefaultTemperature: 0.9,
	})
	if err != nil {
		t.Fatalf("BuildFinalHeadVerifySampleMegakernel: %v", err)
	}

	src, _, err := mega.LineInfoSource()
	if err != nil {
		t.Fatalf("LineInfoSource: %v", err)
	}

	r := rand.New(rand.NewPCG(2026, 1031))
	var bufs []emuBuf

	// Allocate shared temporary buffers and persistent ready buffer (never reset across epochs).
	for _, tmp := range mega.BaseMegakernel.Temps {
		nbytes := tensorElemCount(tmp.Shape) * tmp.DType.Bits() / 8
		bufs = append(bufs, emuBuf{name: tmp.Name, ctype: tmp.DType.CType(), data: poison(nbytes)})
	}
	bufs = append(bufs, emuBuf{
		name:  "ready",
		ctype: "unsigned long long",
		data:  make([]byte, mega.ReadyBytes()),
	})

	type epochSpec struct {
		mask       []uint32
		parentIdx  []int32
		draftIDs   []int32
		draftProbs []float32
		temp       float32
		seed       uint64
		offset     uint64
	}

	allowMask := NewGrammarBitmask(vocab)
	for v := range vocab {
		if v%3 == 1 {
			allowMask.Allow(v)
		}
	}

	epochs := []epochSpec{
		{
			mask:       allowMask.Words,
			parentIdx:  []int32{-1, 0, 0, 1},
			draftIDs:   []int32{1, 2, 4, 7}, // 2 is disallowed (2%3 != 1), 1, 4, 7 are allowed
			draftProbs: []float32{0.01, 0.9, 0.01, 0.01},
			temp:       0.8,
			seed:       0xfeedface12345678,
			offset:     101,
		},
		{
			mask:       allowMask.Words,
			parentIdx:  []int32{-1, 0, 1, 2},
			draftIDs:   []int32{10, 13, 16, 19},
			draftProbs: []float32{0.2, 0.2, 0.2, 0.2},
			temp:       0.0, // greedy verification on epoch 2
			seed:       0xabcdef0123456789,
			offset:     202,
		},
	}

	var launches []string
	for epIdx, ep := range epochs {
		epoch := epIdx + 1
		var args []string
		for _, in := range mega.BaseMegakernel.Inputs {
			bname := fmt.Sprintf("%s_ep%d", in.Name, epoch)
			elems := tensorElemCount(in.Shape)
			vals := make([]float32, elems)
			for i := range vals {
				vals[i] = float32(r.NormFloat64() * 0.25)
			}
			bufs = append(bufs, emuBuf{
				name:  bname,
				ctype: in.DType.CType(),
				data:  encodePlanTensorData(in.DType, vals),
			})
			args = append(args, bname)
		}
		for _, tmp := range mega.BaseMegakernel.Temps {
			args = append(args, tmp.Name)
		}

		logitsName := fmt.Sprintf("logits_ep%d", epoch)
		maskName := fmt.Sprintf("mask_ep%d", epoch)
		parName := fmt.Sprintf("parent_ep%d", epoch)
		draftIDName := fmt.Sprintf("draft_ids_ep%d", epoch)
		draftProbName := fmt.Sprintf("draft_probs_ep%d", epoch)
		accTokName := fmt.Sprintf("accepted_tokens_ep%d", epoch)
		numAccName := fmt.Sprintf("num_accepted_ep%d", epoch)
		bonusName := fmt.Sprintf("next_bonus_ep%d", epoch)

		bufs = append(bufs,
			emuBuf{name: logitsName, ctype: "float", data: poison(4 * mNodes * vocab)},
			emuBuf{name: maskName, ctype: "uint32_t", data: u32Bytes(ep.mask)},
			emuBuf{name: parName, ctype: "int32_t", data: i32Bytes(ep.parentIdx)},
			emuBuf{name: draftIDName, ctype: "int32_t", data: i32Bytes(ep.draftIDs)},
			emuBuf{name: draftProbName, ctype: "float", data: f32Bytes(ep.draftProbs)},
			emuBuf{name: accTokName, ctype: "int32_t", data: poison(4 * mNodes)},
			emuBuf{name: numAccName, ctype: "int32_t", data: poison(4)},
			emuBuf{name: bonusName, ctype: "int32_t", data: poison(4)},
		)

		args = append(args,
			logitsName,
			maskName,
			parName,
			draftIDName,
			draftProbName,
			accTokName,
			numAccName,
			bonusName,
			floatLiteral(ep.temp),
			fmt.Sprintf("%#xULL", ep.seed),
			fmt.Sprintf("%#xULL", ep.offset),
			"ready",
			fmt.Sprintf("%du", epoch),
			"0",
		)
		launches = append(launches, fmt.Sprintf("%s_run(%s)", mega.Name, strings.Join(args, ", ")))
	}

	// Execute both epochs in a single program without resetting ready between launches.
	// Note: mega.LineInfoSource() already starts with hopper.Prelude, so pass src[len(hopper.Prelude):] to runEmu.
	trimmedSrc := strings.TrimPrefix(src, hopper.Prelude)
	res := runEmu(t, trimmedSrc, strings.Join(launches, ";\n    "), bufs...)

	for epIdx, ep := range epochs {
		epoch := epIdx + 1
		flatLogits := bytesF32(res[fmt.Sprintf("logits_ep%d", epoch)])
		logits2D := make([][]float32, mNodes)
		for m := range mNodes {
			row := flatLogits[m*vocab : (m+1)*vocab]
			for v, x := range row {
				if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
					t.Fatalf("epoch %d logits[%d][%d] = %v", epoch, m, v, x)
				}
			}
			logits2D[m] = row
		}

		want, err := EvalFinalHeadVerifySample(
			logits2D,
			ep.mask,
			ep.draftIDs,
			ep.draftProbs,
			ep.parentIdx,
			ep.temp,
			ep.seed,
			ep.offset,
		)
		if err != nil {
			t.Fatalf("epoch %d EvalFinalHeadVerifySample: %v", epoch, err)
		}

		gotNumAcc := bytesI32(res[fmt.Sprintf("num_accepted_ep%d", epoch)])[0]
		gotBonus := bytesI32(res[fmt.Sprintf("next_bonus_ep%d", epoch)])[0]
		gotAccAll := bytesI32(res[fmt.Sprintf("accepted_tokens_ep%d", epoch)])
		if gotNumAcc < 0 || int(gotNumAcc) > mNodes {
			t.Fatalf("epoch %d invalid num_accepted = %d", epoch, gotNumAcc)
		}
		gotAcc := gotAccAll[:gotNumAcc]

		if gotNumAcc != want.NumAccepted {
			t.Fatalf("epoch %d num_accepted = %d, want %d", epoch, gotNumAcc, want.NumAccepted)
		}
		if !slices.Equal(gotAcc, want.AcceptedTokens) {
			t.Fatalf("epoch %d accepted_tokens = %v, want %v", epoch, gotAcc, want.AcceptedTokens)
		}
		if gotBonus != want.NextBonusToken {
			t.Fatalf("epoch %d next_bonus_token = %d, want %d", epoch, gotBonus, want.NextBonusToken)
		}
		for _, tok := range gotAcc {
			if !allowMask.Allowed(int(tok)) {
				t.Fatalf("epoch %d accepted token %d disallowed by grammar bitmask", epoch, tok)
			}
		}
		if !allowMask.Allowed(int(gotBonus)) {
			t.Fatalf("epoch %d bonus token %d disallowed by grammar bitmask", epoch, gotBonus)
		}
	}

	// Verify ready counter words advanced to epoch 2 across all 6 producer stages (including head_lm_head).
	readyBytes := res["ready"]
	for i := 0; i < len(mega.Stages)-1; i++ {
		gotWord := binary.LittleEndian.Uint64(readyBytes[8*i : 8*(i+1)])
		wantWord := hopper.PackReadyWord(uint32(len(epochs)), uint32(mega.Stages[i].Blocks))
		if gotWord != wantWord {
			t.Fatalf("ready[%d] = %#x, want %#x", i, gotWord, wantWord)
		}
	}
}

func TestFinalHeadVerifySampleErrors(t *testing.T) {
	finalPost, head := buildTestFinalPostAndHeadPlans(t, 4, 256, 256, 256, 256, false)

	if _, err := BuildFinalHeadVerifySampleMegakernel(nil, head, FusedVerifySampleConfig{}); err == nil {
		t.Error("nil finalPost: expected error")
	}
	if _, err := BuildFinalHeadVerifySampleMegakernel(finalPost, nil, FusedVerifySampleConfig{}); err == nil {
		t.Error("nil head: expected error")
	}
	if _, err := BuildFinalHeadVerifySampleMegakernel(finalPost, head, FusedVerifySampleConfig{DraftNodes: 8}); err == nil {
		t.Error("mismatched DraftNodes: expected error")
	}
	if _, err := BuildFinalHeadVerifySampleMegakernel(finalPost, head, FusedVerifySampleConfig{VocabSize: 128}); err == nil {
		t.Error("mismatched VocabSize: expected error")
	}
	if _, err := BuildFinalHeadVerifySampleMegakernel(finalPost, head, FusedVerifySampleConfig{Name: "eg_bad"}); err == nil {
		t.Error("reserved name: expected error")
	}

	logits := [][]float32{make([]float32, 32), make([]float32, 32)}
	if _, err := EvalFinalHeadVerifySample(nil, nil, nil, nil, nil, 1.0, 0, 0); err == nil {
		t.Error("empty logits: expected error")
	}
	if _, err := EvalFinalHeadVerifySample(logits, []uint32{1, 2, 3}, []int32{0, 1}, []float32{1, 1}, []int32{-1, 0}, 1.0, 0, 0); err == nil {
		t.Error("bad vocabMask length: expected error")
	}
	if _, err := EvalFinalHeadVerifySample(logits, nil, []int32{0}, []float32{1, 1}, []int32{-1, 0}, 1.0, 0, 0); err == nil {
		t.Error("bad draftIDs length: expected error")
	}
	if _, err := EvalFinalHeadVerifySample(logits, nil, []int32{0, 1}, []float32{1, 1}, []int32{0, 0}, 1.0, 0, 0); err == nil {
		t.Error("bad parentIdx[0]: expected error")
	}
	if _, err := EvalFinalHeadVerifySample(logits, nil, []int32{0, 1}, []float32{1, 1}, []int32{-1, 2}, 1.0, 0, 0); err == nil {
		t.Error("out-of-range parentIdx[1]: expected error")
	}
	if _, err := EvalFinalHeadVerifySample(logits, nil, []int32{0, 1}, []float32{1, 1}, []int32{-1, 0}, -1.0, 0, 0); err == nil {
		t.Error("negative temperature: expected error")
	}
}
