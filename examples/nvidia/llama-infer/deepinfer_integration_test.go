package main

import (
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/tmc/lib/lm/speculative"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudagrammar"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudakv"
)

func TestSampleStepLogitsAndGrammarSpeculative(t *testing.T) {
	logits := []float32{-1.0, 3.5, 1.2, 4.0, 0.5, 2.8}

	// 1. Greedy argmax when temp == 0.
	gotGreedy, err := sampleStepLogits(logits, 0, sampleConfig{temp: 0})
	if err != nil {
		t.Fatalf("sampleStepLogits greedy: %v", err)
	}
	if gotGreedy != 3 {
		t.Fatalf("sampleStepLogits greedy = %d, want 3", gotGreedy)
	}

	// 2. Deterministic Philox-4x32-10 reproducibility and step variation.
	cfg := sampleConfig{
		seed: 123456789,
		temp: 0.8,
		topK: 3,
		topP: 0.9,
		minP: 0.05,
	}
	tokA, err := sampleStepLogits(logits, 4, cfg)
	if err != nil {
		t.Fatalf("sampleStepLogits seeded A: %v", err)
	}
	tokB, err := sampleStepLogits(logits, 4, cfg)
	if err != nil {
		t.Fatalf("sampleStepLogits seeded B: %v", err)
	}
	if tokA != tokB {
		t.Fatalf("sampleStepLogits not deterministic for identical (seed, step): %d != %d", tokA, tokB)
	}

	// Bitmask constraint masking out token 3 (only allowing token 1).
	mask := cudagrammar.NewBitmask(len(logits))
	mask.Set(1, true)
	tokMasked, err := sampleStepLogits(logits, 0, sampleConfig{
		seed:           42,
		temp:           0.7,
		allowedBitmask: mask.Words,
	})
	if err != nil {
		t.Fatalf("sampleStepLogits masked: %v", err)
	}
	if tokMasked != 1 {
		t.Fatalf("sampleStepLogits masked = %d, want 1", tokMasked)
	}

	// Must-fail controls for invalid inputs.
	if _, err := sampleStepLogits(logits, -1, cfg); err == nil {
		t.Fatal("sampleStepLogits with negative step succeeded, want error")
	}
	if _, err := sampleStepLogits(nil, 0, cfg); err == nil {
		t.Fatal("sampleStepLogits with empty logits succeeded, want error")
	}

	// 3. Grammar speculative masks and seeded speculative verification with prefix commit.
	const schema = `{"type":"object","properties":{"n":{"type":"integer"}}}`
	grammar, err := cudagrammar.CompileString(schema)
	if err != nil {
		t.Fatalf("CompileString: %v", err)
	}
	vocab := cudagrammar.NewVocabulary([]string{
		0: `{`,
		1: `"n"`,
		2: `:`,
		3: `1`,
		4: `}`,
	})
	matcher := grammar.Match()
	drafts := []int32{0, 1, 2}
	masks, err := speculativeGrammarMasks(matcher, vocab, drafts)
	if err != nil {
		t.Fatalf("speculativeGrammarMasks: %v", err)
	}
	if len(masks) != len(drafts)+1 {
		t.Fatalf("len(masks) = %d, want %d", len(masks), len(drafts)+1)
	}
	if matcher.Pos() != 0 {
		t.Fatalf("matcher.Pos() after speculativeGrammarMasks = %d, want 0 (rolled back)", matcher.Pos())
	}

	targetProbs := [][]float32{
		{1, 0, 0, 0, 0},
		{0, 1, 0, 0, 0},
		{0, 0, 1, 0, 0},
		{0, 0, 0, 1, 0},
	}
	draftProbs := [][]float32{
		{1, 0, 0, 0, 0},
		{0, 1, 0, 0, 0},
		{0, 0, 1, 0, 0},
	}
	accepted, bonus, err := verifySeededSpeculativeStep(targetProbs, draftProbs, drafts, 77, 0, matcher, vocab)
	if err != nil {
		t.Fatalf("verifySeededSpeculativeStep: %v", err)
	}
	if !slices.Equal(accepted, drafts) || bonus != 3 {
		t.Fatalf("verifySeededSpeculativeStep = (accepted=%v, bonus=%d), want (%v, 3)", accepted, bonus, drafts)
	}
	if matcher.Pos() == 0 {
		t.Fatal("matcher.Pos() did not advance after CommitPrefix")
	}
}

func TestDecodeGraphSyncArbiterAndPingPongTimer(t *testing.T) {
	saveFuse, saveDecode, saveQual := *fuseEGraphFlag, *fuseDecodeFlag, fusedKernelsQualified
	t.Cleanup(func() {
		*fuseEGraphFlag = saveFuse
		*fuseDecodeFlag = saveDecode
		fusedKernelsQualified = saveQual
	})
	fusedKernelsQualified = true
	*fuseEGraphFlag = false
	*fuseDecodeFlag = false

	// 1. Parse sync-guard modes.
	for _, tc := range []struct {
		in      string
		want    cudacapture.SyncMode
		wantErr bool
	}{
		{"", cudacapture.SyncAllow, false},
		{"allow", cudacapture.SyncAllow, false},
		{"WARN", cudacapture.SyncWarn, false},
		{"reject", cudacapture.SyncReject, false},
		{"invalid", cudacapture.SyncAllow, true},
	} {
		got, err := parseSyncGuardMode(tc.in)
		if (err != nil) != tc.wantErr || got != tc.want {
			t.Fatalf("parseSyncGuardMode(%q) = (%v, %v), want (%v, err=%v)", tc.in, got, err, tc.want, tc.wantErr)
		}
	}

	gpuW, cfg, sp, kvs, gm := fakeStepFixture(t, 2, weightF32)
	layerKV := func(i int) (cudakv.LayerCache, error) { return kvs[i], nil }
	h := newFakeGraphHarness()
	dg, err := h.build(gpuW, cfg, layerKV, sp, gm, 100, 101)
	if err != nil {
		t.Fatalf("build decode graph: %v", err)
	}

	// 2. SyncReject blocks synchronous Step(pos, nil) and Step(pos, nonNilStream) before launch.
	arbiter := newDecodeSyncArbiter(cudacapture.SyncReject, nil)
	dg.SetSyncArbiter(arbiter)

	if _, err := dg.Step(1, nil); !errors.Is(err, cudacapture.ErrForbiddenSync) {
		t.Fatalf("dg.Step(1, nil) under SyncReject = %v, want ErrForbiddenSync", err)
	}
	if len(h.launched) != 0 {
		t.Fatalf("dg.Step under SyncReject launched %d times, want 0", len(h.launched))
	}
	fakeStream := new(cuda.Stream)
	if _, err := dg.Step(1, fakeStream); !errors.Is(err, cudacapture.ErrForbiddenSync) {
		t.Fatalf("dg.Step(1, stream) under SyncReject = %v, want ErrForbiddenSync", err)
	}

	// Allow scope temporarily permits synchronous Step.
	restoreAllow := arbiter.Allow()
	tok, err := dg.Step(1, nil)
	restoreAllow()
	if err != nil || tok != h.hostOutToken {
		t.Fatalf("dg.Step inside Allow() = (%d, %v), want (%d, nil)", tok, err, h.hostOutToken)
	}
	dg.SetSyncArbiter(nil)

	// 3. PingPongTimer simulated replay integration on DecodeGraph.
	timer, err := cudagraph.NewPingPongTimer(cudagraph.PingPongTimerConfig{
		NumEventNodes: 2,
		HistoryCap:    4,
	})
	if err != nil {
		t.Fatalf("NewPingPongTimer: %v", err)
	}
	nodes := []cudagraph.EventRecordNode{
		cudagraph.NewEventRecordNode(11),
		cudagraph.NewEventRecordNode(12),
	}
	if err := dg.AttachPingPongTimer(timer, 77, nodes); err != nil {
		t.Fatalf("AttachPingPongTimer: %v", err)
	}
	if err := timer.RecordSimulatedReplay([]float64{1.25}); err != nil {
		t.Fatalf("RecordSimulatedReplay #1: %v", err)
	}
	if err := timer.RecordSimulatedReplay([]float64{1.50}); err != nil {
		t.Fatalf("RecordSimulatedReplay #2: %v", err)
	}
	if got := dg.LastReplayDurationsMS(); !slices.Equal(got, []float64{1.25}) {
		t.Fatalf("LastReplayDurationsMS = %v, want [1.25]", got)
	}

	// 4. BucketRunner integration.
	runner, err := newDecodeBucketRunner(16, dg)
	if err != nil {
		t.Fatalf("newDecodeBucketRunner: %v", err)
	}
	gotDG, pad, err := selectDecodeBucket(runner, 3)
	if err != nil {
		t.Fatalf("selectDecodeBucket(3): %v", err)
	}
	if gotDG != dg || pad.BucketBsz != 4 || pad.PadCount != 1 {
		t.Fatalf("selectDecodeBucket(3) = (%p, %+v), want (%p, BucketBsz=4, PadCount=1)", gotDG, pad, dg)
	}
	if _, _, err := selectDecodeBucket(runner, 17); err == nil {
		t.Fatal("selectDecodeBucket(17) succeeded above MaxBsz=16, want error")
	}

	// Clean up fake graph.
	if err := dg.close(h.destroyExec, h.destroyGraph); err != nil {
		t.Fatalf("dg.close: %v", err)
	}
}

func TestSpecVarlenSchedulerAndZOSTransition(t *testing.T) {
	sched, err := newSpecVarlenScheduler(4)
	if err != nil {
		t.Fatalf("newSpecVarlenScheduler: %v", err)
	}
	budget, err := scheduleDraftBudget(sched, 0, 6, 4)
	if err != nil {
		t.Fatalf("scheduleDraftBudget: %v", err)
	}
	if budget < 1 || budget > 4 {
		t.Fatalf("scheduleDraftBudget = %d, want in [1, 4]", budget)
	}

	// Run decodeState with VarlenScheduler attached and verify greedy parity.
	prompt := []int32{0, 1, 2, 3, 0, 1, 2}
	const maxSeq, steps, draftN = 32, 10, 3
	wantTokens, wantNext, wantPos, err := plainGreedy(cycleModel(maxSeq, 4), prompt, maxSeq, steps)
	if err != nil {
		t.Fatalf("plainGreedy: %v", err)
	}

	drafter := speculative.NewNgramDrafter(2)
	st := newDecodeState(prompt, steps, drafter)
	st.setScheduler(sched)
	model := cycleModel(maxSeq, 4)
	var next int32
	for pos, tok := range prompt {
		next, err = model(tok, pos)
		if err != nil {
			t.Fatalf("prefill %d: %v", pos, err)
		}
	}
	plain := func(step int, tok int32, pos int) (int32, error) {
		return model(tok, pos)
	}
	noopShow := func(int32) error { return nil }
	gotNext, gotPos, err := decode(st, model, plain, noopShow, next, len(prompt), maxSeq, steps, draftN)
	if err != nil {
		t.Fatalf("decode with scheduler: %v", err)
	}
	if !slices.Equal(st.generated, wantTokens) || gotNext != wantNext || gotPos != wantPos {
		t.Fatalf("decode with scheduler = (tokens=%v, next=%d, pos=%d), want (%v, %d, %d)",
			st.generated, gotNext, gotPos, wantTokens, wantNext, wantPos)
	}
	if ewma, ok := sched.EWMAProb(0); !ok || ewma <= 0 {
		t.Fatalf("sched.EWMAProb(0) = (%v, %v), want tracked > 0", ewma, ok)
	}

	// ZOS transition planning.
	prevTokens := []int32{10, 11, 12, 13}
	nextDrafts := []int32{21, 22}
	plan, nextIDs, nextK, err := planZOSDecodeTransition(prevTokens, 3, nextDrafts, 5)
	if err != nil {
		t.Fatalf("planZOSDecodeTransition: %v", err)
	}
	if plan.Remap[0] != 2 || nextK != 8 || !slices.Equal(nextIDs, []int32{12, 21, 22}) {
		t.Fatalf("planZOSDecodeTransition = (remap=%v, nextIDs=%v, nextK=%d), want (remap=[2], nextIDs=[12 21 22], nextK=8)",
			plan.Remap, nextIDs, nextK)
	}

	// Confidence-ranked seqlens_q.
	seqlensQ, cuSeqlensQ, err := rankDraftsByConfidence(
		[][]float32{
			{0.9, 0.8, 0.2},
			{0.4, 0.9, 0.9},
		},
		nil,
		4,
		3,
	)
	if err != nil {
		t.Fatalf("rankDraftsByConfidence: %v", err)
	}
	if !slices.Equal(seqlensQ, []int32{3, 1}) || !slices.Equal(cuSeqlensQ, []int32{0, 3, 4}) {
		t.Fatalf("rankDraftsByConfidence = (%v, %v), want ([3 1], [0 3 4])", seqlensQ, cuSeqlensQ)
	}
}

func TestDeepInferQuantAndKVTopology(t *testing.T) {
	// 1. Block-scaled FP8 UE8M0 weight quantization + TMA column-major scale round-trip.
	const rows, cols, blockSize = 4, 128, 128
	host := make([]float32, rows*cols)
	for i := range host {
		host[i] = float32((i%17)-8) * 0.25
	}
	q, colMajorScales, alignedRows, err := quantizeWeightFP8UE8M0(host, rows, cols, blockSize)
	if err != nil {
		t.Fatalf("quantizeWeightFP8UE8M0: %v", err)
	}
	if alignedRows != 16 {
		t.Fatalf("alignedRows = %d, want 16", alignedRows)
	}
	deq, err := dequantizeWeightFP8UE8M0(q, colMajorScales, rows, cols, blockSize, alignedRows)
	if err != nil {
		t.Fatalf("dequantizeWeightFP8UE8M0: %v", err)
	}
	for i := range host {
		if math.Abs(float64(deq[i]-host[i])) > 0.15 {
			t.Fatalf("deq[%d] = %v, want ~%v", i, deq[i], host[i])
		}
	}

	// 2. Fused SwiGLU per-token FP8 quantization.
	gateUp := make([]float32, 2*2*blockSize)
	for i := range gateUp {
		gateUp[i] = 0.5
	}
	fp8Out, scales, packedColMajor, err := quantizeLayerSwiGLUFP8(gateUp, 2, blockSize, blockSize)
	if err != nil {
		t.Fatalf("quantizeLayerSwiGLUFP8: %v", err)
	}
	if len(fp8Out) == 0 || len(scales) != 2 || len(packedColMajor) == 0 {
		t.Fatalf("quantizeLayerSwiGLUFP8 lengths = (%d, %d, %d)", len(fp8Out), len(scales), len(packedColMajor))
	}

	// 3. YoCo shared KV topology + SWA bounded replay.
	topo, err := buildModelYOCOTopology(16, 8, 128)
	if err != nil {
		t.Fatalf("buildModelYOCOTopology: %v", err)
	}
	if got := topo.MemorySavingsRatio(); math.Abs(got-0.5) > 1e-6 {
		t.Fatalf("MemorySavingsRatio = %v, want 0.5", got)
	}
	if !topo.IsConsumer(12) || topo.IsConsumer(3) {
		t.Fatalf("unexpected producer/consumer classification in YoCo topology")
	}
	slot3, err := topo.PhysicalSlotForLayer(3)
	if err != nil {
		t.Fatalf("PhysicalSlotForLayer(3): %v", err)
	}
	slot11, err := topo.PhysicalSlotForLayer(11)
	if err != nil {
		t.Fatalf("PhysicalSlotForLayer(11): %v", err)
	}
	if slot3 != slot11 {
		t.Fatalf("consumer layer 11 slot %d != producer layer 3 slot %d", slot11, slot3)
	}

	win, err := planLayerSWAReplay(300, 1, 128, 6, 64)
	if err != nil {
		t.Fatalf("planLayerSWAReplay: %v", err)
	}
	if win.ActiveTokens != 134 || win.StartPos != 166 || win.EndPos != 300 || win.RequiredPages != 3 {
		t.Fatalf("planLayerSWAReplay = %+v, want ActiveTokens=134 StartPos=166 EndPos=300 RequiredPages=3", win)
	}
}
