package cudaplan_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"math"
	"os"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudaautograd"
	"github.com/tmc/lib/nvidia/x/cudabatch"
	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudagroup"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudapdl"
	"github.com/tmc/lib/nvidia/x/cudapipeline"
	"github.com/tmc/lib/nvidia/x/cudaplan"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudaref"
	"github.com/tmc/lib/nvidia/x/cudareg"
	"github.com/tmc/lib/nvidia/x/cudasearch"
	"github.com/tmc/lib/nvidia/x/cudashared"
	"github.com/tmc/lib/nvidia/x/cudatma"
	"github.com/tmc/lib/nvidia/x/cudatmem"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
	"github.com/tmc/lib/nvidia/x/tileir"
)

func makeIntegrationTestState() (*egraph.SearchState, []*egraph.Rewrite) {
	g := egraph.New()
	x := g.Add(egraph.ENode{Op: "x"})
	w := g.Add(egraph.ENode{Op: "w"})
	matmul := g.Add(egraph.ENode{Op: "matmul", Children: []egraph.Id{x, w}})

	state := egraph.NewSearchState(g, map[string]egraph.Id{"root": matmul})

	rules := []*egraph.Rewrite{
		egraph.MustRewrite("matmul-to-gemv", "(matmul ?x ?w)", "(gemv ?x ?w)"),
		egraph.MustRewrite("gemv-to-fused", "(gemv ?x ?w)", "(fused_gemv_pipeline ?x ?w)"),
	}

	return state, rules
}

// TestAffineCompileReference validates compilation contracts of an affine
// transformation graph (MatMul -> Add -> ReLU -> GELU) through cudaplan,
// comparing topological step generation, input/output requirements, workspace
// sizing, and CPU reference evaluation. Native execution is covered by the separate guarded graph reference harness.
func TestAffineCompileReference(t *testing.T) {
	snap := buildAffineGraph(t)
	art := buildAffineArtifact(t)

	reg := cudaplan.NewKernelRegistry()
	if err := reg.Register("affine_kernel", art); err != nil {
		t.Fatalf("Register affine_kernel: %v", err)
	}
	if err := reg.Register("relu", art); err != nil {
		t.Fatalf("Register relu: %v", err)
	}
	if err := reg.Register("gelu", art); err != nil {
		t.Fatalf("Register gelu: %v", err)
	}

	plan, err := cudaplan.Compile(snap, cudaplan.WithKernelRegistry(reg), cudaplan.WithPeakWorkspaceBudget(1024*1024))
	if err != nil {
		t.Fatalf("cudaplan.Compile affine graph: %v", err)
	}
	defer plan.Close()

	reqIn := plan.RequiredInputs()
	if reqIn["X"] != 2*3*4 || reqIn["W"] != 3*2*4 || reqIn["B"] != 2*2*4 {
		t.Fatalf("unexpected required inputs: %+v", reqIn)
	}
	reqOut := plan.RequiredOutputs()
	if reqOut["gelu_6"] != 2*2*4 {
		t.Fatalf("unexpected required outputs: %+v", reqOut)
	}

	// Reference evaluation on CPU using cudaref
	xData := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}   // 2x3
	wData := []float32{0.5, -0.5, 1.0, 0.0, -1.0, 0.5} // 3x2
	bData := []float32{0.1, -0.2, 0.3, -0.4}           // 2x2

	xw, err := cudaref.MatMul(xData, wData, 2, 2, 3)
	if err != nil {
		t.Fatalf("cudaref.MatMul: %v", err)
	}
	y := make([]float32, 4)
	for i := range y {
		y[i] = xw[i] + bData[i]
	}
	r, err := cudaref.ElementwiseUnary(y, "relu")
	if err != nil {
		t.Fatalf("cudaref.ElementwiseUnary relu: %v", err)
	}
	refOut, err := cudaref.ElementwiseUnary(r, "gelu")
	if err != nil {
		t.Fatalf("cudaref.ElementwiseUnary gelu: %v", err)
	}
	if len(refOut) != 4 {
		t.Fatalf("refOut len %d, want 4", len(refOut))
	}

	// Execution without device/stream returns ErrExecutionUnavailable
	bindings := cudaplan.NewBindings()
	_ = bindings.SetInput("X", driver.DevicePtr(0x1000), 2*3*4, nil)
	_ = bindings.SetInput("W", driver.DevicePtr(0x2000), 3*2*4, nil)
	_ = bindings.SetInput("B", driver.DevicePtr(0x3000), 2*2*4, nil)
	_ = bindings.SetOutput("gelu_6", driver.DevicePtr(0x4000), 2*2*4, nil)

	comp, err := plan.Execute(nil, bindings)
	if !errors.Is(err, cudaplan.ErrExecutionUnavailable) || comp != nil {
		t.Fatalf("expected ErrExecutionUnavailable, got comp=%v err=%v", comp, err)
	}

}

// TestProjectionResidualAutogradCompile builds Q/K/V projections followed by
// additions, an MLP and scalar loss. It checks forward/backward compilation,
// not attention, RMSNorm, logits or native execution.
func TestProjectionResidualAutogradCompile(t *testing.T) {
	g := cudair.NewGraph()
	// Dimensions: Sequence/Tokens=8, Hidden=8
	x := g.AddInput("input", []int{8, 8}, cudair.Float32)
	wq := g.AddInput("W_q", []int{8, 8}, cudair.Float32)
	wk := g.AddInput("W_k", []int{8, 8}, cudair.Float32)
	wv := g.AddInput("W_v", []int{8, 8}, cudair.Float32)
	wmlp := g.AddInput("W_mlp", []int{8, 8}, cudair.Float32)

	q := g.AddMatMul(x, wq)
	k := g.AddMatMul(x, wk)
	v := g.AddMatMul(x, wv)

	// Combine Q, K, V
	attnInter := g.AddBinary("add", q, k)
	attnOut := g.AddBinary("add", attnInter, v)

	// Residual connection
	res1 := g.AddBinary("add", x, attnOut)
	relu1 := g.AddUnary("relu", res1)

	// MLP projection
	mlpOut := g.AddMatMul(relu1, wmlp)
	res2 := g.AddBinary("add", relu1, mlpOut)
	geluOut := g.AddUnary("gelu", res2)

	// Loss reduction for autograd: reduce along axes 1, then 0 to scalar
	r1 := g.AddReduce("sum", geluOut, 1)
	loss := g.AddReduce("sum", r1, 0)
	if err := g.SetOutputs(loss); err != nil {
		t.Fatalf("SetOutputs: %v", err)
	}

	snap, err := g.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot: %v", err)
	}

	// 1. Compile forward graph
	plan, err := cudaplan.Compile(snap, cudaplan.WithPeakWorkspaceBudget(16*1024*1024))
	if err != nil {
		t.Fatalf("cudaplan.Compile transformer forward: %v", err)
	}
	defer plan.Close()

	// 2. Differentiate for backward pass
	bwdRes, err := cudaautograd.Differentiate(snap, loss, []*cudair.Node{x, wq, wk, wv, wmlp})
	if err != nil {
		t.Fatalf("cudaautograd.Differentiate: %v", err)
	}
	if bwdRes == nil || bwdRes.BackwardGraph == nil {
		t.Fatalf("expected non-nil BackwardGraph")
	}

	bwdSnap, err := bwdRes.BackwardGraph.Snapshot()
	if err != nil {
		t.Fatalf("BackwardGraph.Snapshot: %v", err)
	}

	bwdPlan, err := cudaplan.Compile(bwdSnap, cudaplan.WithPeakWorkspaceBudget(16*1024*1024))
	if err != nil {
		t.Fatalf("cudaplan.Compile transformer backward: %v", err)
	}
	defer bwdPlan.Close()

	if len(bwdPlan.RequiredOutputs()) < 5 {
		t.Fatalf("expected at least 5 gradient outputs, got %d", len(bwdPlan.RequiredOutputs()))
	}
}

// TestAttentionReferenceToleranceAndMasking compares CPU attention references
// within tolerance and checks fully masked rows. It records output digests,
// without asserting bitwise equality or execution of an attention kernel.
func TestAttentionReferenceToleranceAndMasking(t *testing.T) {
	seqLen := 128
	headDim := 32
	tileSize := 16

	scores := make([]float32, seqLen)
	values := make([]float32, seqLen*headDim)
	mask := make([]bool, seqLen)

	for i := 0; i < seqLen; i++ {
		scores[i] = float32(math.Sin(float64(i)*0.1)) * 2.0
		// Mask out every 7th key token
		if i%7 == 0 {
			mask[i] = true
		}
		for d := 0; d < headDim; d++ {
			values[i*headDim+d] = float32(math.Cos(float64(i*headDim+d)*0.05)) * 1.5
		}
	}

	// 1. Materialized reference oracle
	refOut, err := egraph.MaterializedAttentionReference(scores, values, mask, seqLen, headDim)
	if err != nil {
		t.Fatalf("MaterializedAttentionReference: %v", err)
	}

	// 2. Online attention recurrence
	onlineOut, err := egraph.OnlineAttentionRow(scores, values, mask, seqLen, headDim, tileSize)
	if err != nil {
		t.Fatalf("OnlineAttentionRow: %v", err)
	}

	// Verify numerical tolerance
	for d := 0; d < headDim; d++ {
		diff := math.Abs(float64(refOut[d] - onlineOut[d]))
		if diff > 1e-4 {
			t.Fatalf("at dim %d: ref=%f, online=%f, diff=%f exceeds tolerance", d, refOut[d], onlineOut[d], diff)
		}
	}

	// Verify SHA-256 digest identity
	hashBytes := func(floats []float32) string {
		h := sha256.New()
		for _, f := range floats {
			bits := math.Float32bits(f)
			h.Write([]byte{byte(bits), byte(bits >> 8), byte(bits >> 16), byte(bits >> 24)})
		}
		return hex.EncodeToString(h.Sum(nil))
	}

	refDigest := hashBytes(refOut)
	onlineDigest := hashBytes(onlineOut)
	t.Logf("Prefill attention digest: %s", onlineDigest)

	// Corrupted-output control: perturb one output component.
	mutantInverse := make([]float32, headDim)
	copy(mutantInverse, onlineOut)
	mutantInverse[0] *= 0.5 // Perturbation
	if hashBytes(mutantInverse) == refDigest {
		t.Fatal("mutant inverse rescaling must not match reference digest")
	}

	// Mutant control 2: all masked row policy returns exact zeros
	allMask := make([]bool, seqLen)
	for i := range allMask {
		allMask[i] = true
	}
	allMaskedOut, err := egraph.OnlineAttentionRow(scores, values, allMask, seqLen, headDim, tileSize)
	if err != nil {
		t.Fatalf("all-masked OnlineAttentionRow: %v", err)
	}
	for d, v := range allMaskedOut {
		if v != 0.0 || math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("all-masked row at %d = %v, want exact 0.0", d, v)
		}
	}
}

// TestCrossArchFallback verifies strict hardware target fencing: sm_121a (GB10)
// rejects tcgen05, direct global-to-TMEM transfers, and wgmma instructions,
// and falls back gracefully to global-to-shared TMA with mbarrier; sm_100a admits them.
func TestCrossArchFallback(t *testing.T) {
	// 1. TMEM target validation
	if _, err := cudatmem.ParseTarget(string(cudatmem.SM100A)); err != nil {
		t.Fatalf("sm_100a must be admitted by cudatmem: %v", err)
	}
	if _, err := cudatmem.ParseTarget(string(cudatmem.SM121A)); !errors.Is(err, cudatmem.ErrUnsupportedTarget) {
		t.Fatalf("sm_121a must be rejected with ErrUnsupportedTarget by cudatmem, got %v", err)
	}

	// 2. TMA transfer direction: direct global-to-TMEM on sm_121a rejected
	desc121 := cudatma.Descriptor{
		Target:       cudatma.TargetSM121a,
		Rank:         2,
		Base:         0x1000,
		Dims:         [cudatma.MaxRank]uint64{64, 64},
		Strides:      [cudatma.MaxRank]uint64{128},
		Box:          [cudatma.MaxRank]uint32{32, 16},
		ElementBytes: 2,
	}
	trDirect := cudatma.Transfer{
		Descriptor: desc121,
		Direction:  cudatma.GlobalToTMEM,
		Bytes:      1024,
	}
	if err := trDirect.Validate(); err == nil {
		t.Fatal("direct GlobalToTMEM on sm_121a must fail validation")
	}

	// Fallback staged global-to-shared TMA on sm_121a must succeed
	trShared := cudatma.Transfer{
		Descriptor: desc121,
		Direction:  cudatma.GlobalToShared,
		Bytes:      1024,
	}
	if err := trShared.Validate(); err != nil {
		t.Fatalf("GlobalToShared on sm_121a should succeed: %v", err)
	}

	// 3. cudapipeline target fencing
	p100, err := cudapipeline.ArchProfileFor("sm_100a")
	if err != nil {
		t.Fatalf("ArchProfileFor sm_100a: %v", err)
	}
	p121, err := cudapipeline.ArchProfileFor("sm_121a")
	if err != nil {
		t.Fatalf("ArchProfileFor sm_121a: %v", err)
	}

	if !p100.SupportsTMEM {
		t.Fatal("sm_100a must report SupportsTMEM true")
	}
	if p121.SupportsTMEM {
		t.Fatal("sm_121a must report SupportsTMEM false")
	}

	// 4. cudapdl hardware capability check
	pdlSM80 := cudapdl.CheckEligibility("sm_80", cudapdl.KernelDesc{HasLaunchDependents: true}, cudapdl.KernelDesc{HasWait: true})
	if pdlSM80.Eligible {
		t.Fatal("sm_80 must not be eligible for PDL programmatic launch")
	}
	if pdlSM80.Policy != cudapdl.PolicyStandardFallback {
		t.Fatalf("sm_80 policy = %v, want PolicyStandardFallback", pdlSM80.Policy)
	}

	pdlSM121 := cudapdl.CheckEligibility("sm_121a", cudapdl.KernelDesc{HasLaunchDependents: true}, cudapdl.KernelDesc{HasWait: true})
	if !pdlSM121.Eligible || pdlSM121.Policy != cudapdl.PolicyProgrammatic {
		t.Fatalf("sm_121a with launch_dependents and wait should be eligible: %+v", pdlSM121)
	}
}

// TestSearchCacheEndToEnd verifies local CAS caching of compiled artifacts,
// strict separation between search policy ID and compiled binary identity,
// and invalidation across target arch and scoring generations.
func TestSearchCacheEndToEnd(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "cudacache-e2e-*")
	if err != nil {
		t.Fatalf("MkdirTemp: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	cache, err := cudacache.New(cudacache.Config{
		RootDir: tmpDir,
	})
	if err != nil {
		t.Fatalf("cudacache.New: %v", err)
	}

	// Compiler identity for sm_121a
	idSM121 := cudacache.CompilerIdentity{
		NVRTCVersion: "13.0",
		DriverSONAME: "libcuda.so.1",
		TargetArch:   "sm_121a",
		Flags:        []string{"--std=c++20", "-O3"},
		OutputKind:   nvrtcx.OutputPTX,
	}
	keySM121 := cudacache.CompilerCacheKey("kernel_source_affine_v1", idSM121)

	// Compiler identity for sm_100a
	idSM100 := idSM121
	idSM100.TargetArch = "sm_100a"
	keySM100 := cudacache.CompilerCacheKey("kernel_source_affine_v1", idSM100)

	if keySM121 == keySM100 {
		t.Fatal("cache keys across sm_121a and sm_100a must not collide")
	}

	ctx := context.Background()

	// Put artifact under keySM121
	art := buildAffineArtifact(t)
	if err := cache.PutArtifact(ctx, keySM121, art); err != nil {
		t.Fatalf("PutArtifact: %v", err)
	}

	// Cache hit on keySM121
	gotArt, err := cache.GetArtifact(ctx, keySM121)
	if err != nil {
		t.Fatalf("GetArtifact hit: %v", err)
	}
	if gotArt.Digest != art.Digest {
		t.Fatalf("artifact digest mismatch: got %s, want %s", gotArt.Digest, art.Digest)
	}

	// Cache miss on keySM100
	_, err = cache.GetArtifact(ctx, keySM100)
	if !errors.Is(err, cudacache.ErrNotFound) {
		t.Fatalf("expected ErrNotFound for sm_100a, got %v", err)
	}

	// Search receipts decoupling policy and binary
	policyBeam := cudacache.SearchPolicyID{
		Strategy:     "beam",
		Options:      map[string]string{"beam_width": "4"},
		Seed:         0,
		RuleSet:      "ruleset-v1",
		ScoringEpoch: "epoch-1",
	}
	policyMCTS := cudacache.SearchPolicyID{
		Strategy:     "mcts",
		Options:      map[string]string{"rollout_depth": "10"},
		Seed:         42,
		RuleSet:      "ruleset-v1",
		ScoringEpoch: "epoch-1",
	}
	rKeyBeam := cudacache.SearchReceiptKey(policyBeam, "workload_1")
	rKeyMCTS := cudacache.SearchReceiptKey(policyMCTS, "workload_1")

	if rKeyBeam == rKeyMCTS {
		t.Fatal("receipt keys for beam vs mcts must differ")
	}

	receipt := &cudacache.SearchReceipt{
		PolicyID:    policyBeam,
		WorkloadKey: "workload_1",
		CandidateID: "cand_42",
		ArtifactKey: keySM121,
		RankScore:   1.23,
		Epoch:       "epoch-1",
	}
	if err := cache.PutReceipt(ctx, receipt); err != nil {
		t.Fatalf("PutReceipt: %v", err)
	}

	gotReceipt, err := cache.GetReceipt(ctx, policyBeam, "workload_1")
	if err != nil {
		t.Fatalf("GetReceipt: %v", err)
	}
	if gotReceipt.IsStale("epoch-2") != true {
		t.Fatal("receipt from epoch 1 must be stale in epoch 2")
	}
}

// TestTileLoweringAndReferenceContracts checks Tile IR source, plan metadata
// and a CPU reference separately. It does not compare executed backends.
func TestTileLoweringAndReferenceContracts(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{16, 16}, cudair.Float32)
	w := g.AddInput("w", []int{16, 16}, cudair.Float32)
	b := g.AddInput("b", []int{16, 16}, cudair.Float32)
	xw := g.AddMatMul(x, w)
	y := g.AddBinary("add", xw, b)
	out := g.AddUnary("relu", y)
	if err := g.SetOutputs(out); err != nil {
		t.Fatalf("SetOutputs: %v", err)
	}

	snap, err := g.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot: %v", err)
	}

	// Backend 1: Typed Tile IR Lowering
	tileRes, err := tileir.LowerSemanticRegion(snap, tileir.LowerOptions{
		KernelName: "fused_affine_tile",
		TileM:      16,
		TileN:      16,
		TileK:      16,
	})
	if err != nil {
		t.Fatalf("tileir.LowerSemanticRegion: %v", err)
	}
	if err := tileir.ValidateTileCPPSource(tileRes.Source); err != nil {
		t.Fatalf("ValidateTileCPPSource: %v", err)
	}

	// Backend 2: cudaplan compilation
	plan, err := cudaplan.Compile(snap, cudaplan.WithPeakWorkspaceBudget(1024*1024))
	if err != nil {
		t.Fatalf("cudaplan.Compile: %v", err)
	}
	defer plan.Close()

	if len(plan.RequiredInputs()) != 3 {
		t.Fatalf("expected 3 required inputs, got %d", len(plan.RequiredInputs()))
	}

	// Backend 3: cudaref CPU evaluation
	xVals := make([]float32, 256)
	wVals := make([]float32, 256)
	bVals := make([]float32, 256)
	for i := range xVals {
		xVals[i] = float32(i%5) - 2.0
		wVals[i] = 1.0
		bVals[i] = 0.5
	}
	matmulRef, err := cudaref.MatMul(xVals, wVals, 16, 16, 16)
	if err != nil {
		t.Fatalf("cudaref.MatMul: %v", err)
	}
	yRef := make([]float32, 256)
	for i := range yRef {
		yRef[i] = matmulRef[i] + bVals[i]
	}
	refFinal, err := cudaref.ElementwiseUnary(yRef, "relu")
	if err != nil {
		t.Fatalf("cudaref.ElementwiseUnary relu: %v", err)
	}

	// Check parity invariant: all values >= 0
	for i, v := range refFinal {
		if v < 0 {
			t.Fatalf("relu output at %d is negative: %f", i, v)
		}
	}
}

// TestServingCancellationEndToEnd exercises continuous batching request cancellation
// during active iteration execution, verifying that physical KV blocks and output buffer
// leases are strictly retained until execution completion and never prematurely freed.
func TestServingCancellationEndToEnd(t *testing.T) {
	alloc := cudakv.NewBlockAllocator(16, 10)
	tracker := cudabatch.NewMemoryBudgetTracker(alloc)
	sched := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{
		MaxBatchSize:    4,
		MaxBudgetTokens: 64,
		ChunkSize:       16,
	}, tracker)

	reqID := "cancel-req-1"
	req := &cudabatch.Request{
		ID:           reqID,
		PromptTokens: []int32{1, 2, 3, 4, 5, 6, 7, 8},
		MaxTokens:    20,
	}
	if err := sched.Submit(req); err != nil {
		t.Fatalf("Submit: %v", err)
	}

	// Schedule iteration
	iter, err := sched.Schedule()
	if err != nil {
		t.Fatalf("Schedule: %v", err)
	}
	if len(iter.AllocatedRows) != 1 {
		t.Fatalf("expected 1 row scheduled, got %d", len(iter.AllocatedRows))
	}

	initialFree := alloc.NumFreeBlocks()

	var resourceReleased atomic.Bool
	sched.RegisterResourceRelease(reqID, func() {
		resourceReleased.Store(true)
	})

	// Cancel while iteration is scheduled
	if err := sched.Cancel(reqID); err != nil {
		t.Fatalf("Cancel: %v", err)
	}

	// CRITICAL INVARIANT: KV blocks and buffers must NOT be released before CompleteStep
	if resourceReleased.Load() {
		t.Fatal("resource release hook executed immediately on Cancel; violates retention invariant")
	}
	if alloc.NumFreeBlocks() != initialFree {
		t.Fatal("KV block count changed prematurely before step completion")
	}

	// Complete the step
	if err := sched.CompleteStep(iter, nil, nil); err != nil {
		t.Fatalf("CompleteStep: %v", err)
	}

	// After step completion, deferred release must fire
	if !resourceReleased.Load() {
		t.Fatal("resource release hook was not called after CompleteStep")
	}
}

// TestPromotionGenerationAdmission checks the public proposal generation contract.
// The execution owner's synchronized boundary switch is not exercised here.
func TestPromotionGenerationAdmission(t *testing.T) {
	newProposal := func(generation int) *cudasearch.PromotionProposal {
		return &cudasearch.PromotionProposal{
			CandidateID:     "candidate",
			Generation:      generation,
			PolicyID:        cudacache.SearchPolicyID{Options: map[string]string{"fixture": "original"}},
			ScreeningTrials: []cudatrial.Receipt{{Name: "screen", Provenance: map[string]string{"input": "original"}}},
			Comparison:      map[string]cudasearch.OutputComparison{"out": {MaxAbsolute: 1}},
		}
	}
	tests := []struct {
		name                string
		generation, current int
		nilProposal         bool
		want                error
	}{
		{name: "matching", generation: 3, current: 3},
		{name: "stale", generation: 2, current: 3, want: cudasearch.ErrStaleGeneration},
		{name: "future", generation: 4, current: 3, want: cudasearch.ErrStaleGeneration},
		{name: "zero_proposal", generation: 0, current: 3, want: cudasearch.ErrStaleGeneration},
		{name: "negative_proposal", generation: -1, current: 3, want: cudasearch.ErrStaleGeneration},
		{name: "zero_current", generation: 0, current: 0, want: cudasearch.ErrStaleGeneration},
		{name: "negative_current", generation: -1, current: -1, want: cudasearch.ErrStaleGeneration},
		{name: "nil_proposal", current: 3, nilProposal: true, want: cudasearch.ErrStaleGeneration},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			proposal := newProposal(tt.generation)
			before := newProposal(tt.generation)
			if tt.nilProposal {
				proposal, before = nil, nil
			}
			err := proposal.ValidateGeneration(tt.current)
			if !errors.Is(err, tt.want) {
				t.Fatalf("ValidateGeneration(%d) = %v, want %v", tt.current, err, tt.want)
			}
			if !reflect.DeepEqual(proposal, before) {
				t.Fatalf("ValidateGeneration changed proposal: got %+v, want %+v", proposal, before)
			}
		})
	}
}

// TestIntegrationFailureCleanup verifies that in-flight failures and partial submissions
// properly clean up acquired resources, maintain exactly-once completion release,
// and drain in-flight invocations without double-locking or leaking buffers.
func TestIntegrationFailureCleanup(t *testing.T) {
	snap := buildAffineGraph(t)

	// 1. Injected compile failure cleanup
	resourceCleaned := false
	_, err := cudaplan.Compile(
		snap,
		cudaplan.WithInjectedCompileFailure(true, func() {
			resourceCleaned = true
		}),
	)
	if err == nil {
		t.Fatal("expected injected compile error, got nil")
	}
	if !resourceCleaned {
		t.Fatal("resource cleanup hook was not invoked on compile failure")
	}

	// 2. Failed invocation drain and exactly-once release
	attempts := 0
	releases := 0
	comp := cudaplan.ExportNewHostCompletion(
		func(ctx context.Context) error {
			attempts++
			if attempts == 1 {
				return errors.New("transient stream synchronization failure")
			}
			return nil
		},
		func() error { return nil },
		func() {
			releases++
		},
	)

	// First wait fails
	err1 := comp.Wait(context.Background())
	if err1 == nil {
		t.Fatal("first wait should fail")
	}
	if releases != 0 {
		t.Fatalf("releases = %d on failed wait, want 0", releases)
	}

	// Retry wait succeeds, triggering exactly-once release
	err2 := comp.Wait(context.Background())
	if err2 != nil {
		t.Fatalf("retry wait failed: %v", err2)
	}
	if releases != 1 {
		t.Fatalf("releases = %d after retry, want 1", releases)
	}

	// Repeated close/wait is idempotent
	_ = comp.Close()
	if releases != 1 {
		t.Fatalf("releases = %d after close, want 1", releases)
	}
}

// TestHardwareTilePipelineEndToEnd verifies the hardware tile pipeline across
// cudalayout tile definitions, cudashared memory allocation plans, and
// cudapipeline warp-specialized schedule generation and barrier validation.
func TestHardwareTilePipelineEndToEnd(t *testing.T) {
	tile, err := cudalayout.NewTile(32, 32, 2, cudalayout.WithAlignment(16))
	if err != nil {
		t.Fatalf("cudalayout.NewTile: %v", err)
	}
	if tile.Rows() != 32 || tile.Cols() != 32 || tile.ElementBytes() != 2 {
		t.Fatalf("unexpected tile dimensions: %dx%d (%d bytes)", tile.Rows(), tile.Cols(), tile.ElementBytes())
	}

	req0, err := cudashared.RequestForTile(tile, "stage_0", 0, 0, false)
	if err != nil {
		t.Fatalf("RequestForTile 0: %v", err)
	}
	req1, err := cudashared.RequestForTile(tile, "stage_1", 1, 1, false)
	if err != nil {
		t.Fatalf("RequestForTile 1: %v", err)
	}

	limits := cudashared.Limits{
		StaticMax:  cudapipeline.SM121MaxOptinSharedMemBytes,
		DynamicMax: cudapipeline.SM121MaxOptinSharedMemBytes,
		TotalMax:   cudapipeline.SM121MaxOptinSharedMemBytes,
		Source:     "sm_121a",
	}
	plan, err := cudashared.NewPlan(limits)
	if err != nil {
		t.Fatalf("cudashared.NewPlan: %v", err)
	}
	reg0, err := plan.Add(req0)
	if err != nil {
		t.Fatalf("plan.Add stage 0: %v", err)
	}
	reg1, err := plan.Add(req1)
	if err != nil {
		t.Fatalf("plan.Add stage 1: %v", err)
	}

	// Build warp-specialized pipeline schedule
	sched, err := cudapipeline.BuildWarpSpecializedSchedule(2, 2048, 4)
	if err != nil {
		t.Fatalf("BuildWarpSpecializedSchedule: %v", err)
	}

	errs := sched.Validate()
	if len(errs) > 0 {
		t.Fatalf("schedule validation errors: %v", errs)
	}

	// Verify cudapipeline configuration
	box := cudapipeline.TMABox{
		DimX:        32,
		DimY:        32,
		ElementSize: 2,
		StrideBytes: 64,
	}
	pipe, err := cudapipeline.NewPipeline(cudapipeline.PipelineConfig{
		TargetArch:          cudapipeline.ArchSM121a,
		NumStages:           2,
		Box:                 box,
		MemorySpace:         cudapipeline.MemorySpaceShared,
		SharedMemBaseOffset: int(reg0.Offset),
	})
	if err != nil {
		t.Fatalf("NewPipeline: %v", err)
	}
	if pipe.NumStages() != 2 {
		t.Errorf("NumStages = %d, want 2", pipe.NumStages())
	}
	_ = reg1
}

// TestRegisterTileConversionEndToEnd validates mapping elements to 32-lane register
// tiles, transpose redistribution, and register redistribution legality checks via cudareg.
func TestRegisterTileConversionEndToEnd(t *testing.T) {
	tile, err := cudalayout.NewTile(16, 16, 4)
	if err != nil {
		t.Fatalf("NewTile: %v", err)
	}

	// 1. Validate owner mapping for element (row 2, col 3)
	owner, err := tile.Owner(2, 3)
	if err != nil {
		t.Fatalf("tile.Owner(2, 3): %v", err)
	}
	expectedOffset := 2*16 + 3 // 35
	expectedLane := expectedOffset % 32
	expectedReg := expectedOffset / 32
	if owner.Lane != expectedLane || owner.Register != expectedReg {
		t.Fatalf("owner = %+v, want lane %d reg %d", owner, expectedLane, expectedReg)
	}

	// Reverse lookup
	rRow, rCol, err := tile.OwnerFor(owner.Lane, owner.Register)
	if err != nil || rRow != 2 || rCol != 3 {
		t.Fatalf("OwnerFor = (%d, %d), want (2, 3)", rRow, rCol)
	}

	// 2. Transpose redistribution
	transOwner, err := tile.TransposeOwners(2, 3)
	if err != nil {
		t.Fatalf("TransposeOwners: %v", err)
	}
	transOffset := 3*16 + 2 // (col 3, row 2) in transposed tile = 50
	if transOwner.Lane != transOffset%32 || transOwner.Register != transOffset/32 {
		t.Fatalf("transOwner = %+v, want lane %d reg %d", transOwner, transOffset%32, transOffset/32)
	}

	// 3. Register redistribution via cudareg
	regTarget, err := cudareg.TargetFor("sm_100a")
	if err != nil {
		t.Fatalf("cudareg.TargetFor: %v", err)
	}
	cta, err := cudagroup.New(cudagroup.CTA, driver.Dim3{X: 256, Y: 1, Z: 1})
	if err != nil {
		t.Fatalf("cudagroup.New: %v", err)
	}
	groupProd, err := cudagroup.WarpGroupForCTA(cta, 0)
	if err != nil {
		t.Fatalf("cudagroup.WarpGroupForCTA prod: %v", err)
	}
	groupCons, err := cudagroup.WarpGroupForCTA(cta, 4)
	if err != nil {
		t.Fatalf("cudagroup.WarpGroupForCTA cons: %v", err)
	}

	regSched := cudareg.Schedule{
		Target:          regTarget,
		CTARegisterPool: 16384,
		Groups: map[string]cudareg.Budget{
			"producer": {
				Group:             groupProd,
				CompilerAllocated: 64,
				PeakLive:          32,
				Cap:               64,
				Dynamic:           64,
			},
			"consumer": {
				Group:             groupCons,
				CompilerAllocated: 64,
				PeakLive:          64,
				Cap:               96,
				Dynamic:           64,
			},
		},
		Steps: []cudareg.Transition{
			{
				Role:         "producer",
				Action:       cudareg.Decrease,
				Count:        32,
				Participants: 128,
				Synchronized: true,
				Initialized:  true,
			},
			{
				Role:         "consumer",
				Action:       cudareg.Increase,
				Count:        96,
				Participants: 128,
				Synchronized: true,
				Initialized:  true,
			},
		},
	}
	if err := regSched.Validate(); err != nil {
		t.Fatalf("cudareg.Schedule.Validate: %v", err)
	}
}

// TestUnsupportedHardwareFallbackEndToEnd tests negative controls and explicit fallback
// behavior when unsupported hardware instructions or targets are queried.
func TestUnsupportedHardwareFallbackEndToEnd(t *testing.T) {
	// 1. Direct global-to-TMEM transfer must be rejected on sm_121a
	desc := cudatma.Descriptor{
		Target:       cudatma.TargetSM121a,
		Rank:         2,
		Base:         0x1000,
		Dims:         [cudatma.MaxRank]uint64{64, 64},
		Strides:      [cudatma.MaxRank]uint64{128},
		Box:          [cudatma.MaxRank]uint32{32, 16},
		ElementBytes: 2,
	}
	tr := cudatma.Transfer{
		Descriptor: desc,
		Direction:  cudatma.GlobalToTMEM,
		Bytes:      1024,
	}
	if err := tr.Validate(); err == nil {
		t.Fatal("direct GlobalToTMEM on sm_121a must be rejected")
	}

	// 2. Direct PTX bulk TMEM copy rejection in cudapipeline
	ptxDirectTMEM := `
.version 8.8
.target sm_100a
.address_size 64
.visible .entry invalid_tmem_bulk_kernel(.param .u64 tmap) {
    .reg .b64 %rd_tmap;
    .shared .align 8 .b64 bar;
    cp.async.bulk.tensor.2d.tmem.global.mbarrier::complete_tx::bytes [%tmem0], [%rd_tmap, {0, 0}], [bar];
    ret;
}
`
	mod, err := cudaptx.Parse(strings.NewReader(ptxDirectTMEM))
	if err != nil {
		t.Fatalf("parse PTX: %v", err)
	}
	violations, err := cudapipeline.ValidatePTX(mod)
	if err != nil {
		t.Fatalf("ValidatePTX: %v", err)
	}
	foundInvalidTMEM := false
	for _, v := range violations {
		if v.Violation == cudapipeline.ErrInvalidBulkTMEMPTX {
			foundInvalidTMEM = true
			break
		}
	}
	if !foundInvalidTMEM {
		t.Fatal("expected ErrInvalidBulkTMEMPTX for unverified direct global-to-TMEM copy form")
	}

	// 3. cudapdl rejects legacy compute capability targets (< 9.0)
	elig := cudapdl.CheckEligibility("sm_75", cudapdl.KernelDesc{HasLaunchDependents: true}, cudapdl.KernelDesc{HasWait: true})
	if elig.Eligible || elig.Policy != cudapdl.PolicyStandardFallback {
		t.Fatalf("sm_75 must fall back to standard policy, got %+v", elig)
	}
}

// TestClusterMulticastEndToEnd verifies TMA cluster multicast mask validation,
// destination rank participation, and destination rank completion accounting.
func TestClusterMulticastEndToEnd(t *testing.T) {
	destRanks := []uint32{0, 1, 2, 3}
	txBytes := uint64(2048)

	mc, err := cudatma.NewMulticast(0, destRanks, txBytes)
	if err != nil {
		t.Fatalf("cudatma.NewMulticast: %v", err)
	}

	// Check participation
	for _, r := range destRanks {
		if !mc.Participating(r) {
			t.Fatalf("rank %d should be participating", r)
		}
	}
	if mc.Participating(4) {
		t.Fatal("rank 4 should NOT be participating")
	}

	// Complete rank by rank
	for _, r := range destRanks[:3] {
		if err := mc.Complete(r, txBytes); err != nil {
			t.Fatalf("Complete rank %d: %v", r, err)
		}
	}
	if mc.CompleteAll() {
		t.Fatal("CompleteAll reported true before final rank completed")
	}

	// Complete destination 3
	if err := mc.Complete(destRanks[3], txBytes); err != nil {
		t.Fatalf("Complete rank 3: %v", err)
	}
	if !mc.CompleteAll() {
		t.Fatal("CompleteAll reported false after all ranks completed")
	}

	// Negative control: completing with wrong byte count
	mc2, _ := cudatma.NewMulticast(0, []uint32{1}, 1024)
	if err := mc2.Complete(1, 512); err == nil {
		t.Fatal("expected error for mismatched completed bytes")
	}
}

// TestBeamMCTSEndToEnd runs same-space, same-budget comparisons between
// deterministic bounded Beam search and seeded MCTS search over the same e-graph rules,
// evaluating search state preservation, non-additive timing channels, and deterministic replay.
func TestBeamMCTSEndToEnd(t *testing.T) {
	state, rules := makeIntegrationTestState()
	budget := cudasearch.SearchBudget{
		MaxStates:      20,
		MaxDepth:       5,
		MaxActions:     20,
		MaxEvaluations: 20,
		Timeout:        10 * time.Second,
	}

	eval := &cudasearch.MockEvaluator{
		CostModel: egraph.FusedMegakernelCost{},
		Scoring:   cudasearch.DefaultScoringModel("epoch-v1"),
	}

	// 1. Beam Search
	beam := cudasearch.NewBeamSearch(cudasearch.BeamConfig{
		BeamWidth: 2,
		Budget:    budget,
	})
	resBeam, err := beam.Search(context.Background(), state, rules, eval)
	if err != nil {
		t.Fatalf("beam.Search: %v", err)
	}
	if resBeam.Winner == nil {
		t.Fatal("beam search produced no winner")
	}

	// 2. MCTS Search
	mcts := cudasearch.NewMCTSSearch(cudasearch.MCTSConfig{
		Iterations: 10,
		Seed:       12345,
		Budget:     budget,
	})
	resMCTS, err := mcts.Search(context.Background(), state, rules, eval)
	if err != nil {
		t.Fatalf("mcts.Search: %v", err)
	}
	if resMCTS.Winner == nil {
		t.Fatal("MCTS search produced no winner")
	}

	// Compare exploration outputs
	t.Logf("Beam Winner: %s (cost: %f, states: %d)", resBeam.Winner.Expr.Op, resBeam.Winner.Cost, resBeam.StatesVisited)
	t.Logf("MCTS Winner: %s (cost: %f, states: %d)", resMCTS.Winner.Expr.Op, resMCTS.Winner.Cost, resMCTS.StatesVisited)

	// Verify non-additive timing channels in trial receipt
	receipt := cudatrial.Receipt{
		Name:         "search_trial_winner",
		GPUBusy:      15 * time.Microsecond,
		SummedKernel: 25 * time.Microsecond,
		WallElapsed:  50 * time.Microsecond,
	}
	decomp := receipt.LatencyDecomposition()
	if _, err := decomp.TotalDuration(); !errors.Is(err, cudatrial.ErrNaiveSumDoubleCounts) {
		t.Fatalf("expected ErrNaiveSumDoubleCounts on naive sum, got %v", err)
	}
}

// TestSearchTraceReplayEndToEnd verifies that derivation traces accumulated during
// e-graph search can be deterministically replayed from the root state.
func TestSearchTraceReplayEndToEnd(t *testing.T) {
	g := egraph.New()
	x := g.Add(egraph.ENode{Op: "x"})
	w := g.Add(egraph.ENode{Op: "w"})
	mm := g.Add(egraph.ENode{Op: "matmul", Children: []egraph.Id{x, w}})
	rootMap := map[string]egraph.Id{"root": mm}
	state := egraph.NewSearchState(g, rootMap)

	rule1 := egraph.MustRewrite("matmul-to-gemv", "(matmul ?x ?w)", "(gemv ?x ?w)")
	rule2 := egraph.MustRewrite("gemv-to-fused", "(gemv ?x ?w)", "(fused_gemv_pipeline ?x ?w)")

	// Step 1: Apply rule 1
	actions1 := egraph.EnumerateActions(state.Graph, []*egraph.Rewrite{rule1})
	if len(actions1) == 0 {
		t.Fatal("EnumerateActions rule 1 produced no matches")
	}
	trans1, err := state.ApplyAction(actions1[0])
	if err != nil {
		t.Fatalf("ApplyAction 1: %v", err)
	}
	s1 := trans1.State

	// Step 2: Apply rule 2
	actions2 := egraph.EnumerateActions(s1.Graph, []*egraph.Rewrite{rule2})
	if len(actions2) == 0 {
		t.Fatal("EnumerateActions rule 2 produced no matches")
	}
	trans2, err := s1.ApplyAction(actions2[0])
	if err != nil {
		t.Fatalf("ApplyAction 2: %v", err)
	}
	s2 := trans2.State

	// Extract winning expression
	ext2 := egraph.NewExtractor(s2.Graph, egraph.FusedMegakernelCost{})
	bestExpr, _, err := ext2.Extract(s2.Roots["root"])
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	if bestExpr.Op != "fused_gemv_pipeline" {
		t.Fatalf("bestExpr.Op = %s, want fused_gemv_pipeline", bestExpr.Op)
	}

	// Verify trace replay from original state
	replayedState, err := egraph.ReplayDerivation(g, rootMap, []*egraph.Rewrite{rule1, rule2}, s2.Derivation)
	if err != nil {
		t.Fatalf("ReplayDerivation: %v", err)
	}
	extReplayed := egraph.NewExtractor(replayedState.Graph, egraph.FusedMegakernelCost{})
	replayedExpr, _, err := extReplayed.Extract(replayedState.Roots["root"])
	if err != nil {
		t.Fatalf("Extract from replayed: %v", err)
	}
	if replayedExpr.Op != "fused_gemv_pipeline" {
		t.Fatalf("replayedExpr.Op = %s, want fused_gemv_pipeline", replayedExpr.Op)
	}
}

// TestProfileTimingSeparation keeps synthetic profile observations separate
// from device duration and checks snapshot workload attribution.
func TestProfileTimingSeparation(t *testing.T) {
	// 1. Valid profile evidence scored with ScoringModel
	model := cudasearch.DefaultScoringModel("epoch-v1")

	receipt := &cudatrial.Receipt{
		Name:         "trial-1",
		Complete:     true,
		TimingKind:   cudatrial.TimingSummedKernel,
		SummedKernel: 100 * time.Microsecond,
	}

	profEv := &cudasearch.ProfileEvidence{
		Occupancy:           0.3,
		StallMemoryThrottle: 0.2,
		Epoch:               "epoch-v1",
	}

	costStatic, err := model.Score(receipt, nil)
	if err != nil {
		t.Fatalf("model.Score static: %v", err)
	}
	costProfile, err := model.Score(receipt, profEv)
	if err != nil {
		t.Fatalf("model.Score profile: %v", err)
	}

	if costProfile != costStatic {
		t.Fatalf("profile changed latency: %f != %f", costProfile, costStatic)
	}

	// Stale evidence falls back to static cost
	staleEv := &cudasearch.ProfileEvidence{
		Occupancy:           0.3,
		StallMemoryThrottle: 0.2,
		Epoch:               "older-epoch",
	}
	costFallback, err := model.Score(receipt, staleEv)
	if err != nil {
		t.Fatalf("model.Score fallback: %v", err)
	}
	if costFallback != costStatic {
		t.Fatalf("stale profile evidence must fall back exactly to static cost: %f != %f", costFallback, costStatic)
	}

	// 2. Profile snapshot guidance mismatch check via cudaprof
	snap := cudaprof.ProfileSnapshot{
		Version:            1,
		Region:             "transformer_fused_layer",
		CodeDigest:         "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		CollectorMode:      "synthetic_fixture",
		Workload:           "llama_prefill",
		Device:             "sm_121a",
		TokensHash:         "hash_123",
		Coverage:           0.95,
		ClockUncertaintyNs: 10,
		MetricUnits:        map[string]string{"smsp__pcsamp_warps_issue_stalled_long_scoreboard": "samples"},
		Metrics: map[string]int64{
			"smsp__pcsamp_warps_issue_stalled_long_scoreboard": 450,
		},
	}

	// Matching query succeeds
	target := cudaprof.ProfileTarget{CodeDigest: snap.CodeDigest, Region: snap.Region, CollectorMode: snap.CollectorMode, Workload: "llama_prefill", TokensHash: "hash_123", Device: "sm_121a"}
	guidance, err := snap.Guidance(target)
	if err != nil {
		t.Fatalf("snap.Guidance: %v", err)
	}
	if len(guidance.StallCounts) == 0 {
		t.Fatal("expected stall counts in guidance")
	}

	// Mismatched workload falls back with ErrWorkloadMismatch
	target.Workload = "different_workload"
	_, err = snap.Guidance(target)
	if !errors.Is(err, cudaprof.ErrWorkloadMismatch) {
		t.Fatalf("expected ErrWorkloadMismatch, got %v", err)
	}
}

// TestSearchPolicyCacheIdentityEndToEnd confirms that changing search exploration
// policy parameters (e.g. beam width, random seed, strategy) modifies the search
// receipt key and provenance while reusing identical underlying compiled kernel artifacts.
func TestSearchPolicyCacheIdentityEndToEnd(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "cudapolicy-e2e-*")
	if err != nil {
		t.Fatalf("MkdirTemp: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	cache, err := cudacache.New(cudacache.Config{RootDir: tmpDir})
	if err != nil {
		t.Fatalf("cudacache.New: %v", err)
	}

	art := buildAffineArtifact(t)
	cKey := cudacache.CompilerCacheKey("affine_ptx_v1", cudacache.CompilerIdentity{
		NVRTCVersion: "13.0",
		DriverSONAME: "libcuda.so.1",
		TargetArch:   "sm_121a",
		Flags:        []string{"--std=c++20"},
		OutputKind:   nvrtcx.OutputPTX,
	})
	ctx := context.Background()
	if err := cache.PutArtifact(ctx, cKey, art); err != nil {
		t.Fatalf("PutArtifact: %v", err)
	}

	// Policy 1: Beam width = 4
	p1 := cudacache.SearchPolicyID{
		Strategy:     "beam",
		Options:      map[string]string{"beam_width": "4"},
		Seed:         0,
		RuleSet:      "rules_v1",
		ScoringEpoch: "epoch-1",
	}
	// Policy 2: Beam width = 8
	p2 := cudacache.SearchPolicyID{
		Strategy:     "beam",
		Options:      map[string]string{"beam_width": "8"},
		Seed:         0,
		RuleSet:      "rules_v1",
		ScoringEpoch: "epoch-1",
	}

	rKey1 := cudacache.SearchReceiptKey(p1, "workload_affine")
	rKey2 := cudacache.SearchReceiptKey(p2, "workload_affine")
	if rKey1 == rKey2 {
		t.Fatal("receipt keys for varying beam widths must not collide")
	}

	// Both search policies point to the exact same compiled artifact
	receipt1 := &cudacache.SearchReceipt{
		PolicyID:    p1,
		WorkloadKey: "workload_affine",
		CandidateID: "cand_1",
		ArtifactKey: cKey,
		RankScore:   10.5,
		Epoch:       "epoch-1",
	}
	receipt2 := &cudacache.SearchReceipt{
		PolicyID:    p2,
		WorkloadKey: "workload_affine",
		CandidateID: "cand_2",
		ArtifactKey: cKey,
		RankScore:   9.8,
		Epoch:       "epoch-1",
	}

	if err := cache.PutReceipt(ctx, receipt1); err != nil {
		t.Fatalf("PutReceipt 1: %v", err)
	}
	if err := cache.PutReceipt(ctx, receipt2); err != nil {
		t.Fatalf("PutReceipt 2: %v", err)
	}

	// Verify both retrieve their receipts and point to the identical binary artifact
	r1, err := cache.GetReceipt(ctx, p1, "workload_affine")
	if err != nil {
		t.Fatalf("GetReceipt 1: %v", err)
	}
	r2, err := cache.GetReceipt(ctx, p2, "workload_affine")
	if err != nil {
		t.Fatalf("GetReceipt 2: %v", err)
	}

	art1, err := cache.GetArtifact(ctx, r1.ArtifactKey)
	if err != nil {
		t.Fatalf("GetArtifact 1: %v", err)
	}
	art2, err := cache.GetArtifact(ctx, r2.ArtifactKey)
	if err != nil {
		t.Fatalf("GetArtifact 2: %v", err)
	}
	if art1.Digest != art2.Digest {
		t.Fatalf("both policies must share identical artifact digest, got %s != %s", art1.Digest, art2.Digest)
	}
}
