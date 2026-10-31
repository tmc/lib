package qualify

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/internal/cudatest"
	"github.com/tmc/lib/nvidia/x/pairedtiming"
)

const qualifyHoldoutNativeEnv = "CUDATEST_NATIVE"

type qualifyHoldoutNativeGateClosedError struct{}

func (qualifyHoldoutNativeGateClosedError) Error() string {
	return "qualify: native holdout gate is closed: CUDATEST_NATIVE=1 unset, no device examined"
}

func (qualifyHoldoutNativeGateClosedError) Unwrap() error {
	return cudaerrors.ErrHardwarePending
}

func checkQualifyHoldoutNativeGate(getenv func(string) string) error {
	if getenv(qualifyHoldoutNativeEnv) != "1" {
		return qualifyHoldoutNativeGateClosedError{}
	}
	if _, err := cudalock.Inherited(getenv); err != nil {
		return fmt.Errorf("qualify: native holdout gate open without inherited gpu lock: %w", err)
	}
	return nil
}

// VerifyDigest checks that data is non-empty, want is non-zero, and the
// SHA-256 digest of data equals want.
func VerifyDigest(data []byte, want [32]byte) error {
	if len(data) == 0 {
		return errors.New("verify digest: empty artifact data")
	}
	if want == ([32]byte{}) {
		return errors.New("verify digest: zero expected digest")
	}
	got := sha256.Sum256(data)
	if got != want {
		return fmt.Errorf("verify digest: sha256=%x, want %x", got, want)
	}
	return nil
}

// VerifyTokenSequence checks that ref and cand are non-empty frozen-prompt
// greedy token-ID trajectories and match at every token position.
func VerifyTokenSequence(ref, cand []int) error {
	if len(ref) == 0 || len(cand) == 0 {
		return errors.New("verify token sequence: non-empty reference and candidate token sequences required")
	}
	n := min(len(ref), len(cand))
	for i := range n {
		if ref[i] != cand[i] {
			return fmt.Errorf("verify token sequence: greedy token mismatch at index %d: reference=%d candidate=%d", i, ref[i], cand[i])
		}
	}
	if len(ref) != len(cand) {
		return fmt.Errorf("verify token sequence: trajectory length mismatch: reference=%d candidate=%d", len(ref), len(cand))
	}
	return nil
}

// SelectMeasuredWithHoldout verifies frozen-prompt greedy token-ID parity and
// runs the two-stage 3-block screen + 6-block held-out ABBA + A/A null-control
// promotion gate via [SelectHoldoutPromoted].
func SelectMeasuredWithHoldout(query MeasuredIdentity, ev HoldoutCandidateEvidence, verify func(MeasuredPair) error) (HoldoutPromotionResult, error) {
	return SelectHoldoutPromoted(query, ev, verify)
}

// TestQualifyHoldoutNativeClosedGateControl verifies on any host that when
// CUDATEST_NATIVE is unset, the native qualification gate reports
// cudaerrors.ErrHardwarePending, refuses a presence-only GPU_LOCK_HELD=1,
// accepts a genuine inherited lock, and deterministically exercises
// ProposeStepCandidates, VerifyDigest, VerifyTokenSequence, and
// SelectMeasuredWithHoldout on compiled hopper.TileGEMM and hopper.SkinnyGEMM
// steps.
func TestQualifyHoldoutNativeClosedGateControl(t *testing.T) {
	closedGetenv := os.Getenv
	if os.Getenv(qualifyHoldoutNativeEnv) == "1" {
		closedGetenv = func(k string) string {
			if k == qualifyHoldoutNativeEnv {
				return ""
			}
			return os.Getenv(k)
		}
	}

	err := checkQualifyHoldoutNativeGate(closedGetenv)
	if !errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("checkQualifyHoldoutNativeGate(closed) = %v, want errors.Is(..., cudaerrors.ErrHardwarePending)", err)
	}
	if errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("closed gate matched ErrUnsupported: %v", err)
	}

	// Presence-only GPU_LOCK_HELD=1 must be rejected.
	badEnv := func(k string) string {
		switch k {
		case qualifyHoldoutNativeEnv, "GPU_LOCK_HELD":
			return "1"
		default:
			return ""
		}
	}
	if err := checkQualifyHoldoutNativeGate(badEnv); err == nil || errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("checkQualifyHoldoutNativeGate with GPU_LOCK_HELD=1 = %v, want non-pending lock error", err)
	}

	// Genuine inherited lock on host must pass gate check.
	lockDir := filepath.Join(t.TempDir(), "gpu-lock")
	h, err := cudalock.Acquire(t.Context(), cudalock.WithDir(lockDir), cudalock.WithLabel("qualify-holdout-native-gate-test"))
	if err != nil {
		t.Fatalf("cudalock.Acquire: %v", err)
	}
	t.Cleanup(func() { _ = h.Release() })
	goodEnv := func(k string) string {
		switch k {
		case qualifyHoldoutNativeEnv:
			return "1"
		case "GPU_LOCK_HELD":
			return strconv.Itoa(os.Getpid())
		case "GPU_LOCKDIR":
			return lockDir
		default:
			return ""
		}
	}
	if err := checkQualifyHoldoutNativeGate(goodEnv); err != nil {
		t.Fatalf("checkQualifyHoldoutNativeGate with inherited lock: %v", err)
	}

	// 1. Deterministic ProposeStepCandidates + VerifyDigest + VerifyTokenSequence +
	// SelectMeasuredWithHoldout on hopper.TileGEMM.
	baseTile := hopper.TileGEMM{
		Name:   "holdout_tile_gemm_base",
		N:      128,
		K:      128,
		BM:     128,
		BN:     128,
		Stages: 2,
		WGMMA:  true,
		Weight: hopper.WeightBF16,
		Out:    hopper.F32,
	}
	tileStep := Step{Tile: &baseTile, M: 128}
	tileCausal := KernelObservation{
		Name:               baseTile.Name,
		SharedBankConflict: 0.24,
		WGMMASerialized:    true,
		DeviceSMs:          48,
	}
	tileStalls := StallSample{
		TotalSamples: 100,
		MIOThrottle:  30,
		WaitTMEM:     20,
	}
	tileProposals := ProposeStepCandidates(tileStep, tileCausal, tileStalls)
	if len(tileProposals) < 2 {
		t.Fatalf("ProposeStepCandidates(TileGEMM) returned %d proposals, want >= 2", len(tileProposals))
	}
	if !tileProposals[0].Causal || tileProposals[0].Step.Tile == nil || !tileProposals[0].Step.Tile.Swizzle128B || !tileProposals[0].Step.Tile.WarpSpecialized {
		t.Fatalf("unexpected causal TileGEMM proposal: %+v", tileProposals[0])
	}
	candTile := *tileProposals[0].Step.Tile
	candTile.Name = "holdout_tile_gemm_guided"

	baseTileCUDA, err := baseTile.CUDA()
	if err != nil {
		t.Fatalf("baseTile.CUDA: %v", err)
	}
	candTileCUDA, err := candTile.CUDA()
	if err != nil {
		t.Fatalf("candTile.CUDA: %v", err)
	}
	baseTileDigest := sha256.Sum256([]byte(baseTileCUDA))
	candTileDigest := sha256.Sum256([]byte(candTileCUDA))
	if baseTileDigest == candTileDigest {
		t.Fatal("baseline and guided TileGEMM CUDA sources produced identical SHA-256 digests")
	}
	if err := VerifyDigest([]byte(baseTileCUDA), baseTileDigest); err != nil {
		t.Fatalf("VerifyDigest(baseTileCUDA): %v", err)
	}
	if err := VerifyDigest([]byte(candTileCUDA), candTileDigest); err != nil {
		t.Fatalf("VerifyDigest(candTileCUDA): %v", err)
	}
	if err := VerifyDigest([]byte(candTileCUDA), baseTileDigest); err == nil {
		t.Fatal("VerifyDigest accepted mismatched digest")
	}

	// 2. Deterministic ProposeStepCandidates on hopper.SkinnyGEMM.
	baseSkinny := hopper.SkinnyGEMM{
		Name:         "holdout_skinny_gemm_base",
		N:            4,
		K:            2048,
		MaxM:         2,
		Weight:       hopper.WeightNVFP4,
		Out:          hopper.F32,
		RowsPerBlock: 2,
		Warps:        1,
	}
	skinnyStep := Step{GEMM: &baseSkinny, M: 2}
	skinnyCausal := KernelObservation{
		Name:      baseSkinny.Name,
		DeviceSMs: 48,
	}
	skinnyStalls := StallSample{
		TotalSamples:   100,
		LongScoreboard: 40, // 0.40 -> BottleneckMemoryBoundKV
	}
	skinnyProposals := ProposeStepCandidates(skinnyStep, skinnyCausal, skinnyStalls)
	if len(skinnyProposals) == 0 || skinnyProposals[0].Step.GEMM == nil {
		t.Fatalf("ProposeStepCandidates(SkinnyGEMM) returned %+v, want non-empty GEMM proposals", skinnyProposals)
	}
	candSkinny := *skinnyProposals[0].Step.GEMM
	candSkinny.Name = "holdout_skinny_gemm_guided"
	if candSkinny.Prefetch <= 0 || !candSkinny.BlackwellFP4 {
		t.Fatalf("expected guided SkinnyGEMM with Prefetch > 0 and BlackwellFP4=true, got %+v", candSkinny)
	}
	baseSkinnyCUDA, err := baseSkinny.CUDA()
	if err != nil {
		t.Fatalf("baseSkinny.CUDA: %v", err)
	}
	candSkinnyCUDA, err := candSkinny.CUDA()
	if err != nil {
		t.Fatalf("candSkinny.CUDA: %v", err)
	}
	baseSkinnyDigest := sha256.Sum256([]byte(baseSkinnyCUDA))
	candSkinnyDigest := sha256.Sum256([]byte(candSkinnyCUDA))
	if err := VerifyDigest([]byte(baseSkinnyCUDA), baseSkinnyDigest); err != nil {
		t.Fatalf("VerifyDigest(baseSkinnyCUDA): %v", err)
	}
	if err := VerifyDigest([]byte(candSkinnyCUDA), candSkinnyDigest); err != nil {
		t.Fatalf("VerifyDigest(candSkinnyCUDA): %v", err)
	}

	// Compute deterministic reference greedy tokens on SkinnyGEMM across 2 rows.
	rng := rand.New(rand.NewPCG(20261031, 121))
	const mRows = 2
	xBF16 := randBF16(rng, mRows*baseSkinny.K, 1.0)
	wNVFP4 := randNVFP4(rng, baseSkinny.N, baseSkinny.K)
	refOut, err := baseSkinny.Reference(mRows, xBF16, wNVFP4, nil)
	if err != nil {
		t.Fatalf("baseSkinny.Reference: %v", err)
	}
	candOut, err := candSkinny.Reference(mRows, xBF16, wNVFP4, nil)
	if err != nil {
		t.Fatalf("candSkinny.Reference: %v", err)
	}
	refTokens := greedyArgmaxTokens(refOut, mRows, baseSkinny.N)
	candTokens := greedyArgmaxTokens(candOut, mRows, candSkinny.N)
	if err := VerifyTokenSequence(refTokens, candTokens); err != nil {
		t.Fatalf("VerifyTokenSequence(refTokens, candTokens): %v", err)
	}
	flippedTokens := append([]int(nil), candTokens...)
	flippedTokens[len(flippedTokens)-1] = (flippedTokens[len(flippedTokens)-1] + 1) % baseSkinny.N
	if err := VerifyTokenSequence(refTokens, flippedTokens); err == nil {
		t.Fatal("VerifyTokenSequence accepted mismatched token sequence")
	}

	// Verify SelectMeasuredWithHoldout promotion and rejection paths.
	ctxMaterial := append(append([]byte("sm_121a:skinny_gemm:"), baseSkinnyDigest[:]...), candSkinnyDigest[:]...)
	id := MeasuredIdentity{
		Context:   sha256.Sum256(ctxMaterial),
		Admission: sha256.Sum256([]byte("HOST_CLOSED_GATE_CONTROL_ADMISSION")),
	}
	winBlock := pairedtiming.Block{A1: 12.4, B1: 10.0, B2: 10.1, A2: 12.3}
	nullBlock := pairedtiming.Block{A1: 10.0, B1: 10.0, B2: 10.0, A2: 10.0}
	ev := HoldoutCandidateEvidence{
		Screen: MeasuredPair{
			Identity:  id,
			Reference: baseSkinny.Name,
			Candidate: candSkinny.Name,
			Blocks:    []pairedtiming.Block{winBlock, winBlock, winBlock},
		},
		HoldoutBlocks: []pairedtiming.Block{
			winBlock, winBlock, winBlock, winBlock, winBlock, winBlock,
		},
		AANullBlocks:    []pairedtiming.Block{nullBlock, nullBlock, nullBlock},
		ReferenceTokens: refTokens,
		CandidateTokens: candTokens,
	}
	verify := func(p MeasuredPair) error {
		if p.Identity != id {
			return errors.New("identity mismatch")
		}
		if p.Reference != baseSkinny.Name || p.Candidate != candSkinny.Name {
			return errors.New("handle mismatch")
		}
		return nil
	}
	res, err := SelectMeasuredWithHoldout(id, ev, verify)
	if err != nil {
		t.Fatalf("SelectMeasuredWithHoldout: %v", err)
	}
	if res.Status != "EXACT_HOLDOUT_PROMOTED" || res.CandidateState != "PROMOTED_HOLDOUT_VERIFIED" || !res.TokensMatched {
		t.Fatalf("SelectMeasuredWithHoldout = %+v, want EXACT_HOLDOUT_PROMOTED", res)
	}
	if res.Selected == nil || res.Selected.Name != candSkinny.Name {
		t.Fatalf("SelectMeasuredWithHoldout Selected = %+v, want %q", res.Selected, candSkinny.Name)
	}

	// Verify that a token flip forces REJECTED_TOKEN_MISMATCH even when timing wins.
	badEv := ev
	badEv.CandidateTokens = flippedTokens
	badRes, err := SelectMeasuredWithHoldout(id, badEv, verify)
	if err != nil {
		t.Fatalf("SelectMeasuredWithHoldout(flippedTokens): %v", err)
	}
	if badRes.Status != "REJECTED_TOKEN_MISMATCH" || badRes.TokensMatched {
		t.Fatalf("SelectMeasuredWithHoldout(flippedTokens) = %+v, want REJECTED_TOKEN_MISMATCH", badRes)
	}

	// Verify that the native CUDA harness builder emits all required kernels and markers.
	harness, _, _ := buildQualifyHoldoutNativeHarness(t)
	if strings.Contains(harness, "%!") {
		t.Fatalf("buildQualifyHoldoutNativeHarness contains format error marker %%!:\n%s", harness)
	}
	for _, want := range []string{
		"holdout_skinny_gemm_base",
		"holdout_skinny_gemm_guided",
		"holdout_tile_gemm_base",
		"holdout_tile_gemm_guided",
		"AA_BLOCK",
		"SCREEN_BLOCK",
		"HOLDOUT_BLOCK",
		"REF_TOKENS=",
		"CAND_TOKENS=",
		"MUT_TOKENS=",
		"mutant_rejected=1",
		"QUALIFY_HOLDOUT_NATIVE_OK",
	} {
		if !strings.Contains(harness, want) {
			t.Fatalf("buildQualifyHoldoutNativeHarness missing %q", want)
		}
	}
}

// TestQualifyHoldoutNativeSM121A runs the native GPU qualification and
// guided-vs-blind ABBA + holdout benchmark harness on sm_121a (or the local
// target arch) when CUDATEST_NATIVE=1 is set under gpu-lock.
func TestQualifyHoldoutNativeSM121A(t *testing.T) {
	if os.Getenv(qualifyHoldoutNativeEnv) == "" {
		t.Skip("native qualification pending: skipping native qualification test: set CUDATEST_NATIVE=1 under gpu-lock")
	}
	if err := checkQualifyHoldoutNativeGate(os.Getenv); err != nil {
		t.Fatalf("native gate enabled but %v: run under gpu-lock", err)
	}
	nvcc, cudaVer := cudatest.NVCC(t)
	target := qualifyNativeArch()
	cudatest.SkipUnlessTarget(t, nvcc, target)

	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("user home dir: %v", err)
	}
	tmpBase := filepath.Join(home, "tmp", "nvidia-native")
	if err := os.MkdirAll(tmpBase, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", tmpBase, err)
	}
	dir, err := os.MkdirTemp(tmpBase, "qualify-holdout-native-")
	if err != nil {
		t.Fatalf("mkdirtemp: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })

	harnessSrc, baseUnitSrc, candUnitSrc := buildQualifyHoldoutNativeHarness(t)

	// Stage 1: Compile baseline and guided candidate translation units to cubin
	// with -lineinfo and verify SHA-256 digests with VerifyDigest.
	baseCubin, baseDigest := compileCubinAndVerifyDigest(t, dir, nvcc, target, cudaVer.String(), "holdout_base", baseUnitSrc)
	candCubin, candDigest := compileCubinAndVerifyDigest(t, dir, nvcc, target, cudaVer.String(), "holdout_guided", candUnitSrc)
	t.Logf("cubin digests verified: base=%x (%d bytes) guided=%x (%d bytes)",
		baseDigest[:8], len(baseCubin), candDigest[:8], len(candCubin))

	// Stage 2: Compile and run the ABBA + holdout + A/A null + mutant benchmark binary.
	cuPath := filepath.Join(dir, "qualify_holdout_native.cu")
	binPath := filepath.Join(dir, "qualify_holdout_native")
	if err := os.WriteFile(cuPath, []byte(harnessSrc), 0o600); err != nil {
		t.Fatalf("write %s: %v", cuPath, err)
	}

	buildCtx, buildCancel := context.WithTimeout(t.Context(), 3*time.Minute)
	defer buildCancel()
	buildCmd := exec.CommandContext(buildCtx, nvcc, "-std=c++17", "-O2", "-lineinfo", "-arch="+target, cuPath, "-o", binPath)
	if out, err := buildCmd.CombinedOutput(); err != nil {
		t.Fatalf("nvcc compile (%s, CUDA %s): %v\n%s", target, cudaVer, err, out)
	}

	runCtx, runCancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer runCancel()
	runCmd := exec.CommandContext(runCtx, binPath)
	out, err := runCmd.CombinedOutput()
	if err != nil {
		exitCode := -1
		if runCmd.ProcessState != nil {
			exitCode = runCmd.ProcessState.ExitCode()
		}
		t.Fatalf("native holdout qualification binary failed (exit=%d, arch=%s, CUDA=%s): %v\n%s",
			exitCode, target, cudaVer, err, out)
	}

	outStr := string(out)
	if !strings.Contains(outStr, "QUALIFY_HOLDOUT_NATIVE_OK") || !strings.Contains(outStr, "mutant_rejected=1") {
		t.Fatalf("native holdout qualification output missing expected markers:\n%s", outStr)
	}

	parsed, err := parseHoldoutHarnessOutput(outStr)
	if err != nil {
		t.Fatalf("parseHoldoutHarnessOutput: %v\n%s", err, outStr)
	}
	if err := VerifyTokenSequence(parsed.refTokens, parsed.candTokens); err != nil {
		t.Fatalf("VerifyTokenSequence(ref, cand): %v", err)
	}
	if err := VerifyTokenSequence(parsed.refTokens, parsed.mutTokens); err == nil {
		t.Fatalf("VerifyTokenSequence failed to reject mutated device token trajectory %v", parsed.mutTokens)
	}

	var ctxBuf bytes.Buffer
	ctxBuf.WriteString("qualify-holdout-native:")
	ctxBuf.WriteString(target)
	ctxBuf.WriteString(":")
	ctxBuf.WriteString(cudaVer.String())
	ctxBuf.Write(baseDigest[:])
	ctxBuf.Write(candDigest[:])
	id := MeasuredIdentity{
		Context:   sha256.Sum256(ctxBuf.Bytes()),
		Admission: sha256.Sum256([]byte("NATIVE_GPU_LOCK_ADMISSION:" + parsed.deviceLine)),
	}
	ev := HoldoutCandidateEvidence{
		Screen: MeasuredPair{
			Identity:  id,
			Reference: "holdout_skinny_gemm_base",
			Candidate: "holdout_skinny_gemm_guided",
			Blocks:    parsed.screenBlocks,
		},
		HoldoutBlocks:   parsed.holdoutBlocks,
		AANullBlocks:    parsed.aaBlocks,
		ReferenceTokens: parsed.refTokens,
		CandidateTokens: parsed.candTokens,
		LogitMaxAbs:     parsed.logitMaxAbs,
	}
	promo, err := SelectMeasuredWithHoldout(id, ev, func(p MeasuredPair) error {
		if p.Identity != id {
			return errors.New("identity mismatch")
		}
		if err := VerifyDigest(baseCubin, baseDigest); err != nil {
			return err
		}
		if err := VerifyDigest(candCubin, candDigest); err != nil {
			return err
		}
		return nil
	})
	if err != nil {
		t.Fatalf("SelectMeasuredWithHoldout: %v", err)
	}
	if !promo.TokensMatched {
		t.Fatalf("SelectMeasuredWithHoldout TokensMatched=false: %+v", promo)
	}
	if promo.Status != "EXACT_HOLDOUT_PROMOTED" {
		t.Fatalf("SelectMeasuredWithHoldout Status=%q (state=%q reason=%q), want EXACT_HOLDOUT_PROMOTED",
			promo.Status, promo.CandidateState, promo.Reason)
	}
	if promo.ScreenObserved == nil {
		t.Fatalf("SelectMeasuredWithHoldout ScreenObserved is nil: %+v", promo)
	}

	holdoutRatio := 0.0
	if promo.HoldoutObserved != nil {
		holdoutRatio = promo.HoldoutObserved.Ratio
	} else if sum, err := pairedtiming.Analyze(parsed.holdoutBlocks, 0); err == nil {
		holdoutRatio = sum.Ratio
	}
	aaRatio := 0.0
	if promo.AANullObserved != nil {
		aaRatio = promo.AANullObserved.Ratio
	} else if sum, err := pairedtiming.Analyze(parsed.aaBlocks, 0); err == nil {
		aaRatio = sum.Ratio
	}

	t.Logf("CLAIM: native-qualified on %s (%s), CUDA %s; workload TestQualifyHoldoutNativeSM121A (SkinnyGEMM 4x2048 M=2 + TileGEMM 64x4096 M=64, 3-screen + 6-holdout + 3-AA-null ABBA); metric status=%s state=%s screen_ratio=%.4f holdout_ratio=%.4f aa_null_ratio=%.4f sign_p=%.4f logit_max_abs=%.6f tokens_matched=%v",
		parsed.deviceLine, target, cudaVer, promo.Status, promo.CandidateState,
		promo.ScreenObserved.Ratio, holdoutRatio, aaRatio, promo.OneSidedSignP, parsed.logitMaxAbs, promo.TokensMatched)
}

func compileCubinAndVerifyDigest(t *testing.T, dir, nvcc, target, cudaVer, prefix, src string) ([]byte, [32]byte) {
	t.Helper()
	cuPath := filepath.Join(dir, prefix+".cu")
	cubinPath := filepath.Join(dir, prefix+".cubin")
	if err := os.WriteFile(cuPath, []byte(src), 0o600); err != nil {
		t.Fatalf("write %s: %v", cuPath, err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, nvcc, "-std=c++17", "-O2", "-lineinfo", "--cubin", "-arch="+target, cuPath, "-o", cubinPath)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("nvcc --cubin %s (%s, CUDA %s): %v\n%s", prefix, target, cudaVer, err, out)
	}
	cubin, err := os.ReadFile(cubinPath)
	if err != nil {
		t.Fatalf("read %s: %v", cubinPath, err)
	}
	want := sha256.Sum256(cubin)
	if err := VerifyDigest(cubin, want); err != nil {
		t.Fatalf("VerifyDigest(%s): %v", prefix, err)
	}
	return cubin, want
}

func greedyArgmaxTokens(logits []float32, rows, cols int) []int {
	tokens := make([]int, rows)
	for r := range rows {
		row := logits[r*cols : (r+1)*cols]
		bestIdx := 0
		bestVal := row[0]
		for c := 1; c < cols; c++ {
			if row[c] > bestVal {
				bestVal = row[c]
				bestIdx = c
			}
		}
		tokens[r] = bestIdx
	}
	return tokens
}

type parsedHoldoutOutput struct {
	deviceLine    string
	aaBlocks      []pairedtiming.Block
	screenBlocks  []pairedtiming.Block
	holdoutBlocks []pairedtiming.Block
	refTokens     []int
	candTokens    []int
	mutTokens     []int
	logitMaxAbs   float64
}

func parseHoldoutHarnessOutput(out string) (parsedHoldoutOutput, error) {
	var p parsedHoldoutOutput
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		switch {
		case strings.HasPrefix(line, "device="):
			p.deviceLine = line
		case strings.HasPrefix(line, "AA_BLOCK "):
			b, err := parseTimingBlockLine(line)
			if err != nil {
				return p, err
			}
			p.aaBlocks = append(p.aaBlocks, b)
		case strings.HasPrefix(line, "SCREEN_BLOCK "):
			b, err := parseTimingBlockLine(line)
			if err != nil {
				return p, err
			}
			p.screenBlocks = append(p.screenBlocks, b)
		case strings.HasPrefix(line, "HOLDOUT_BLOCK "):
			b, err := parseTimingBlockLine(line)
			if err != nil {
				return p, err
			}
			p.holdoutBlocks = append(p.holdoutBlocks, b)
		case strings.HasPrefix(line, "REF_TOKENS="):
			toks, err := parseTokenCSV(strings.TrimPrefix(line, "REF_TOKENS="))
			if err != nil {
				return p, err
			}
			p.refTokens = toks
		case strings.HasPrefix(line, "CAND_TOKENS="):
			toks, err := parseTokenCSV(strings.TrimPrefix(line, "CAND_TOKENS="))
			if err != nil {
				return p, err
			}
			p.candTokens = toks
		case strings.HasPrefix(line, "MUT_TOKENS="):
			toks, err := parseTokenCSV(strings.TrimPrefix(line, "MUT_TOKENS="))
			if err != nil {
				return p, err
			}
			p.mutTokens = toks
		case strings.HasPrefix(line, "LOGIT_MAX_ABS="):
			v, err := strconv.ParseFloat(strings.TrimPrefix(line, "LOGIT_MAX_ABS="), 64)
			if err != nil {
				return p, err
			}
			p.logitMaxAbs = v
		}
	}
	if len(p.aaBlocks) != 3 {
		return p, fmt.Errorf("got %d AA_BLOCK lines, want 3", len(p.aaBlocks))
	}
	if len(p.screenBlocks) != 3 {
		return p, fmt.Errorf("got %d SCREEN_BLOCK lines, want 3", len(p.screenBlocks))
	}
	if len(p.holdoutBlocks) != 6 {
		return p, fmt.Errorf("got %d HOLDOUT_BLOCK lines, want 6", len(p.holdoutBlocks))
	}
	if len(p.refTokens) == 0 || len(p.candTokens) == 0 || len(p.mutTokens) == 0 {
		return p, errors.New("missing REF_TOKENS, CAND_TOKENS, or MUT_TOKENS line")
	}
	return p, nil
}

func parseTimingBlockLine(line string) (pairedtiming.Block, error) {
	var b pairedtiming.Block
	for _, f := range strings.Fields(line) {
		k, v, ok := strings.Cut(f, "=")
		if !ok {
			continue
		}
		val, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return b, fmt.Errorf("parse %s in %q: %w", f, line, err)
		}
		switch k {
		case "a1":
			b.A1 = val
		case "b1":
			b.B1 = val
		case "b2":
			b.B2 = val
		case "a2":
			b.A2 = val
		}
	}
	if b.A1 <= 0 || b.B1 <= 0 || b.B2 <= 0 || b.A2 <= 0 {
		return b, fmt.Errorf("nonpositive timing block in %q: %+v", line, b)
	}
	return b, nil
}

func parseTokenCSV(s string) ([]int, error) {
	parts := strings.Split(strings.TrimSpace(s), ",")
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.Atoi(p)
		if err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, nil
}

func buildQualifyHoldoutNativeHarness(t *testing.T) (fullHarness, baseUnit, candUnit string) {
	t.Helper()

	// 1. Baseline and guided candidate SkinnyGEMM via ProposeStepCandidates.
	baseSkinny := hopper.SkinnyGEMM{
		Name:         "holdout_skinny_gemm_base",
		N:            4,
		K:            2048,
		MaxM:         2,
		Weight:       hopper.WeightNVFP4,
		Out:          hopper.F32,
		RowsPerBlock: 2,
		Warps:        1,
	}
	skinnyProposals := ProposeStepCandidates(
		Step{GEMM: &baseSkinny, M: 2},
		KernelObservation{Name: baseSkinny.Name, DeviceSMs: 48},
		StallSample{TotalSamples: 100, LongScoreboard: 40},
	)
	if len(skinnyProposals) == 0 || skinnyProposals[0].Step.GEMM == nil {
		t.Fatal("ProposeStepCandidates failed to produce guided SkinnyGEMM candidate")
	}
	candSkinny := *skinnyProposals[0].Step.GEMM
	candSkinny.Name = "holdout_skinny_gemm_guided"

	baseSkinnyNorm, err := baseSkinny.Norm()
	if err != nil {
		t.Fatalf("baseSkinny.Norm: %v", err)
	}
	candSkinnyNorm, err := candSkinny.Norm()
	if err != nil {
		t.Fatalf("candSkinny.Norm: %v", err)
	}
	baseSkinnyCUDA, err := baseSkinnyNorm.CUDA()
	if err != nil {
		t.Fatalf("baseSkinny.CUDA: %v", err)
	}
	baseSkinnyLaunch, err := baseSkinnyNorm.Launcher()
	if err != nil {
		t.Fatalf("baseSkinny.Launcher: %v", err)
	}
	candSkinnyCUDA, err := candSkinnyNorm.CUDA()
	if err != nil {
		t.Fatalf("candSkinny.CUDA: %v", err)
	}
	candSkinnyLaunch, err := candSkinnyNorm.Launcher()
	if err != nil {
		t.Fatalf("candSkinny.Launcher: %v", err)
	}

	// 2. Baseline and guided candidate TileGEMM via ProposeStepCandidates.
	baseTile := hopper.TileGEMM{
		Name:   "holdout_tile_gemm_base",
		N:      64,
		K:      4096,
		BM:     64,
		BN:     64,
		Stages: 2,
		Weight: hopper.WeightBF16,
		Out:    hopper.F32,
	}
	tileProposals := ProposeStepCandidates(
		Step{Tile: &baseTile, M: 64},
		KernelObservation{Name: baseTile.Name, GridBlocks: 58, DeviceSMs: 48},
		StallSample{TotalSamples: 100, MIOThrottle: 35},
	)
	if len(tileProposals) == 0 || tileProposals[0].Step.Tile == nil {
		t.Fatal("ProposeStepCandidates failed to produce guided TileGEMM candidate")
	}
	candTile := *tileProposals[0].Step.Tile
	candTile.Name = "holdout_tile_gemm_guided"
	// Keep non-WGMMA portable across sm_121a while preserving the guided schedule knobs.
	candTile.WGMMA = false
	candTile.WarpSpecialized = false

	baseTileNorm, err := baseTile.Norm()
	if err != nil {
		t.Fatalf("baseTile.Norm: %v", err)
	}
	candTileNorm, err := candTile.Norm()
	if err != nil {
		t.Fatalf("candTile.Norm: %v", err)
	}
	baseTileCUDA, err := baseTileNorm.CUDA()
	if err != nil {
		t.Fatalf("baseTile.CUDA: %v", err)
	}
	baseTileLaunch, err := baseTileNorm.Launcher()
	if err != nil {
		t.Fatalf("baseTile.Launcher: %v", err)
	}
	candTileCUDA, err := candTileNorm.CUDA()
	if err != nil {
		t.Fatalf("candTile.CUDA: %v", err)
	}
	candTileLaunch, err := candTileNorm.Launcher()
	if err != nil {
		t.Fatalf("candTile.Launcher: %v", err)
	}

	header := "#include <cuda_runtime.h>\n#include <cstdio>\n#include <cmath>\n#include <cstdint>\n#include <cstring>\n" + hopper.Prelude + "\n"
	baseUnit = header + baseSkinnyCUDA + "\n" + baseSkinnyLaunch + "\n" + baseTileCUDA + "\n" + baseTileLaunch + "\n"
	candUnit = header + candSkinnyCUDA + "\n" + candSkinnyLaunch + "\n" + candTileCUDA + "\n" + candTileLaunch + "\n"

	// Build deterministic inputs and reference outputs for SkinnyGEMM (M=2, N=4, K=2048).
	rng := rand.New(rand.NewPCG(20261031, 121))
	const mRows = 2
	xBF16 := randBF16(rng, mRows*baseSkinnyNorm.K, 1.0)
	wNVFP4 := randNVFP4(rng, baseSkinnyNorm.N, baseSkinnyNorm.K)
	wantSkinny, err := baseSkinnyNorm.Reference(mRows, xBF16, wNVFP4, nil)
	if err != nil {
		t.Fatalf("baseSkinny.Reference: %v", err)
	}

	var b strings.Builder
	b.WriteString(header)
	b.WriteString(baseSkinnyCUDA + "\n" + baseSkinnyLaunch + "\n")
	b.WriteString(candSkinnyCUDA + "\n" + candSkinnyLaunch + "\n")
	b.WriteString(baseTileCUDA + "\n" + baseTileLaunch + "\n")
	b.WriteString(candTileCUDA + "\n" + candTileLaunch + "\n")

	fmt.Fprintf(&b, `
#define CK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::printf("CUDA error %%s at line %%d\n", cudaGetErrorString(err__), __LINE__); \
        return 2; \
    } \
} while (0)

static const uint16_t h_x[] = {%s};
static const uint8_t h_w[] = {%s};
static const uint8_t h_wsf[] = {%s};
static const float h_wgs[] = {%.9ff};
static const uint32_t want_bits[] = {%s};

static int argmax_row(const float* row, int cols) {
    int best = 0;
    float best_v = row[0];
    for (int c = 1; c < cols; ++c) {
        if (row[c] > best_v) {
            best_v = row[c];
            best = c;
        }
    }
    return best;
}

template <typename FnA, typename FnB>
static int measure_abba_block(cudaEvent_t ev0, cudaEvent_t ev1, int reps, FnA fnA, FnB fnB, float* a1, float* b1, float* b2, float* a2) {
    CK(cudaEventRecord(ev0));
    for (int r = 0; r < reps; ++r) fnA();
    CK(cudaEventRecord(ev1));
    CK(cudaEventSynchronize(ev1));
    CK(cudaEventElapsedTime(a1, ev0, ev1));

    CK(cudaEventRecord(ev0));
    for (int r = 0; r < reps; ++r) fnB();
    CK(cudaEventRecord(ev1));
    CK(cudaEventSynchronize(ev1));
    CK(cudaEventElapsedTime(b1, ev0, ev1));

    CK(cudaEventRecord(ev0));
    for (int r = 0; r < reps; ++r) fnB();
    CK(cudaEventRecord(ev1));
    CK(cudaEventSynchronize(ev1));
    CK(cudaEventElapsedTime(b2, ev0, ev1));

    CK(cudaEventRecord(ev0));
    for (int r = 0; r < reps; ++r) fnA();
    CK(cudaEventRecord(ev1));
    CK(cudaEventSynchronize(ev1));
    CK(cudaEventElapsedTime(a2, ev0, ev1));
    return 0;
}

int main() {
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    int runtime_ver = 0, driver_ver = 0;
    CK(cudaRuntimeGetVersion(&runtime_ver));
    CK(cudaDriverGetVersion(&driver_ver));
    std::printf("device=%%s sm=%%d.%%d driver=%%d runtime=%%d\n",
                prop.name, prop.major, prop.minor, driver_ver, runtime_ver);

    uint16_t* d_x = nullptr;
    uint8_t *d_w = nullptr, *d_wsf = nullptr;
    float *d_wgs = nullptr, *d_out_ref = nullptr, *d_out_cand = nullptr;
    CK(cudaMalloc(&d_x, sizeof(h_x)));
    CK(cudaMalloc(&d_w, sizeof(h_w)));
    CK(cudaMalloc(&d_wsf, sizeof(h_wsf)));
    CK(cudaMalloc(&d_wgs, sizeof(h_wgs)));
    CK(cudaMalloc(&d_out_ref, %d * sizeof(float)));
    CK(cudaMalloc(&d_out_cand, %d * sizeof(float)));

    CK(cudaMemcpy(d_x, h_x, sizeof(h_x), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_w, h_w, sizeof(h_w), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_wsf, h_wsf, sizeof(h_wsf), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_wgs, h_wgs, sizeof(h_wgs), cudaMemcpyHostToDevice));

    holdout_skinny_gemm_base_launch(%d, d_x, d_w, d_wsf, d_wgs, d_out_ref, 0);
    holdout_skinny_gemm_guided_launch(%d, d_x, d_w, d_wsf, d_wgs, d_out_cand, 0);
    CK(cudaDeviceSynchronize());

    float h_ref[%d] = {};
    float h_cand[%d] = {};
    CK(cudaMemcpy(h_ref, d_out_ref, sizeof(h_ref), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_cand, d_out_cand, sizeof(h_cand), cudaMemcpyDeviceToHost));

    float max_err_ref = 0.0f, max_err_cand = 0.0f, logit_max_abs = 0.0f;
    for (int i = 0; i < %d; ++i) {
        float w = 0.0f;
        std::memcpy(&w, &want_bits[i], sizeof(float));
        float er = std::fabs(h_ref[i] - w);
        float ec = std::fabs(h_cand[i] - w);
        float dc = std::fabs(h_ref[i] - h_cand[i]);
        if (er > max_err_ref) max_err_ref = er;
        if (ec > max_err_cand) max_err_cand = ec;
        if (dc > logit_max_abs) logit_max_abs = dc;
        if (std::isnan(h_ref[i]) || std::isnan(h_cand[i]) || er > 1e-2f || ec > 1e-2f) {
            std::printf("FAIL skinny_gemm idx=%%d ref=%%f cand=%%f want=%%f\n", i, h_ref[i], h_cand[i], w);
            return 3;
        }
    }
    std::printf("LOGIT_MAX_ABS=%%.9f\n", logit_max_abs);

    std::printf("REF_TOKENS=");
    for (int r = 0; r < %d; ++r) {
        if (r > 0) std::printf(",");
        std::printf("%%d", argmax_row(&h_ref[r * %d], %d));
    }
    std::printf("\n");

    std::printf("CAND_TOKENS=");
    for (int r = 0; r < %d; ++r) {
        if (r > 0) std::printf(",");
        std::printf("%%d", argmax_row(&h_cand[r * %d], %d));
    }
    std::printf("\n");

    // Device mutant control: invert scale bytes for row 0's winning column and
    // zero the rest so argmax token flips on device.
    uint8_t h_wsf_mut[sizeof(h_wsf)];
    std::memset(h_wsf_mut, 0, sizeof(h_wsf_mut));
    int win0 = argmax_row(&h_ref[0], %d);
    int alt0 = (win0 + 1) %% %d;
    int sf_per_row = %d / 16;
    for (int s = 0; s < sf_per_row; ++s) {
        h_wsf_mut[alt0 * sf_per_row + s] = 0x70u;
    }
    uint8_t h_w_mut[sizeof(h_w)];
    std::memset(h_w_mut, 0x22, sizeof(h_w_mut));
    uint16_t h_x_mut[%d];
    for (int i = 0; i < %d; ++i) h_x_mut[i] = 0x3c00u;
    CK(cudaMemcpy(d_x, h_x_mut, sizeof(h_x_mut), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_w, h_w_mut, sizeof(h_w_mut), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_wsf, h_wsf_mut, sizeof(h_wsf_mut), cudaMemcpyHostToDevice));
    holdout_skinny_gemm_guided_launch(%d, d_x, d_w, d_wsf, d_wgs, d_out_cand, 0);
    CK(cudaDeviceSynchronize());
    float h_mut[%d] = {};
    CK(cudaMemcpy(h_mut, d_out_cand, sizeof(h_mut), cudaMemcpyDeviceToHost));
    std::printf("MUT_TOKENS=");
    bool mutant_rejected = false;
    for (int r = 0; r < %d; ++r) {
        int mt = argmax_row(&h_mut[r * %d], %d);
        int rt = argmax_row(&h_ref[r * %d], %d);
        if (mt != rt) mutant_rejected = true;
        if (r > 0) std::printf(",");
        std::printf("%%d", mt);
    }
    std::printf("\n");
    if (!mutant_rejected) {
        std::printf("FAIL mutant control did not flip greedy token ID\n");
        return 4;
    }
    std::printf("mutant_rejected=1\n");

    // Restore clean inputs before timing blocks.
    CK(cudaMemcpy(d_x, h_x, sizeof(h_x), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_w, h_w, sizeof(h_w), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_wsf, h_wsf, sizeof(h_wsf), cudaMemcpyHostToDevice));

    // Allocate and verify TileGEMM (M=64, N=64, K=4096) baseline vs guided Stream-K candidate.
    const int tile_m = 64, tile_n = %d, tile_k = %d;
    uint16_t *d_tx = nullptr, *d_tw = nullptr;
    float *d_tout_ref = nullptr, *d_tout_cand = nullptr, *d_tws = nullptr;
    CK(cudaMalloc(&d_tx, (size_t)tile_m * tile_k * sizeof(uint16_t)));
    CK(cudaMalloc(&d_tw, (size_t)tile_n * tile_k * sizeof(uint16_t)));
    CK(cudaMalloc(&d_tout_ref, (size_t)tile_m * tile_n * sizeof(float)));
    CK(cudaMalloc(&d_tout_cand, (size_t)tile_m * tile_n * sizeof(float)));
    CK(cudaMalloc(&d_tws, %d));
    CK(cudaMemset(d_tx, 0x3c, (size_t)tile_m * tile_k * sizeof(uint16_t)));
    CK(cudaMemset(d_tw, 0x3c, (size_t)tile_n * tile_k * sizeof(uint16_t)));
    CK(cudaMemset(d_tws, 0, %d));

    holdout_tile_gemm_base_launch(tile_m, d_tx, d_tw, d_tout_ref, 0);
    holdout_tile_gemm_guided_launch(tile_m, d_tx, d_tw, d_tout_cand, d_tws, 0);
    CK(cudaDeviceSynchronize());

    float h_tref[64 * 64] = {};
    float h_tcand[64 * 64] = {};
    CK(cudaMemcpy(h_tref, d_tout_ref, sizeof(h_tref), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_tcand, d_tout_cand, sizeof(h_tcand), cudaMemcpyDeviceToHost));
    for (int i = 0; i < tile_m * tile_n; ++i) {
        if (std::isnan(h_tref[i]) || std::isnan(h_tcand[i]) || std::fabs(h_tref[i] - h_tcand[i]) > 1e-2f) {
            std::printf("FAIL tile_gemm idx=%%d ref=%%f cand=%%f\n", i, h_tref[i], h_tcand[i]);
            return 8;
        }
    }

    cudaEvent_t ev0, ev1;
    CK(cudaEventCreate(&ev0));
    CK(cudaEventCreate(&ev1));
    const int reps = 50;
    auto runBase = [&]() {
        holdout_skinny_gemm_base_launch(%d, d_x, d_w, d_wsf, d_wgs, d_out_ref, 0);
        holdout_tile_gemm_base_launch(tile_m, d_tx, d_tw, d_tout_ref, 0);
    };
    auto runCand = [&]() {
        holdout_skinny_gemm_guided_launch(%d, d_x, d_w, d_wsf, d_wgs, d_out_cand, 0);
        holdout_tile_gemm_guided_launch(tile_m, d_tx, d_tw, d_tout_cand, d_tws, 0);
    };

    // Warmup
    for (int i = 0; i < 10; ++i) {
        runBase();
        runCand();
    }
    CK(cudaDeviceSynchronize());

    for (int blk = 0; blk < 3; ++blk) {
        float a1 = 0, b1 = 0, b2 = 0, a2 = 0;
        if (measure_abba_block(ev0, ev1, reps, runBase, runBase, &a1, &b1, &b2, &a2) != 0) return 5;
        std::printf("AA_BLOCK %%d a1=%%.6f b1=%%.6f b2=%%.6f a2=%%.6f\n", blk, a1 * 1e6, b1 * 1e6, b2 * 1e6, a2 * 1e6);
    }
    for (int blk = 0; blk < 3; ++blk) {
        float a1 = 0, b1 = 0, b2 = 0, a2 = 0;
        if (measure_abba_block(ev0, ev1, reps, runBase, runCand, &a1, &b1, &b2, &a2) != 0) return 6;
        std::printf("SCREEN_BLOCK %%d a1=%%.6f b1=%%.6f b2=%%.6f a2=%%.6f\n", blk, a1 * 1e6, b1 * 1e6, b2 * 1e6, a2 * 1e6);
    }
    for (int blk = 0; blk < 6; ++blk) {
        float a1 = 0, b1 = 0, b2 = 0, a2 = 0;
        if (measure_abba_block(ev0, ev1, reps, runBase, runCand, &a1, &b1, &b2, &a2) != 0) return 7;
        std::printf("HOLDOUT_BLOCK %%d a1=%%.6f b1=%%.6f b2=%%.6f a2=%%.6f\n", blk, a1 * 1e6, b1 * 1e6, b2 * 1e6, a2 * 1e6);
    }

    std::printf("QUALIFY_HOLDOUT_NATIVE_OK\n");
    return 0;
}
`,
		formatU16Array(xBF16),
		formatU8Array(wNVFP4.Packed),
		formatU8Array(wNVFP4.Scales),
		wNVFP4.GlobalScal,
		cF32s(wantSkinny),
		mRows*baseSkinnyNorm.N, mRows*baseSkinnyNorm.N,
		mRows, mRows,
		mRows*baseSkinnyNorm.N, mRows*baseSkinnyNorm.N,
		mRows*baseSkinnyNorm.N,
		mRows, baseSkinnyNorm.N, baseSkinnyNorm.N,
		mRows, baseSkinnyNorm.N, baseSkinnyNorm.N,
		baseSkinnyNorm.N, baseSkinnyNorm.N, baseSkinnyNorm.K,
		mRows*baseSkinnyNorm.K, mRows*baseSkinnyNorm.K,
		mRows, mRows*baseSkinnyNorm.N,
		mRows, baseSkinnyNorm.N, baseSkinnyNorm.N, baseSkinnyNorm.N, baseSkinnyNorm.N,
		baseTileNorm.N, baseTileNorm.K,
		max(candTileNorm.WorkspaceBytes(), 64), max(candTileNorm.WorkspaceBytes(), 64),
		mRows, mRows,
	)

	return b.String(), baseUnit, candUnit
}

func formatU16Array(v []uint16) string {
	var b strings.Builder
	for i, x := range v {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "%#04xu", x)
	}
	return b.String()
}

func formatU8Array(v []uint8) string {
	var b strings.Builder
	for i, x := range v {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "%#02xu", x)
	}
	return b.String()
}
