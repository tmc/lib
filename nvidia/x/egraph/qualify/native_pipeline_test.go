package qualify

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
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
	"github.com/tmc/lib/nvidia/x/egraph/autotune"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/internal/cudatest"
	"github.com/tmc/lib/nvidia/x/pairedtiming"
)

// TestQualifyPipelineNativeClosedGateControl verifies on any host that when
// CUDATEST_NATIVE is masked as unset, the native qualification gate reports
// cudaerrors.ErrHardwarePending, refuses a presence-only GPU_LOCK_HELD=1,
// accepts a genuine inherited lock, and deterministically exercises
// ExtractStructuralPlanBeam, hopper.BuildCrossLayerMegakernel,
// RefinePlanByPTXASLoop, SelectStructuralBeamWithHoldout, and
// autotune.VerifiedPGOCache without skipping even when CUDATEST_NATIVE=1 is
// set in the outer environment.
func TestQualifyPipelineNativeClosedGateControl(t *testing.T) {
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
	h, err := cudalock.Acquire(t.Context(), cudalock.WithDir(lockDir), cudalock.WithLabel("qualify-pipeline-native-gate-test"))
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

	// 1. Structural beam extraction via ExtractStructuralPlanBeam.
	bGraph, root := buildTransformerFFNGraph(2, 256, 256)
	costCfg := hopper.CostConfig{Device: "NVIDIA GB10"}
	beam, err := ExtractStructuralPlanBeam(bGraph.G, root, "pipeline_ffn", hopper.F32, costCfg, 6)
	if err != nil {
		t.Fatalf("ExtractStructuralPlanBeam: %v", err)
	}
	if len(beam) < 2 {
		t.Fatalf("ExtractStructuralPlanBeam returned %d candidates, want >= 2", len(beam))
	}

	// 2. Cross-layer persistent megakernel lowering via hopper.BuildCrossLayerMegakernel.
	postPlan, nextAttnPlan := buildCrossLayerPipelinePlans(t)
	fusedPlan, crossMega, err := hopper.BuildCrossLayerMegakernel(postPlan, nextAttnPlan, hopper.CrossLayerGlue{
		Name:               "cross_layer_post_attnin",
		LayerLPrefix:       "l0_",
		LayerNextPrefix:    "l1_",
		BridgeInput:        "xn",
		RetainBridgeOutput: true,
	})
	if err != nil {
		t.Fatalf("BuildCrossLayerMegakernel: %v", err)
	}
	if len(crossMega.Stages) != 3 {
		t.Fatalf("len(crossMega.Stages) = %d, want 3", len(crossMega.Stages))
	}
	if len(crossMega.StageAttributionMap()) != 3 {
		t.Fatalf("len(crossMega.StageAttributionMap()) = %d, want 3", len(crossMega.StageAttributionMap()))
	}

	// 3. Closed-loop ptxas refinement via RefinePlanByPTXASLoop with a deterministic PTXASCompiler.
	mockCompiler := &mockPTXASCompiler{
		fn: func(iter int, _ string, arch string) (string, error) {
			if arch != "sm_121a" {
				return "", fmt.Errorf("unexpected arch %q", arch)
			}
			var sb strings.Builder
			sb.WriteString("ptxas info    : 0 bytes gmem\n")
			for i, k := range fusedPlan.Kernels {
				kName := stepKernelName(k)
				if kName == "" {
					kName = fmt.Sprintf("kern_%d", i)
				}
				regs := 72
				spills := 0
				if iter == 1 && i == 0 {
					regs = 168
				}
				fmt.Fprintf(&sb, "ptxas info    : Compiling entry function '%s' for 'sm_121a'\n", kName)
				fmt.Fprintf(&sb, "ptxas info    : Function properties for %s\n", kName)
				fmt.Fprintf(&sb, "    0 bytes stack frame, %d bytes spill stores, %d bytes spill loads\n", spills, spills)
				fmt.Fprintf(&sb, "ptxas info    : Used %d registers, 1024 bytes smem, 384 bytes cmem[0]\n", regs)
			}
			return sb.String(), nil
		},
	}
	loopRes, err := RefinePlanByPTXASLoop(t.Context(), fusedPlan, mockCompiler, PTXASLoopConfig{
		Arch:          "sm_121a",
		Device:        "NVIDIA GB10",
		DeviceSMs:     48,
		MaxIterations: 3,
		Gate: StaticGate{
			NoSpills:     true,
			MaxRegisters: 192,
			MinCTAsPerSM: 1,
		},
	})
	if err != nil {
		t.Fatalf("RefinePlanByPTXASLoop: %v", err)
	}
	if !loopRes.Converged || len(loopRes.Iterations) == 0 || loopRes.Iterations[len(loopRes.Iterations)-1].TotalSpillBytes != 0 {
		t.Fatalf("unexpected RefinePlanByPTXASLoop result: %+v", loopRes)
	}

	// 4. Whole-graph tournament via SelectStructuralBeamWithHoldout + VerifiedPGOCache reload.
	unfusedCand := StructuralBeamCandidate{
		Name:          "unfused_cross_layer_baseline",
		Plan:          fusedPlan,
		PlanSignature: ComputePlanSignature(fusedPlan, nil),
		EstimatedNs:   estimateCandidateNs(fusedPlan, nil, costCfg),
		CostConfig:    costCfg,
	}
	megaCopy := crossMega
	megaCand := StructuralBeamCandidate{
		Name:          "fused_cross_layer_megakernel",
		Plan:          fusedPlan,
		Megakernel:    &megaCopy,
		PlanSignature: ComputePlanSignature(fusedPlan, &megaCopy),
		EstimatedNs:   estimateCandidateNs(fusedPlan, &megaCopy, costCfg),
		CostConfig:    costCfg,
	}

	unfusedSrc, unfusedDigest, err := unfusedCand.Source()
	if err != nil {
		t.Fatalf("unfusedCand.Source: %v", err)
	}
	megaSrc, megaDigest, err := megaCand.Source()
	if err != nil {
		t.Fatalf("megaCand.Source: %v", err)
	}
	if err := VerifyDigest([]byte(unfusedSrc), unfusedDigest); err != nil {
		t.Fatalf("VerifyDigest(unfusedSrc): %v", err)
	}
	if err := VerifyDigest([]byte(megaSrc), megaDigest); err != nil {
		t.Fatalf("VerifyDigest(megaSrc): %v", err)
	}

	query := MeasuredIdentity{
		Context:   sha256.Sum256(append(unfusedDigest[:], megaDigest[:]...)),
		Admission: sha256.Sum256([]byte("HOST_PIPELINE_CLOSED_GATE_ADMISSION")),
	}
	winBlock := pairedtiming.Block{A1: 18.0, B1: 11.2, B2: 11.3, A2: 17.9}
	nullBlock := pairedtiming.Block{A1: 18.0, B1: 18.0, B2: 18.0, A2: 18.0}
	refTokens := []int{11, 29}
	candTokens := []int{11, 29}
	evidence := map[string]HoldoutCandidateEvidence{
		megaCand.Handle(): {
			Screen: MeasuredPair{
				Identity:  query,
				Reference: unfusedCand.Handle(),
				Candidate: megaCand.Handle(),
				Blocks:    []pairedtiming.Block{winBlock, winBlock, winBlock},
			},
			HoldoutBlocks: []pairedtiming.Block{
				winBlock, winBlock, winBlock, winBlock, winBlock, winBlock,
			},
			AANullBlocks:    []pairedtiming.Block{nullBlock, nullBlock, nullBlock},
			ReferenceTokens: refTokens,
			CandidateTokens: candTokens,
		},
	}
	winner, err := SelectStructuralBeamWithHoldout(query, unfusedCand, []StructuralBeamCandidate{megaCand}, evidence, func(p MeasuredPair) error {
		if p.Identity != query {
			return errors.New("identity mismatch")
		}
		if err := VerifyDigest([]byte(unfusedSrc), unfusedDigest); err != nil {
			return err
		}
		return VerifyDigest([]byte(megaSrc), megaDigest)
	})
	if err != nil {
		t.Fatalf("SelectStructuralBeamWithHoldout: %v", err)
	}
	if !winner.Promoted || winner.FallbackToBaseline || winner.Winner.Handle() != megaCand.Handle() {
		t.Fatalf("SelectStructuralBeamWithHoldout = %+v, want promoted %s", winner, megaCand.Handle())
	}

	pgoCache := autotune.NewVerifiedPGOCache(filepath.Join(t.TempDir(), "pgo-cache"))
	cacheKey := autotune.PGOCacheKey{
		Arch:          "sm_121a",
		DriverVersion: "590.48.01",
		CUDAVersion:   "13.2",
		WeightDigest:  fmt.Sprintf("%x", unfusedDigest[:8]),
		PlanKey:       megaCand.PlanSignature,
		ShapeBucket:   2,
	}
	proof := autotune.PGOPromotionProof{
		Key:            cacheKey,
		CandidateName:  winner.Winner.Handle(),
		CubinSHA256:    megaDigest,
		SourceSHA256:   megaDigest,
		ScreenBlocks:   3,
		HoldoutBlocks:  6,
		ScreenRatio:    winner.Screen.Ratio,
		HoldoutRatio:   winner.Holdout.Ratio,
		AANullRatio:    winner.AANullObserved.Ratio,
		OneSidedPValue: winner.OneSidedSignP,
		TokensMatched:  winner.TokensMatched,
	}
	if err := pgoCache.Store(cacheKey, proof, []byte(megaSrc)); err != nil {
		t.Fatalf("pgoCache.Store: %v", err)
	}
	t0 := time.Now()
	loadedProof, loadedBytes, err := pgoCache.LoadVerified(cacheKey)
	reloadDur := time.Since(t0)
	if err != nil {
		t.Fatalf("pgoCache.LoadVerified: %v", err)
	}
	if loadedProof.CubinSHA256 != megaDigest || !bytes.Equal(loadedBytes, []byte(megaSrc)) {
		t.Fatal("pgoCache.LoadVerified digest or payload mismatch")
	}
	if reloadDur >= 50*time.Millisecond {
		t.Fatalf("pgoCache.LoadVerified took %v", reloadDur)
	}

	// Verify native pipeline CUDA harness generation.
	harnessSrc, unfusedUnit, megaUnit := buildQualifyPipelineNativeHarness(t)
	if strings.Contains(harnessSrc, "%!") || strings.Contains(unfusedUnit, "%!") || strings.Contains(megaUnit, "%!") {
		t.Fatal("buildQualifyPipelineNativeHarness produced format error marker %!")
	}
	for _, want := range []string{
		"l0_o_proj",
		"l0_post_norm",
		"l1_qkv_proj",
		"cross_layer_post_attnin_mega_run",
		"AA_BLOCK",
		"SCREEN_BLOCK",
		"HOLDOUT_BLOCK",
		"REF_TOKENS=",
		"CAND_TOKENS=",
		"MUT_TOKENS=",
		"mutant_rejected=1",
		"QUALIFY_PIPELINE_NATIVE_OK",
	} {
		if !strings.Contains(harnessSrc, want) {
			t.Fatalf("buildQualifyPipelineNativeHarness missing %q", want)
		}
	}
}

// TestQualifyPipelineNativeSM121A runs the native end-to-end pipeline
// qualification on NVIDIA GB10 (sm_121a) under gpu-lock when CUDATEST_NATIVE=1:
//   - Part A: RefinePlanByPTXASLoop with real NVCCPTXASCompiler (nvcc --cubin
//     -arch=sm_121a -Xptxas -v -lineinfo) on a multi-step hopper.Plan.
//   - Part B: hopper.BuildCrossLayerMegakernel (Layer[L].post + Layer[L+1].attnIn,
//     M=2, D=256) vs. unfused multi-launch baseline under 3 A/A null blocks +
//     3 ABBA screen blocks + 6 ABBA holdout blocks + greedy token-ID parity +
//     device mutant rejection + SelectStructuralBeamWithHoldout +
//     autotune.VerifiedPGOCache (<1 ms verified SHA-256 reload).
func TestQualifyPipelineNativeSM121A(t *testing.T) {
	if os.Getenv(qualifyHoldoutNativeEnv) == "" {
		t.Skip("native qualification pending: set CUDATEST_NATIVE=1 under gpu-lock")
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
	dir, err := os.MkdirTemp(tmpBase, "qualify-pipeline-native-")
	if err != nil {
		t.Fatalf("mkdirtemp: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })

	// Part A: RefinePlanByPTXASLoop with real NVCCPTXASCompiler on sm_121a.
	postPlan, nextAttnPlan := buildCrossLayerPipelinePlans(t)
	fusedPlan, crossMega, err := hopper.BuildCrossLayerMegakernel(postPlan, nextAttnPlan, hopper.CrossLayerGlue{
		Name:               "cross_layer_post_attnin",
		LayerLPrefix:       "l0_",
		LayerNextPrefix:    "l1_",
		BridgeInput:        "xn",
		RetainBridgeOutput: true,
	})
	if err != nil {
		t.Fatalf("BuildCrossLayerMegakernel: %v", err)
	}

	// Build a multi-step plan whose initial schedule exercises real ptxas -v
	// parsing and compile-time refinement under NVCCPTXASCompiler.
	loopPlan := clonePlan(fusedPlan)
	if loopPlan.Kernels[0].GEMM != nil {
		// Trigger wave-tail / occupancy refinement on 48 SMs (GridBlocks=64 -> 1.333 waves).
		loopPlan.Kernels[0].GEMM.RowsPerBlock = 4
	}
	ptxasCompiler := NVCCPTXASCompiler{
		NVCC:       nvcc,
		ExtraFlags: []string{"-std=c++17", "-O3", "-lineinfo"},
	}
	loopCtx, loopCancel := context.WithTimeout(t.Context(), 3*time.Minute)
	defer loopCancel()
	loopRes, err := RefinePlanByPTXASLoop(loopCtx, loopPlan, ptxasCompiler, PTXASLoopConfig{
		Arch:          target,
		Device:        "NVIDIA GB10",
		DeviceSMs:     48,
		MaxIterations: 4,
		Gate: StaticGate{
			NoSpills:     true,
			MaxRegisters: 192,
			MinCTAsPerSM: 1,
		},
	})
	if err != nil {
		t.Fatalf("RefinePlanByPTXASLoop (real NVCCPTXASCompiler on %s): %v", target, err)
	}
	if len(loopRes.Iterations) < 1 {
		t.Fatalf("len(loopRes.Iterations) = %d, want >= 1", len(loopRes.Iterations))
	}
	if !loopRes.Converged {
		t.Fatalf("RefinePlanByPTXASLoop Converged = false (iterations=%+v)", loopRes.Iterations)
	}
	lastIter := loopRes.Iterations[len(loopRes.Iterations)-1]
	if lastIter.TotalSpillBytes != 0 {
		t.Fatalf("RefinePlanByPTXASLoop final TotalSpillBytes = %d, want 0", lastIter.TotalSpillBytes)
	}
	for i, obs := range lastIter.Observations {
		if obs.Registers <= 0 {
			t.Fatalf("RefinePlanByPTXASLoop step %d (%s) parsed Registers = %d, want > 0", i, obs.Name, obs.Registers)
		}
		t.Logf("PTXASLoop iter=%d step=%d name=%s regs=%d spills=%d grid=%d clean=%v",
			lastIter.Iteration, i, obs.Name, obs.Registers, obs.SpillStoreBytes+obs.SpillLoadBytes, obs.GridBlocks, lastIter.Clean)
	}

	// Part B: Cross-layer persistent megakernel vs. unfused multi-launch baseline on sm_121a.
	harnessSrc, unfusedUnitSrc, megaUnitSrc := buildQualifyPipelineNativeHarness(t)

	unfusedCubin, unfusedDigest := compileCubinO3AndVerifyDigest(t, dir, nvcc, target, cudaVer.String(), "cross_layer_unfused", unfusedUnitSrc)
	megaCubin, megaDigest := compileCubinO3AndVerifyDigest(t, dir, nvcc, target, cudaVer.String(), "cross_layer_mega", megaUnitSrc)
	t.Logf("cubin digests verified: unfused=%x (%d bytes) cross_layer_mega=%x (%d bytes)",
		unfusedDigest[:8], len(unfusedCubin), megaDigest[:8], len(megaCubin))

	cuPath := filepath.Join(dir, "qualify_pipeline_native.cu")
	binPath := filepath.Join(dir, "qualify_pipeline_native")
	if err := os.WriteFile(cuPath, []byte(harnessSrc), 0o600); err != nil {
		t.Fatalf("write %s: %v", cuPath, err)
	}

	buildCtx, buildCancel := context.WithTimeout(t.Context(), 3*time.Minute)
	defer buildCancel()
	buildCmd := exec.CommandContext(buildCtx, nvcc, "-std=c++17", "-O3", "-lineinfo", "-arch="+target, cuPath, "-o", binPath)
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
		t.Fatalf("native pipeline qualification binary failed (exit=%d, arch=%s, CUDA=%s): %v\n%s",
			exitCode, target, cudaVer, err, out)
	}

	outStr := string(out)
	if !strings.Contains(outStr, "QUALIFY_PIPELINE_NATIVE_OK") || !strings.Contains(outStr, "mutant_rejected=1") {
		t.Fatalf("native pipeline qualification output missing expected markers:\n%s", outStr)
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

	costCfg := hopper.CostConfig{Device: "NVIDIA GB10"}
	unfusedCand := StructuralBeamCandidate{
		Name:          "unfused_cross_layer_baseline",
		Plan:          fusedPlan,
		PlanSignature: ComputePlanSignature(fusedPlan, nil),
		EstimatedNs:   estimateCandidateNs(fusedPlan, nil, costCfg),
		CostConfig:    costCfg,
	}
	megaCopy := crossMega
	megaCand := StructuralBeamCandidate{
		Name:          "fused_cross_layer_megakernel",
		Plan:          fusedPlan,
		Megakernel:    &megaCopy,
		PlanSignature: ComputePlanSignature(fusedPlan, &megaCopy),
		EstimatedNs:   estimateCandidateNs(fusedPlan, &megaCopy, costCfg),
		CostConfig:    costCfg,
	}

	var ctxBuf bytes.Buffer
	ctxBuf.WriteString("qualify-pipeline-native:")
	ctxBuf.WriteString(target)
	ctxBuf.WriteString(":")
	ctxBuf.WriteString(cudaVer.String())
	ctxBuf.Write(unfusedDigest[:])
	ctxBuf.Write(megaDigest[:])
	id := MeasuredIdentity{
		Context:   sha256.Sum256(ctxBuf.Bytes()),
		Admission: sha256.Sum256([]byte("NATIVE_GPU_LOCK_PIPELINE_ADMISSION:" + parsed.deviceLine)),
	}
	ev := HoldoutCandidateEvidence{
		Screen: MeasuredPair{
			Identity:  id,
			Reference: unfusedCand.Handle(),
			Candidate: megaCand.Handle(),
			Blocks:    parsed.screenBlocks,
		},
		HoldoutBlocks:   parsed.holdoutBlocks,
		AANullBlocks:    parsed.aaBlocks,
		ReferenceTokens: parsed.refTokens,
		CandidateTokens: parsed.candTokens,
		LogitMaxAbs:     parsed.logitMaxAbs,
	}
	verify := func(p MeasuredPair) error {
		if p.Identity != id {
			return errors.New("identity mismatch")
		}
		if err := VerifyDigest(unfusedCubin, unfusedDigest); err != nil {
			return err
		}
		if err := VerifyDigest(megaCubin, megaDigest); err != nil {
			return err
		}
		return nil
	}

	// Verify both SelectMeasuredWithHoldout and SelectStructuralBeamWithHoldout.
	promo, err := SelectMeasuredWithHoldout(id, ev, verify)
	if err != nil {
		t.Fatalf("SelectMeasuredWithHoldout: %v", err)
	}
	if promo.Status != "EXACT_HOLDOUT_PROMOTED" || !promo.TokensMatched {
		t.Fatalf("SelectMeasuredWithHoldout Status=%q (state=%q reason=%q), want EXACT_HOLDOUT_PROMOTED",
			promo.Status, promo.CandidateState, promo.Reason)
	}

	beamWinner, err := SelectStructuralBeamWithHoldout(
		id,
		unfusedCand,
		[]StructuralBeamCandidate{unfusedCand, megaCand},
		map[string]HoldoutCandidateEvidence{megaCand.Handle(): ev},
		verify,
	)
	if err != nil {
		t.Fatalf("SelectStructuralBeamWithHoldout: %v", err)
	}
	if !beamWinner.Promoted || beamWinner.FallbackToBaseline || beamWinner.Winner.Handle() != megaCand.Handle() {
		t.Fatalf("SelectStructuralBeamWithHoldout = %+v, want promoted %q", beamWinner, megaCand.Handle())
	}

	// Store the promoted sm_121a cross-layer megakernel cubin in autotune.VerifiedPGOCache
	// and verify < 1 ms SHA-256 verified reload.
	cacheDir := filepath.Join(dir, "verified-pgo-cache")
	pgoCache := autotune.NewVerifiedPGOCache(cacheDir)
	schedJSON, _ := json.Marshal(struct {
		Candidate     string `json:"candidate"`
		PlanSignature string `json:"plan_signature"`
		Stages        int    `json:"stages"`
	}{
		Candidate:     beamWinner.Winner.Handle(),
		PlanSignature: beamWinner.Winner.PlanSignature,
		Stages:        len(crossMega.Stages),
	})
	_, megaSrcDigest, err := megaCand.Source()
	if err != nil {
		t.Fatalf("megaCand.Source: %v", err)
	}
	cacheKey := autotune.PGOCacheKey{
		Arch:          target,
		DriverVersion: parsed.deviceLine,
		CUDAVersion:   cudaVer.String(),
		WeightDigest:  fmt.Sprintf("%x", unfusedDigest[:16]),
		PlanKey:       beamWinner.Winner.PlanSignature,
		ShapeBucket:   2,
	}
	aaRatio := 1.0
	if beamWinner.AANullObserved != nil {
		aaRatio = beamWinner.AANullObserved.Ratio
	}
	proof := autotune.PGOPromotionProof{
		Key:            cacheKey,
		CandidateName:  beamWinner.Winner.Handle(),
		ScheduleJSON:   schedJSON,
		CubinSHA256:    megaDigest,
		SourceSHA256:   megaSrcDigest,
		ScreenBlocks:   len(parsed.screenBlocks),
		HoldoutBlocks:  len(parsed.holdoutBlocks),
		ScreenRatio:    beamWinner.Screen.Ratio,
		HoldoutRatio:   beamWinner.Holdout.Ratio,
		AANullRatio:    aaRatio,
		OneSidedPValue: beamWinner.OneSidedSignP,
		TokensMatched:  beamWinner.TokensMatched,
	}
	if err := pgoCache.Store(cacheKey, proof, megaCubin); err != nil {
		t.Fatalf("VerifiedPGOCache.Store: %v", err)
	}
	reloadStart := time.Now()
	loadedProof, loadedCubin, err := pgoCache.LoadVerified(cacheKey)
	reloadDur := time.Since(reloadStart)
	if err != nil {
		t.Fatalf("VerifiedPGOCache.LoadVerified: %v", err)
	}
	if err := VerifyDigest(loadedCubin, megaDigest); err != nil {
		t.Fatalf("VerifyDigest(loadedCubin): %v", err)
	}
	if loadedProof.CubinSHA256 != megaDigest || !loadedProof.TokensMatched {
		t.Fatalf("unexpected loaded proof: %+v", loadedProof)
	}
	if reloadDur >= time.Millisecond {
		t.Fatalf("VerifiedPGOCache.LoadVerified took %v, want < 1ms", reloadDur)
	}

	t.Logf("CLAIM: native-qualified on %s (%s), CUDA %s; workload TestQualifyPipelineNativeSM121A (RefinePlanByPTXASLoop iters=%d spills=%d + BuildCrossLayerMegakernel 3-stage M=2 D=256 vs unfused multi-launch, 3-screen + 6-holdout + 3-AA-null ABBA + VerifiedPGOCache reload); metric status=%s winner=%s screen_ratio=%.4f holdout_ratio=%.4f aa_null_ratio=%.4f sign_p=%.4f logit_max_abs=%.6f tokens_matched=%v pgo_cache_reload_us=%.2f",
		parsed.deviceLine, target, cudaVer, len(loopRes.Iterations), lastIter.TotalSpillBytes,
		beamWinner.Status, beamWinner.Winner.Handle(), beamWinner.Screen.Ratio, beamWinner.Holdout.Ratio,
		aaRatio, beamWinner.OneSidedSignP, parsed.logitMaxAbs, beamWinner.TokensMatched,
		float64(reloadDur.Nanoseconds())/1e3)
}

func compileCubinO3AndVerifyDigest(t *testing.T, dir, nvcc, target, cudaVer, prefix, src string) ([]byte, [32]byte) {
	t.Helper()
	cuPath := filepath.Join(dir, prefix+".cu")
	cubinPath := filepath.Join(dir, prefix+".cubin")
	if err := os.WriteFile(cuPath, []byte(src), 0o600); err != nil {
		t.Fatalf("write %s: %v", cuPath, err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, nvcc, "-std=c++17", "-O3", "-lineinfo", "--cubin", "-arch="+target, cuPath, "-o", cubinPath)
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

// buildCrossLayerPipelinePlans builds a layer-L post plan (o_proj +
// post_residual_rmsnorm -> xn) and a layer-(L+1) attnIn plan (qkv_proj + bias ->
// qkv) at M=2, D=256 for cross-layer persistent megakernel qualification.
func buildCrossLayerPipelinePlans(t *testing.T) (post, nextAttnIn *hopper.Plan) {
	t.Helper()
	const (
		m   = 2
		d   = 256
		eps = 1e-5
	)

	// Layer[L].post: o_proj(attn, w_o) + residual x -> RMSNorm(next_norm) -> Cast(BF16) -> xn
	bPost := hopper.NewBuilder()
	attn := bPost.Input("attn", hopper.BF16, m, d)
	x := bPost.Input("x", hopper.F32, m, d)
	wo := bPost.Input("w_o", hopper.BF16, d, d)
	nextNorm := bPost.Input("next_norm", hopper.F32, d)
	down := bPost.Map(hopper.MustLambda("(add $0 $1)"), bPost.MatMul(attn, wo), x)
	xnOut := bPost.Cast(hopper.BF16, bPost.RMSNorm(down, nextNorm, d, eps))

	postExpr, err := hopper.Optimize(bPost.G, xnOut)
	if err != nil {
		t.Fatalf("optimize post: %v", err)
	}
	post, err = hopper.Compile(postExpr, "l0_post", hopper.BF16)
	if err != nil {
		t.Fatalf("compile post: %v", err)
	}
	if len(post.Kernels) != 2 {
		t.Fatalf("post has %d kernels, want 2", len(post.Kernels))
	}
	setStepKernelName(&post.Kernels[0], "l0_o_proj")
	setStepKernelName(&post.Kernels[1], "l0_post_norm")
	if post.Kernels[0].GEMM != nil {
		post.Kernels[0].GEMM.RowsPerBlock = 32
		post.Kernels[0].GEMM.Warps = 8
	}

	// Layer[L+1].attnIn: MatMul(xn, w_qkv) + b_qkv -> qkv
	bAttn := hopper.NewBuilder()
	xnIn := bAttn.Input("xn", hopper.BF16, m, d)
	wqkv := bAttn.Input("w_qkv", hopper.BF16, d, d)
	bqkv := bAttn.Input("b_qkv", hopper.F32, d)
	qkv := bAttn.Map(hopper.MustLambda("(add $0 $1)"), bAttn.MatMul(xnIn, wqkv), bqkv)

	attnExpr, err := hopper.Optimize(bAttn.G, qkv)
	if err != nil {
		t.Fatalf("optimize nextAttnIn: %v", err)
	}
	nextAttnIn, err = hopper.Compile(attnExpr, "l1_attn_in", hopper.F32)
	if err != nil {
		t.Fatalf("compile nextAttnIn: %v", err)
	}
	if len(nextAttnIn.Kernels) != 1 {
		t.Fatalf("nextAttnIn has %d kernels, want 1", len(nextAttnIn.Kernels))
	}
	setStepKernelName(&nextAttnIn.Kernels[0], "l1_qkv_proj")
	if nextAttnIn.Kernels[0].GEMM != nil {
		nextAttnIn.Kernels[0].GEMM.RowsPerBlock = 32
		nextAttnIn.Kernels[0].GEMM.Warps = 8
	}
	return post, nextAttnIn
}

func buildQualifyPipelineNativeHarness(t *testing.T) (fullHarness, unfusedUnit, megaUnit string) {
	t.Helper()
	const (
		m = 2
		d = 256
	)
	postPlan, nextAttnPlan := buildCrossLayerPipelinePlans(t)
	fusedPlan, crossMega, err := hopper.BuildCrossLayerMegakernel(postPlan, nextAttnPlan, hopper.CrossLayerGlue{
		Name:               "cross_layer_post_attnin",
		LayerLPrefix:       "l0_",
		LayerNextPrefix:    "l1_",
		BridgeInput:        "xn",
		RetainBridgeOutput: true,
	})
	if err != nil {
		t.Fatalf("BuildCrossLayerMegakernel: %v", err)
	}
	crossMega.Name = "cross_layer_post_attnin_mega"

	unfusedSrc, err := fusedPlan.Source()
	if err != nil {
		t.Fatalf("fusedPlan.Source: %v", err)
	}
	megaLineSrc, _, err := crossMega.LineInfoSource()
	if err != nil {
		t.Fatalf("crossMega.LineInfoSource: %v", err)
	}
	megaCUDA, _, err := crossMega.LineInfoCUDA()
	if err != nil {
		t.Fatalf("crossMega.LineInfoCUDA: %v", err)
	}
	megaLaunch, err := crossMega.Launcher()
	if err != nil {
		t.Fatalf("crossMega.Launcher: %v", err)
	}

	stdHeader := "#include <cuda_runtime.h>\n#include <cstdio>\n#include <cmath>\n#include <cstdint>\n#include <cstring>\n"
	unfusedUnit = stdHeader + unfusedSrc + "\n"
	megaUnit = stdHeader + megaLineSrc + "\n"

	// Compute deterministic CPU reference outputs across the 3 cross-layer stages:
	// Stage 0: l0_o_proj (attn [2,256] BF16 * w_o [256,256] BF16 + x [2,256] F32 -> l0_post_t1 [2,256] F32)
	// Stage 1: l0_post_norm (RMSNorm(l0_post_t1, next_norm) -> l0_post_out [2,256] BF16)
	// Stage 2: l1_qkv_proj (l0_post_out [2,256] BF16 * w_qkv [256,256] BF16 + b_qkv [256] F32 -> l1_attn_in_out [2,256] F32)
	rng := rand.New(rand.NewPCG(20261031, 1210))
	attnBF16 := randBF16(rng, m*d, 0.8)
	woBF16 := randBF16(rng, d*d, 0.25)
	xF32 := make([]float32, m*d)
	for i := range xF32 {
		xF32[i] = float32(rng.NormFloat64() * 0.5)
	}
	nextNormF32 := make([]float32, d)
	for i := range nextNormF32 {
		nextNormF32[i] = 0.75 + 0.25*float32(rng.Float64())
	}
	wqkvBF16 := randBF16(rng, d*d, 0.25)
	bqkvF32 := make([]float32, d)
	for i := range bqkvF32 {
		bqkvF32[i] = float32(rng.NormFloat64() * 0.2)
	}

	g0, err := fusedPlan.Kernels[0].GEMM.Norm()
	if err != nil {
		t.Fatalf("stage 0 GEMM.Norm: %v", err)
	}
	t1Out, err := g0.Reference(m, attnBF16, hopper.GEMMWeights{BF16: woBF16}, [][]float32{xF32})
	if err != nil {
		t.Fatalf("stage 0 Reference: %v", err)
	}

	rk1 := *fusedPlan.Kernels[1].Row
	rowBufs := map[string][]float64{
		"l0_post_t1":  f64s(t1Out),
		"next_norm":   f64s(nextNormF32),
		"l0_post_out": make([]float64, m*d),
	}
	if err := rk1.Reference(m, rowBufs); err != nil {
		t.Fatalf("stage 1 RowKernel.Reference: %v", err)
	}
	xnBF16 := make([]uint16, m*d)
	for i, v := range rowBufs["l0_post_out"] {
		xnBF16[i] = hopper.BF16Bits(float32(v))
	}

	g2, err := fusedPlan.Kernels[2].GEMM.Norm()
	if err != nil {
		t.Fatalf("stage 2 GEMM.Norm: %v", err)
	}
	wantOut, err := g2.Reference(m, xnBF16, hopper.GEMMWeights{BF16: wqkvBF16}, [][]float32{bqkvF32})
	if err != nil {
		t.Fatalf("stage 2 Reference: %v", err)
	}

	var b strings.Builder
	b.WriteString(stdHeader)
	b.WriteString(unfusedSrc + "\n")
	b.WriteString(megaCUDA + "\n" + megaLaunch + "\n")

	fmt.Fprintf(&b, `
#define CK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::printf("CUDA error %%s at line %%d\n", cudaGetErrorString(err__), __LINE__); \
        return 2; \
    } \
} while (0)

static const uint16_t h_attn[] = {%s};
static const uint16_t h_w_o[] = {%s};
static const uint32_t h_x_bits[] = {%s};
static const uint32_t h_next_norm_bits[] = {%s};
static const uint16_t h_w_qkv[] = {%s};
static const uint32_t h_b_qkv_bits[] = {%s};
static const uint32_t want_out_bits[] = {%s};

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

    const int M = %d, D = %d;
    uint16_t *d_attn = nullptr, *d_w_o = nullptr, *d_w_qkv = nullptr;
    float *d_x = nullptr, *d_next_norm = nullptr, *d_b_qkv = nullptr;
    float *d_t1_ref = nullptr, *d_t1_cand = nullptr;
    uint16_t *d_xn_ref = nullptr, *d_xn_cand = nullptr;
    float *d_out_ref = nullptr, *d_out_cand = nullptr;
    unsigned long long *d_ready = nullptr;

    CK(cudaMalloc(&d_attn, sizeof(h_attn)));
    CK(cudaMalloc(&d_w_o, sizeof(h_w_o)));
    CK(cudaMalloc(&d_x, sizeof(h_x_bits)));
    CK(cudaMalloc(&d_next_norm, sizeof(h_next_norm_bits)));
    CK(cudaMalloc(&d_w_qkv, sizeof(h_w_qkv)));
    CK(cudaMalloc(&d_b_qkv, sizeof(h_b_qkv_bits)));

    CK(cudaMalloc(&d_t1_ref, M * D * sizeof(float)));
    CK(cudaMalloc(&d_t1_cand, M * D * sizeof(float)));
    CK(cudaMalloc(&d_xn_ref, M * D * sizeof(uint16_t)));
    CK(cudaMalloc(&d_xn_cand, M * D * sizeof(uint16_t)));
    CK(cudaMalloc(&d_out_ref, M * D * sizeof(float)));
    CK(cudaMalloc(&d_out_cand, M * D * sizeof(float)));
    CK(cudaMalloc(&d_ready, %d));
    CK(cudaMemset(d_ready, 0, %d));

    CK(cudaMemcpy(d_attn, h_attn, sizeof(h_attn), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_w_o, h_w_o, sizeof(h_w_o), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_x, h_x_bits, sizeof(h_x_bits), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_next_norm, h_next_norm_bits, sizeof(h_next_norm_bits), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_w_qkv, h_w_qkv, sizeof(h_w_qkv), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_b_qkv, h_b_qkv_bits, sizeof(h_b_qkv_bits), cudaMemcpyHostToDevice));

    unsigned int epoch = 1u;
    cross_layer_post_attnin_run(d_attn, d_w_o, d_x, d_next_norm, d_w_qkv, d_b_qkv, d_t1_ref, d_xn_ref, d_out_ref, 0);
    cross_layer_post_attnin_mega_run(d_attn, d_w_o, d_x, d_next_norm, d_w_qkv, d_b_qkv, d_t1_cand, d_xn_cand, d_out_cand, d_ready, epoch++, 0);
    CK(cudaDeviceSynchronize());

    float h_ref[M * D] = {};
    float h_cand[M * D] = {};
    CK(cudaMemcpy(h_ref, d_out_ref, sizeof(h_ref), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_cand, d_out_cand, sizeof(h_cand), cudaMemcpyDeviceToHost));

    float max_err_ref = 0.0f, max_err_cand = 0.0f, logit_max_abs = 0.0f;
    for (int i = 0; i < M * D; ++i) {
        float w = 0.0f;
        std::memcpy(&w, &want_out_bits[i], sizeof(float));
        float er = std::fabs(h_ref[i] - w);
        float ec = std::fabs(h_cand[i] - w);
        float dc = std::fabs(h_ref[i] - h_cand[i]);
        if (er > max_err_ref) max_err_ref = er;
        if (ec > max_err_cand) max_err_cand = ec;
        if (dc > logit_max_abs) logit_max_abs = dc;
        if (std::isnan(h_ref[i]) || std::isnan(h_cand[i]) || er > 2e-2f || ec > 2e-2f) {
            std::printf("FAIL cross_layer idx=%%d ref=%%f cand=%%f want=%%f\n", i, h_ref[i], h_cand[i], w);
            return 3;
        }
    }
    std::printf("LOGIT_MAX_ABS=%%.9f\n", logit_max_abs);

    std::printf("REF_TOKENS=");
    for (int r = 0; r < M; ++r) {
        if (r > 0) std::printf(",");
        std::printf("%%d", argmax_row(&h_ref[r * D], D));
    }
    std::printf("\n");

    std::printf("CAND_TOKENS=");
    for (int r = 0; r < M; ++r) {
        if (r > 0) std::printf(",");
        std::printf("%%d", argmax_row(&h_cand[r * D], D));
    }
    std::printf("\n");

    // Device mutant control: perturb b_qkv so an alternative column wins argmax on device.
    float h_b_mut[D] = {};
    for (int c = 0; c < D; ++c) {
        std::memcpy(&h_b_mut[c], &h_b_qkv_bits[c], sizeof(float));
    }
    int win0 = argmax_row(&h_ref[0], D);
    int alt0 = (win0 + 1) %% D;
    h_b_mut[alt0] += 100.0f;
    CK(cudaMemcpy(d_b_qkv, h_b_mut, sizeof(h_b_mut), cudaMemcpyHostToDevice));
    cross_layer_post_attnin_mega_run(d_attn, d_w_o, d_x, d_next_norm, d_w_qkv, d_b_qkv, d_t1_cand, d_xn_cand, d_out_cand, d_ready, epoch++, 0);
    CK(cudaDeviceSynchronize());

    float h_mut[M * D] = {};
    CK(cudaMemcpy(h_mut, d_out_cand, sizeof(h_mut), cudaMemcpyDeviceToHost));
    std::printf("MUT_TOKENS=");
    bool mutant_rejected = false;
    for (int r = 0; r < M; ++r) {
        int mt = argmax_row(&h_mut[r * D], D);
        int rt = argmax_row(&h_ref[r * D], D);
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

    // Restore clean b_qkv before timing blocks.
    CK(cudaMemcpy(d_b_qkv, h_b_qkv_bits, sizeof(h_b_qkv_bits), cudaMemcpyHostToDevice));

    cudaEvent_t ev0, ev1;
    CK(cudaEventCreate(&ev0));
    CK(cudaEventCreate(&ev1));
    const int reps = 50;

    auto runUnfused = [&]() {
        cross_layer_post_attnin_run(d_attn, d_w_o, d_x, d_next_norm, d_w_qkv, d_b_qkv, d_t1_ref, d_xn_ref, d_out_ref, 0);
    };
    auto runFusedMega = [&]() {
        cross_layer_post_attnin_mega_run(d_attn, d_w_o, d_x, d_next_norm, d_w_qkv, d_b_qkv, d_t1_cand, d_xn_cand, d_out_cand, d_ready, epoch++, 0);
    };

    // Warmup
    for (int i = 0; i < 10; ++i) {
        runUnfused();
        runFusedMega();
    }
    CK(cudaDeviceSynchronize());

    for (int blk = 0; blk < 3; ++blk) {
        float a1 = 0, b1 = 0, b2 = 0, a2 = 0;
        if (measure_abba_block(ev0, ev1, reps, runUnfused, runUnfused, &a1, &b1, &b2, &a2) != 0) return 5;
        std::printf("AA_BLOCK %%d a1=%%.6f b1=%%.6f b2=%%.6f a2=%%.6f\n", blk, a1 * 1e6, b1 * 1e6, b2 * 1e6, a2 * 1e6);
    }
    for (int blk = 0; blk < 3; ++blk) {
        float a1 = 0, b1 = 0, b2 = 0, a2 = 0;
        if (measure_abba_block(ev0, ev1, reps, runUnfused, runFusedMega, &a1, &b1, &b2, &a2) != 0) return 6;
        std::printf("SCREEN_BLOCK %%d a1=%%.6f b1=%%.6f b2=%%.6f a2=%%.6f\n", blk, a1 * 1e6, b1 * 1e6, b2 * 1e6, a2 * 1e6);
    }
    for (int blk = 0; blk < 6; ++blk) {
        float a1 = 0, b1 = 0, b2 = 0, a2 = 0;
        if (measure_abba_block(ev0, ev1, reps, runUnfused, runFusedMega, &a1, &b1, &b2, &a2) != 0) return 7;
        std::printf("HOLDOUT_BLOCK %%d a1=%%.6f b1=%%.6f b2=%%.6f a2=%%.6f\n", blk, a1 * 1e6, b1 * 1e6, b2 * 1e6, a2 * 1e6);
    }

    std::printf("QUALIFY_PIPELINE_NATIVE_OK\n");
    return 0;
}
`,
		formatU16Array(attnBF16),
		formatU16Array(woBF16),
		cF32s(xF32),
		cF32s(nextNormF32),
		formatU16Array(wqkvBF16),
		cF32s(bqkvF32),
		cF32s(wantOut),
		m, d,
		crossMega.ReadyBytes(), crossMega.ReadyBytes(),
	)

	return b.String(), unfusedUnit, megaUnit
}
