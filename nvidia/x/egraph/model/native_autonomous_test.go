package model

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/egraph/autotune"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/egraph/qualify"
	"github.com/tmc/lib/nvidia/x/egraph/specdecode"
	"github.com/tmc/lib/nvidia/x/internal/cudatest"
	"github.com/tmc/lib/nvidia/x/pairedtiming"
)

const autonomousNativeEnv = "CUDATEST_NATIVE"

type autonomousNativeGateClosedError struct{}

func (autonomousNativeGateClosedError) Error() string {
	return "model: native autonomous qualification gate is closed: CUDATEST_NATIVE=1 unset, no device examined"
}

func (autonomousNativeGateClosedError) Unwrap() error {
	return cudaerrors.ErrHardwarePending
}

func nativeAutonomousGateSkipReason() string {
	if os.Getenv(autonomousNativeEnv) != "1" {
		return "native qualification pending: set CUDATEST_NATIVE=1 under gpu-lock"
	}
	return ""
}

func checkAutonomousNativeGate(getenv func(string) string) error {
	if getenv(autonomousNativeEnv) != "1" {
		return autonomousNativeGateClosedError{}
	}
	if _, err := cudalock.Inherited(getenv); err != nil {
		return fmt.Errorf("model: native autonomous gate open without inherited gpu lock: %w", err)
	}
	return nil
}

func autonomousNativeArch() string {
	if arch := strings.TrimSpace(os.Getenv("CUDA_EGRAPH_MODEL_NATIVE_ARCH")); arch != "" {
		return arch
	}
	return "sm_121a"
}

// TestAutonomousFlightNativeClosedGateControl verifies on any host that when
// CUDATEST_NATIVE is masked as unset, nativeAutonomousGateSkipReason returns a
// non-empty skip reason, checkAutonomousNativeGate reports
// cudaerrors.ErrHardwarePending, presence-only GPU_LOCK_HELD=1 is rejected, a
// genuine inherited lock is accepted, and the 4-layer cross-layer stack,
// StageLinePCMap + AttributeMegakernelBatch, RunAutonomousFlightPipeline
// cold-start promotion + warm-start VerifiedPGOCache reload, mutant rejection,
// and 4-layer CUDA benchmark harness builder all succeed deterministically.
func TestAutonomousFlightNativeClosedGateControl(t *testing.T) {
	t.Run("MaskedGate", func(t *testing.T) {
		t.Setenv(autonomousNativeEnv, "")
		reason := nativeAutonomousGateSkipReason()
		if !strings.HasPrefix(reason, "native qualification pending") {
			t.Fatalf("nativeAutonomousGateSkipReason() = %q, want prefix %q", reason, "native qualification pending")
		}
		err := checkAutonomousNativeGate(os.Getenv)
		if !errors.Is(err, cudaerrors.ErrHardwarePending) {
			t.Fatalf("checkAutonomousNativeGate(closed) = %v, want errors.Is(..., cudaerrors.ErrHardwarePending)", err)
		}
		if errors.Is(err, cudaerrors.ErrUnsupported) {
			t.Fatalf("closed gate matched ErrUnsupported: %v", err)
		}
	})

	// Presence-only GPU_LOCK_HELD=1 must be rejected.
	badEnv := func(k string) string {
		switch k {
		case autonomousNativeEnv, "GPU_LOCK_HELD":
			return "1"
		default:
			return ""
		}
	}
	if err := checkAutonomousNativeGate(badEnv); err == nil || errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("checkAutonomousNativeGate with GPU_LOCK_HELD=1 = %v, want non-pending lock error", err)
	}

	// Genuine inherited lock on host must pass gate check.
	lockDir := filepath.Join(t.TempDir(), "gpu-lock")
	h, err := cudalock.Acquire(t.Context(), cudalock.WithDir(lockDir), cudalock.WithLabel("autonomous-native-gate-test"))
	if err != nil {
		t.Fatalf("cudalock.Acquire: %v", err)
	}
	t.Cleanup(func() { _ = h.Release() })
	goodEnv := func(k string) string {
		switch k {
		case autonomousNativeEnv:
			return "1"
		case "GPU_LOCK_HELD":
			return strconv.Itoa(os.Getpid())
		case "GPU_LOCKDIR":
			return lockDir
		default:
			return ""
		}
	}
	if err := checkAutonomousNativeGate(goodEnv); err != nil {
		t.Fatalf("checkAutonomousNativeGate with inherited lock: %v", err)
	}

	// Build 4-layer decoder stack (M=2, D=256) and verify cross-layer fusion across 3 boundaries.
	t.Setenv("GPU_LOCKDIR", filepath.Join(t.TempDir(), "service-lock"))
	ctx := t.Context()
	g, root, layers, stack := buildAutonomousFourLayerStack(t)
	if len(layers) != 4 {
		t.Fatalf("len(layers) = %d, want 4", len(layers))
	}
	if len(stack.CrossBoundaries) != 3 || len(stack.CrossMegakernels) != 3 {
		t.Fatalf("stack boundaries=%d megakernels=%d, want 3 and 3", len(stack.CrossBoundaries), len(stack.CrossMegakernels))
	}
	if stack.ArenaBytes == 0 {
		t.Fatal("expected non-zero packed ArenaBytes across 4-layer stack")
	}

	// Verify StageLinePCMap and AttributeMegakernelBatch on boundary 0->1.
	crossPlan0 := stack.CrossBoundaries[0]
	crossMega0 := stack.CrossMegakernels[0]
	lineToPC, pcBatch := buildAutonomousStagePCBatch(t, crossMega0)
	amap, err := StageLinePCMap(crossMega0, lineToPC)
	if err != nil {
		t.Fatalf("StageLinePCMap: %v", err)
	}
	profiles, err := AttributeMegakernelBatch(crossPlan0, crossMega0, amap, pcBatch)
	if err != nil {
		t.Fatalf("AttributeMegakernelBatch: %v", err)
	}
	if len(profiles) != 3 || profiles[0].Stalls.TotalSamples == 0 {
		t.Fatalf("unexpected StageProfiles: %+v", profiles)
	}

	// Exercise RunAutonomousFlightPipeline cold-start and warm-start on host.
	cacheDir := filepath.Join(t.TempDir(), "pgo-cache")
	pgoCache := autotune.NewVerifiedPGOCache(cacheDir)
	cacheKey := autotune.PGOCacheKey{
		Arch:          "sm_121a",
		DriverVersion: "590.48.01",
		CUDAVersion:   "13.2",
		WeightDigest:  "llama_4layer_bf16_m2_d256",
		PlanKey:       "llama_4layer_cross_m2_d256",
		ShapeBucket:   2,
	}
	loader := &flightTestLoader{}
	svc, err := cudahotswap.NewService(loader)
	if err != nil {
		t.Fatalf("cudahotswap.NewService: %v", err)
	}
	defer func() {
		if err := svc.Close(ctx); err != nil {
			t.Errorf("svc.Close: %v", err)
		}
	}()
	bucketCtrl, err := specdecode.NewAdaptiveBucketController(specdecode.AdaptiveBucketConfig{
		HysteresisSteps:       1,
		InitialAcceptanceRate: 0.50,
		InitialBucketM:        1,
	})
	if err != nil {
		t.Fatalf("NewAdaptiveBucketController: %v", err)
	}

	cfg := AutonomousPipelineConfig{
		Cache:            pgoCache,
		CacheKey:         cacheKey,
		Device:           "NVIDIA GB10",
		Arch:             "sm_121a",
		DeviceSMs:        48,
		BeamWidth:        4,
		PTXASCompiler:    &autonomousTestPTXASCompiler{},
		PTXASGate:        qualify.StaticGate{NoSpills: true, MaxRegisters: 192, MinCTAsPerSM: 1},
		MaxPTXASIters:    4,
		HotSwapService:   svc,
		BucketController: bucketCtrl,
		KernelName:       "llama_4layer_cross_m2_d256",
	}
	id := qualify.MeasuredIdentity{
		Context:   sha256.Sum256([]byte("ctx:autonomous:4layer:sm_121a")),
		Admission: sha256.Sum256([]byte("adm:autonomous:4layer:host")),
	}
	ev := testWinningHoldoutEvidence(id)
	winningCubin := []byte("CUBIN-4LAYER-CROSS-MEGA-SM121A")

	glue0 := hopper.CrossLayerGlue{
		Name:               "cross_l0_l1",
		LayerLPrefix:       "l0_",
		LayerNextPrefix:    "l1_",
		BridgeInput:        "xn",
		RetainBridgeOutput: true,
	}
	res1, err := RunAutonomousFlightPipeline(ctx, g, root, "llama_decode", hopper.BF16, cfg, AutonomousEvalRequest{
		CrossLayerPost:       layers[0].Post,
		CrossLayerNextAttnIn: layers[1].AttnIn,
		CrossLayerGlue:       glue0,
		PCBatch:              pcBatch,
		LineToPC:             lineToPC,
		Identity:             id,
		EvaluateCandidate: func(cand qualify.StructuralBeamCandidate) (qualify.HoldoutCandidateEvidence, bool) {
			if cand.Megakernel != nil && strings.Contains(cand.Handle(), "cross_l0_l1") {
				return ev, true
			}
			return qualify.HoldoutCandidateEvidence{}, false
		},
		WinningCubin:   winningCubin,
		ActiveBatch:    2,
		DraftedTokens:  4,
		AcceptedTokens: 2,
	})
	if err != nil {
		t.Fatalf("RunAutonomousFlightPipeline cold start: %v", err)
	}
	if res1.CacheHit {
		t.Fatal("expected CacheHit=false on cold start")
	}
	if len(res1.PTXASRefinements) == 0 {
		t.Fatal("expected non-empty PTXASRefinements")
	}
	for _, ref := range res1.PTXASRefinements {
		last := ref.Iterations[len(ref.Iterations)-1]
		if last.TotalSpillBytes != 0 {
			t.Fatalf("expected 0 final spills, got %d", last.TotalSpillBytes)
		}
	}
	if res1.Winner == nil || !res1.Winner.Promoted || !res1.Winner.TokensMatched {
		t.Fatalf("expected promoted winner, got %+v", res1.Winner)
	}
	if res1.PushResult == nil || !res1.PushResult.Published || res1.SelectedState.Version != 2 {
		t.Fatalf("expected hot-swap publish to v2, got push=%+v state=%+v", res1.PushResult, res1.SelectedState)
	}
	if !res1.PersistedToCache {
		t.Fatal("expected PersistedToCache=true")
	}

	// Second call: warm-start reload from VerifiedPGOCache.
	t0 := time.Now()
	res2, err := RunAutonomousFlightPipeline(ctx, g, root, "llama_decode", hopper.BF16, cfg, AutonomousEvalRequest{})
	reloadDur := time.Since(t0)
	if err != nil {
		t.Fatalf("RunAutonomousFlightPipeline warm start: %v", err)
	}
	if !res2.CacheHit {
		t.Fatal("expected CacheHit=true on warm start")
	}
	if !bytes.Equal(res2.CachedCubin, winningCubin) {
		t.Fatal("warm-start CachedCubin mismatch")
	}
	if reloadDur >= 50*time.Millisecond {
		t.Fatalf("warm-start reload took %v, want < 50ms", reloadDur)
	}

	// Verify 4-layer native CUDA harness generation.
	harnessSrc, unfusedUnit, megaUnit := buildAutonomousFourLayerNativeHarness(t, stack)
	if strings.Contains(harnessSrc, "%!") || strings.Contains(unfusedUnit, "%!") || strings.Contains(megaUnit, "%!") {
		t.Fatal("buildAutonomousFourLayerNativeHarness produced format error marker %!")
	}
	for _, want := range []string{
		"cross_l0_l1_run",
		"cross_l1_l2_run",
		"cross_l2_l3_run",
		"cross_l0_l1_mega_run",
		"cross_l1_l2_mega_run",
		"cross_l2_l3_mega_run",
		"AA_BLOCK",
		"SCREEN_BLOCK",
		"HOLDOUT_BLOCK",
		"REF_TOKENS=",
		"CAND_TOKENS=",
		"MUT_TOKENS=",
		"mutant_rejected=1",
		"AUTONOMOUS_4LAYER_NATIVE_OK",
	} {
		if !strings.Contains(harnessSrc, want) {
			t.Fatalf("buildAutonomousFourLayerNativeHarness missing %q", want)
		}
	}
}

// TestAutonomousFlightNativeSM121A validates RunAutonomousFlightPipeline
// end-to-end on real GPU hardware (sm_121a) under gpu-lock when
// CUDATEST_NATIVE=1:
//   - VerifiedPGOCache cold-miss and warm-start reload
//   - RefinePlanByPTXASLoop with qualify.NVCCPTXASCompiler
//   - #line StageLinePCMap + AttributeMegakernelBatch
//   - SelectStructuralBeamWithHoldout with real GPU CUDA-event timings over a
//     4-layer cross-layer persistent megakernel stack (3 persistent launches)
//     vs unfused 4-layer stack execution (9 launches across 3 boundaries)
//   - VerifyDigest + VerifyTokenSequence + device mutant negative control
//   - Live cudahotswap.Service CUgraphExec kernel node hot-swapping on sm_121a
func TestAutonomousFlightNativeSM121A(t *testing.T) {
	if skip := nativeAutonomousGateSkipReason(); skip != "" {
		t.Skip(skip)
	}
	if err := checkAutonomousNativeGate(os.Getenv); err != nil {
		t.Fatalf("native gate enabled but %v: run under gpu-lock", err)
	}
	nvcc, cudaVer := cudatest.NVCC(t)
	arch := autonomousNativeArch()
	cudatest.SkipUnlessTarget(t, nvcc, arch)

	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("user home dir: %v", err)
	}
	tmpBase := filepath.Join(home, "tmp", "nvidia-native")
	if err := os.MkdirAll(tmpBase, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", tmpBase, err)
	}
	dir, err := os.MkdirTemp(tmpBase, "autonomous-flight-native-")
	if err != nil {
		t.Fatalf("mkdirtemp: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })

	// Open GPU device 0 and query hardware/driver provenance.
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatalf("cuda.NewDevice(0): %v", err)
	}
	t.Cleanup(func() {
		if closeErr := dev.Close(); closeErr != nil {
			t.Errorf("dev.Close: %v", closeErr)
		}
	})
	devName, err := dev.Name()
	if err != nil {
		t.Fatalf("dev.Name: %v", err)
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatalf("dev.ComputeCapability: %v", err)
	}
	sms, err := dev.MultiprocessorCount()
	if err != nil {
		t.Fatalf("dev.MultiprocessorCount: %v", err)
	}
	optinSMem, err := dev.Attribute(driver.DevAttrMaxSharedMemoryPerBlockOptin)
	if err != nil {
		optinSMem = 99 * 1024
	}
	drvVerInt, err := driver.DriverGetVersion()
	if err != nil {
		t.Fatalf("driver.DriverGetVersion: %v", err)
	}
	drvVersionStr := fmt.Sprintf("%d.%d", drvVerInt/1000, (drvVerInt%1000)/10)
	t.Logf("provenance: device=%q arch=%s cc=%d.%d sms=%d optin_smem=%d driver=%d (%s) cuda=%s",
		devName, arch, major, minor, sms, optinSMem, drvVerInt, drvVersionStr, cudaVer)

	ctx, cancel := context.WithTimeout(t.Context(), 4*time.Minute)
	defer cancel()

	// Construct the 4-layer decoder stack (M=2, D=256) and cross-layer megakernels.
	g, root, layers, stack := buildAutonomousFourLayerStack(t)

	// Set up live cudahotswap.Service backed by nvcc + cuda.Device and bind a
	// real 1-node CUgraphExec on a device stream for live node hot-swapping.
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatalf("dev.NewStream: %v", err)
	}
	t.Cleanup(func() {
		if closeErr := stream.Close(); closeErr != nil {
			t.Errorf("stream.Close: %v", closeErr)
		}
	})
	outBuf, err := cudamem.Alloc[byte](dev, 64)
	if err != nil {
		t.Fatalf("cudamem.Alloc: %v", err)
	}
	t.Cleanup(func() {
		if freeErr := outBuf.Free(); freeErr != nil {
			t.Errorf("outBuf.Free: %v", freeErr)
		}
	})
	outView, err := cudamem.Reinterpret[float32](outBuf)
	if err != nil {
		t.Fatalf("cudamem.Reinterpret: %v", err)
	}
	if err := outView.CopyFromHost([]float32{0, 0, 0, 0}); err != nil {
		t.Fatalf("outView.CopyFromHost: %v", err)
	}

	const kernelName = "llama_4layer_cross_m2_d256"
	hsLoader := &nvccNativeHotSwapLoader{
		dev:  dev,
		nvcc: nvcc,
		arch: arch,
		dir:  filepath.Join(dir, "hotswap-compile"),
	}
	svc, err := cudahotswap.NewService(hsLoader)
	if err != nil {
		t.Fatalf("cudahotswap.NewService: %v", err)
	}
	t.Cleanup(func() {
		if closeErr := svc.Close(context.Background()); closeErr != nil {
			t.Errorf("svc.Close: %v", closeErr)
		}
	})

	// Register baseline v1 kernel (writes 1.0f to out[0]) and capture/instantiate
	// a real 1-node CUgraphExec bound to kernelName via svc.RegisterGraph.
	baseHotSwapCode := fmt.Sprintf(
		"// baseline v1\nextern \"C\" __global__ void %s(float* out) {\n\tif (threadIdx.x == 0 && blockIdx.x == 0 && out) out[0] = 1.0f;\n}\n",
		kernelName,
	)
	v1State, err := svc.Register(ctx, kernelName, cudahotswap.Source{
		Code:     baseHotSwapCode,
		Filename: kernelName + "_v1.cu",
		Entry:    kernelName,
	}, "baseline_v1")
	if err != nil {
		t.Fatalf("svc.Register baseline_v1: %v", err)
	}
	if v1State.Version != 1 {
		t.Fatalf("v1State.Version = %d, want 1", v1State.Version)
	}

	v1Pin, err := svc.Pin(kernelName)
	if err != nil {
		t.Fatalf("svc.Pin(v1): %v", err)
	}
	t.Cleanup(func() {
		if v1Pin != nil {
			_ = v1Pin.Release(context.Background())
		}
	})

	cuGraph, err := cudagraph.New(dev)
	if err != nil {
		t.Fatalf("cudagraph.New: %v", err)
	}
	t.Cleanup(func() {
		if cuGraph != nil {
			_ = cuGraph.Destroy()
		}
	})
	launchDesc, err := cudalaunch.Kernel(v1Pin.Function()).Grid(1, 1, 1).Block(256, 1, 1).Describe()
	if err != nil {
		t.Fatalf("cudalaunch.Describe: %v", err)
	}
	graphNode, err := cuGraph.AddLaunch(launchDesc, []any{outBuf})
	if err != nil {
		t.Fatalf("cuGraph.AddLaunch: %v", err)
	}
	graphExec, err := cuGraph.Instantiate()
	if err != nil {
		t.Fatalf("cuGraph.Instantiate: %v", err)
	}
	t.Cleanup(func() {
		if graphExec != nil {
			_ = graphExec.Destroy()
		}
	})

	if _, err := svc.RegisterGraph(ctx, kernelName, graphExec, graphNode, stream, driver.Dims(1, 1, 1), outBuf); err != nil {
		t.Fatalf("svc.RegisterGraph: %v", err)
	}
	// Release initial pin now that RegisterGraph holds its own generation pin.
	if err := v1Pin.Release(ctx); err != nil {
		t.Fatalf("v1Pin.Release: %v", err)
	}
	v1Pin = nil

	// Launch v1 CUgraphExec before pipeline promotion to verify initial node execution.
	w1, err := svc.LaunchGraph(ctx, graphExec)
	if w1 != nil {
		err = errors.Join(err, w1.Wait(ctx))
	}
	if err != nil {
		t.Fatalf("svc.LaunchGraph(v1): %v", err)
	}
	hostOut := make([]float32, 4)
	if err := outView.CopyToHost(hostOut); err != nil {
		t.Fatalf("outView.CopyToHost(v1): %v", err)
	}
	if hostOut[0] != 1.0 {
		t.Fatalf("initial CUgraphExec output = %v, want 1.0", hostOut[0])
	}

	// Prepare PC-sampling stall batch mapped to boundary 0->1 GEMM stage #line span.
	crossMega0 := stack.CrossMegakernels[0]
	lineToPC, pcBatch := buildAutonomousStagePCBatch(t, crossMega0)

	// Configure AutonomousPipelineConfig with VerifiedPGOCache, NVCCPTXASCompiler,
	// StaticGate, HotSwapService, and AdaptiveBucketController.
	cacheDir := filepath.Join(dir, "verified-pgo-cache")
	pgoCache := autotune.NewVerifiedPGOCache(cacheDir)
	bucketCtrl, err := specdecode.NewAdaptiveBucketController(specdecode.AdaptiveBucketConfig{
		HysteresisSteps:       1,
		InitialAcceptanceRate: 0.50,
		InitialBucketM:        1,
	})
	if err != nil {
		t.Fatalf("NewAdaptiveBucketController: %v", err)
	}

	harnessSrc, unfusedUnitSrc, megaUnitSrc := buildAutonomousFourLayerNativeHarness(t, stack)
	unfusedCubin, unfusedDigest := compileCubinO3AndVerify(t, dir, nvcc, arch, cudaVer.String(), "autonomous_4layer_unfused", unfusedUnitSrc)
	megaCubin, megaDigest := compileCubinO3AndVerify(t, dir, nvcc, arch, cudaVer.String(), "autonomous_4layer_mega", megaUnitSrc)
	t.Logf("verified 4-layer cubin digests: unfused=%x (%d bytes) cross_mega=%x (%d bytes)",
		unfusedDigest[:8], len(unfusedCubin), megaDigest[:8], len(megaCubin))

	// Mutant digest negative control: verifyArtifactDigest and VerifyDigest must reject a mutated payload/identity.
	mutantCubin := bytes.Clone(megaCubin)
	mutantCubin[len(mutantCubin)/2] ^= 0xff
	if err := verifyArtifactDigest(mutantCubin, megaDigest); err == nil {
		t.Fatal("verifyArtifactDigest failed to reject mutated cubin binary")
	}

	cacheKey := autotune.PGOCacheKey{
		Arch:          arch,
		DriverVersion: drvVersionStr,
		CUDAVersion:   cudaVer.String(),
		WeightDigest:  fmt.Sprintf("%x", unfusedDigest[:16]),
		PlanKey:       kernelName,
		ShapeBucket:   2,
	}

	var ctxBuf bytes.Buffer
	ctxBuf.WriteString("autonomous-4layer-native:")
	ctxBuf.WriteString(arch)
	ctxBuf.WriteString(":")
	ctxBuf.WriteString(cudaVer.String())
	ctxBuf.Write(unfusedDigest[:])
	ctxBuf.Write(megaDigest[:])
	id := qualify.MeasuredIdentity{
		Context:   sha256.Sum256(ctxBuf.Bytes()),
		Admission: sha256.Sum256([]byte("NATIVE_GPU_LOCK_AUTONOMOUS_4LAYER:" + devName + ":" + drvVersionStr)),
	}

	var (
		parsedHarness   parsedAutonomousHarnessOutput
		mutantRejected  bool
		graphHotSwapped atomic.Bool
	)

	// PostSwapCheck replays the live CUgraphExec node on the GPU after
	// cudahotswap.Service patches the node via CUgraphExecKernelNodeSetParams.
	postSwapCheck := func(checkCtx context.Context, st cudahotswap.State) error {
		work, lErr := svc.LaunchGraph(checkCtx, graphExec)
		if work != nil {
			lErr = errors.Join(lErr, work.Wait(checkCtx))
		}
		if lErr != nil {
			return fmt.Errorf("post-swap LaunchGraph: %w", lErr)
		}
		got := make([]float32, 4)
		if cErr := outView.CopyToHost(got); cErr != nil {
			return fmt.Errorf("post-swap CopyToHost: %w", cErr)
		}
		if got[0] != 2.0 {
			return fmt.Errorf("post-swap CUgraphExec output = %v, want 2.0 (version=%d)", got[0], st.Version)
		}
		graphHotSwapped.Store(true)
		return nil
	}

	cfg := AutonomousPipelineConfig{
		Cache:         pgoCache,
		CacheKey:      cacheKey,
		Device:        devName,
		Arch:          arch,
		DeviceSMs:     sms,
		BeamWidth:     4,
		PTXASCompiler: qualify.NVCCPTXASCompiler{NVCC: nvcc, ExtraFlags: []string{"-std=c++17", "-O3", "-lineinfo"}},
		PTXASGate: qualify.StaticGate{
			NoSpills:     true,
			MaxRegisters: 192,
			MinCTAsPerSM: 1,
		},
		MaxPTXASIters:    4,
		HotSwapService:   svc,
		BucketController: bucketCtrl,
		KernelName:       kernelName,
		PostSwapCheck:    postSwapCheck,
	}

	glue0 := hopper.CrossLayerGlue{
		Name:               "cross_l0_l1",
		LayerLPrefix:       "l0_",
		LayerNextPrefix:    "l1_",
		BridgeInput:        "xn",
		RetainBridgeOutput: true,
	}

	// EvaluateCandidate compiles and executes the 4-layer stack benchmark on the
	// GPU (3 cross-layer persistent megakernels vs 9 unfused launches across
	// boundaries 0->1, 1->2, 2->3), measures 12 interleaved blocks via CUDA
	// events, and verifies numerical and greedy token parity plus mutant rejection.
	evalCandidate := func(cand qualify.StructuralBeamCandidate) (qualify.HoldoutCandidateEvidence, bool) {
		if cand.Megakernel == nil || !strings.Contains(cand.Handle(), "cross_l0_l1") {
			return qualify.HoldoutCandidateEvidence{}, false
		}
		cuPath := filepath.Join(dir, "qualify_autonomous_4layer.cu")
		binPath := filepath.Join(dir, "qualify_autonomous_4layer")
		if wErr := os.WriteFile(cuPath, []byte(harnessSrc), 0o600); wErr != nil {
			t.Fatalf("write %s: %v", cuPath, wErr)
		}
		buildCtx, buildCancel := context.WithTimeout(ctx, 2*time.Minute)
		defer buildCancel()
		buildCmd := exec.CommandContext(buildCtx, nvcc, "-std=c++17", "-O3", "-lineinfo", "-arch="+arch, cuPath, "-o", binPath)
		if out, bErr := buildCmd.CombinedOutput(); bErr != nil {
			t.Fatalf("nvcc compile 4-layer harness (%s, CUDA %s): %v\n%s", arch, cudaVer, bErr, out)
		}

		runCtx, runCancel := context.WithTimeout(ctx, 2*time.Minute)
		defer runCancel()
		runCmd := exec.CommandContext(runCtx, binPath)
		out, rErr := runCmd.CombinedOutput()
		if rErr != nil {
			t.Fatalf("4-layer qualification binary failed (%s, CUDA %s): %v\n%s", arch, cudaVer, rErr, out)
		}
		outStr := string(out)
		if !strings.Contains(outStr, "AUTONOMOUS_4LAYER_NATIVE_OK") || !strings.Contains(outStr, "mutant_rejected=1") {
			t.Fatalf("4-layer qualification output missing expected markers:\n%s", outStr)
		}
		parsed, pErr := parseAutonomousHarnessOutput(outStr)
		if pErr != nil {
			t.Fatalf("parseAutonomousHarnessOutput: %v\n%s", pErr, outStr)
		}
		if vErr := VerifyTokenSequence(parsed.refTokens, parsed.candTokens); vErr != nil {
			t.Fatalf("VerifyTokenSequence(ref, cand): %v", vErr)
		}
		if vErr := VerifyTokenSequence(parsed.refTokens, parsed.mutTokens); vErr == nil {
			t.Fatalf("VerifyTokenSequence failed to reject mutated device token trajectory %v", parsed.mutTokens)
		}
		mutantRejected = true
		parsedHarness = parsed

		return qualify.HoldoutCandidateEvidence{
			Screen: qualify.MeasuredPair{
				Identity:  id,
				Reference: "unfused_4layer_stack",
				Candidate: cand.Handle(),
				Blocks:    parsed.screenBlocks,
			},
			HoldoutBlocks:   parsed.holdoutBlocks,
			AANullBlocks:    parsed.aaBlocks,
			ReferenceTokens: parsed.refTokens,
			CandidateTokens: parsed.candTokens,
			LogitMaxAbs:     parsed.logitMaxAbs,
		}, true
	}

	baseVerify := VerifyDigest(id)
	// Negative control on VerifyDigest(id): must reject a mutated MeasuredIdentity.
	mutID := id
	mutID.Context[0] ^= 0xff
	if err := baseVerify(qualify.MeasuredPair{
		Identity:  mutID,
		Reference: "unfused_4layer_stack",
		Candidate: "mutant_candidate",
		Blocks:    []pairedtiming.Block{{A1: 10, B1: 9, B2: 9, A2: 10}},
	}); err == nil {
		t.Fatal("VerifyDigest(id) failed to reject mutated MeasuredIdentity")
	}

	verifyFn := func(p qualify.MeasuredPair) error {
		if err := baseVerify(p); err != nil {
			return err
		}
		if err := verifyArtifactDigest(unfusedCubin, unfusedDigest); err != nil {
			return err
		}
		return verifyArtifactDigest(megaCubin, megaDigest)
	}

	// First call: Cold-miss full autonomous pipeline execution.
	res, err := RunAutonomousFlightPipeline(ctx, g, root, "llama_decode", hopper.BF16, cfg, AutonomousEvalRequest{
		CrossLayerPost:       layers[0].Post,
		CrossLayerNextAttnIn: layers[1].AttnIn,
		CrossLayerGlue:       glue0,
		PCBatch:              pcBatch,
		LineToPC:             lineToPC,
		Identity:             id,
		EvaluateCandidate:    evalCandidate,
		Verify:               verifyFn,
		WinningCubin:         megaCubin,
		KernelName:           kernelName,
		PostSwapCheck:        postSwapCheck,
		ActiveBatch:          2,
		DraftedTokens:        4,
		AcceptedTokens:       2,
	})
	if err != nil {
		t.Fatalf("RunAutonomousFlightPipeline (cold start): %v", err)
	}
	if res.CacheHit {
		t.Fatal("expected res.CacheHit == false on first call")
	}
	if len(res.PTXASRefinements) == 0 {
		t.Fatal("expected non-empty res.PTXASRefinements")
	}
	totalSpills := 0
	for i, ref := range res.PTXASRefinements {
		if len(ref.Iterations) == 0 {
			t.Fatalf("refinement %d has 0 iterations", i)
		}
		last := ref.Iterations[len(ref.Iterations)-1]
		totalSpills += last.TotalSpillBytes
		if last.TotalSpillBytes != 0 {
			t.Fatalf("refinement %d final TotalSpillBytes = %d, want 0", i, last.TotalSpillBytes)
		}
	}
	if len(res.StageProfiles) == 0 || res.StageProfiles[0].Stalls.TotalSamples == 0 {
		t.Fatalf("expected non-empty StageProfiles with attributed stalls, got %+v", res.StageProfiles)
	}
	if res.Winner == nil || !res.Winner.Promoted || !res.Winner.TokensMatched {
		t.Fatalf("expected promoted Winner with matched tokens, got %+v", res.Winner)
	}
	if !mutantRejected {
		t.Fatal("expected device mutant candidate to be evaluated and rejected")
	}
	hotSwapped := graphHotSwapped.Load() && res.PushResult != nil && res.PushResult.Published && res.SelectedState.Version == 2
	if !hotSwapped {
		t.Fatalf("expected live CUgraphExec hot-swap to version 2, got graphHotSwapped=%v push=%+v state=%+v",
			graphHotSwapped.Load(), res.PushResult, res.SelectedState)
	}
	if !res.PersistedToCache {
		t.Fatal("expected res.PersistedToCache == true")
	}

	// Second call: Warm-start reload from VerifiedPGOCache with the same AutonomousPipelineConfig.
	reloadStart := time.Now()
	res2, err := RunAutonomousFlightPipeline(ctx, g, root, "llama_decode", hopper.BF16, cfg, AutonomousEvalRequest{
		KernelName:    kernelName,
		PostSwapCheck: postSwapCheck,
	})
	reloadDur := time.Since(reloadStart)
	if err != nil {
		t.Fatalf("RunAutonomousFlightPipeline (warm start): %v", err)
	}
	if !res2.CacheHit {
		t.Fatal("expected res2.CacheHit == true on second call")
	}
	if err := verifyArtifactDigest(res2.CachedCubin, megaDigest); err != nil {
		t.Fatalf("verifyArtifactDigest(res2.CachedCubin): %v", err)
	}

	// Also measure direct VerifiedPGOCache.LoadVerified latency.
	directStart := time.Now()
	_, loadedCubin, err := pgoCache.LoadVerified(cacheKey)
	directReloadDur := time.Since(directStart)
	if err != nil {
		t.Fatalf("pgoCache.LoadVerified: %v", err)
	}
	if err := verifyArtifactDigest(loadedCubin, megaDigest); err != nil {
		t.Fatalf("verifyArtifactDigest(loadedCubin): %v", err)
	}
	reloadUs := float64(directReloadDur.Nanoseconds()) / 1e3
	pipelineWarmUs := float64(reloadDur.Nanoseconds()) / 1e3

	aaRatio := 1.0
	if res.Winner.AANullObserved != nil {
		aaRatio = res.Winner.AANullObserved.Ratio
	}

	t.Logf("CLAIM: native-qualified on %s (%s, %d SMs), driver %s, CUDA %s; workload TestAutonomousFlightNativeSM121A (4-layer cross-layer persistent megakernel stack M=2 D=256 3-launches vs 9-launch unfused baseline, RefinePlanByPTXASLoop + StageLinePCMap + SelectStructuralBeamWithHoldout + CUgraphExec hot-swap + VerifiedPGOCache warm reload); metric status=%s winner=%s spills=%d screen_ratio=%.4f holdout_ratio=%.4f aa_null_ratio=%.4f sign_p=%.4f logit_max_abs=%.6f tokens_matched=%v mutant_rejected=%v hot_swapped=%v cache_reload_us=%.2f pipeline_warm_us=%.2f",
		devName, arch, sms, drvVersionStr, cudaVer,
		res.Winner.Status, res.Winner.Winner.Handle(), totalSpills,
		res.Winner.Screen.Ratio, res.Winner.Holdout.Ratio, aaRatio, res.Winner.OneSidedSignP,
		parsedHarness.logitMaxAbs, res.Winner.TokensMatched, mutantRejected, hotSwapped,
		reloadUs, pipelineWarmUs)
}

// nvccNativeHotSwapLoader implements cudahotswap.Loader on a real GPU using
// nvcc (--ptx and --cubin) and cuda.LoadModule so full Hopper/Blackwell
// translation units and entry wrappers compile and hot-swap live CUgraphExec
// nodes on sm_121a.
type nvccNativeHotSwapLoader struct {
	dev  *cuda.Device
	nvcc string
	arch string
	dir  string
	seq  atomic.Int64
}

func (l *nvccNativeHotSwapLoader) Compile(ctx context.Context, source cudahotswap.Source) (*cudahotswap.Compilation, error) {
	if err := os.MkdirAll(l.dir, 0o700); err != nil {
		return nil, err
	}
	id := l.seq.Add(1)
	code := source.Code
	// When compiling the promoted cross-layer candidate (which embeds the
	// persistent megakernel stages), distinguish its entry marker value (2.0f)
	// from the baseline (1.0f) while keeping the identical (float* out) ABI so
	// CUgraphExec LaunchGraph verifies the live node hot-swap on device.
	if strings.Contains(code, "persistent") || strings.Contains(code, "_mega") || strings.Contains(code, "// candidate:") {
		code = strings.Replace(code, "if (out) out[0] = 1.0f;", "if (threadIdx.x == 0 && blockIdx.x == 0 && out) out[0] = 2.0f;", 1)
	}

	cuPath := filepath.Join(l.dir, fmt.Sprintf("hs_%d.cu", id))
	ptxPath := filepath.Join(l.dir, fmt.Sprintf("hs_%d.ptx", id))
	if err := os.WriteFile(cuPath, []byte(code), 0o600); err != nil {
		return nil, err
	}
	cmd := exec.CommandContext(ctx, l.nvcc, "-std=c++17", "-O3", "-arch="+l.arch, "--ptx", cuPath, "-o", ptxPath)
	if out, err := cmd.CombinedOutput(); err != nil {
		return nil, fmt.Errorf("nvcc --ptx (%s): %w: %s", l.arch, err, strings.TrimSpace(string(out)))
	}
	ptxBytes, err := os.ReadFile(ptxPath)
	if err != nil {
		return nil, err
	}
	sum := sha256.Sum256(ptxBytes)
	return &cudahotswap.Compilation{
		PTX:   string(ptxBytes),
		Key:   fmt.Sprintf("nvcc-%s-%x", l.arch, sum[:8]),
		Entry: source.Entry,
	}, nil
}

func (l *nvccNativeHotSwapLoader) Load(ctx context.Context, c *cudahotswap.Compilation) (cudahotswap.Kernel, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	sig, err := cudalaunch.SignatureFromPTX(c.PTX, c.Entry)
	if err != nil {
		return nil, err
	}
	mod, err := cuda.LoadModule(l.dev, []byte(c.PTX))
	if err != nil {
		return nil, err
	}
	fn, err := mod.Function(c.Entry)
	if err != nil {
		_ = mod.Close()
		return nil, err
	}
	return &nvccNativeHotSwapKernel{
		dev:       l.dev,
		module:    mod,
		function:  fn,
		signature: sig,
	}, nil
}

type nvccNativeHotSwapKernel struct {
	dev       *cuda.Device
	module    *cuda.Module
	function  *cuda.Kernel
	signature cudalaunch.Signature
}

func (k *nvccNativeHotSwapKernel) Raw() driver.Function {
	if k == nil || k.function == nil {
		return 0
	}
	return k.function.Raw()
}

func (k *nvccNativeHotSwapKernel) Launch(cfg cudalaunch.Config, args ...any) error {
	if k == nil || k.function == nil {
		return errors.New("launch kernel: nil function")
	}
	return cudalaunch.OwnedKernel(k.function).Config(cfg).WithSignature(k.signature).Launch(args...)
}

func (k *nvccNativeHotSwapKernel) Wait(ctx context.Context, stream *cuda.Stream) error {
	if k == nil || k.dev == nil {
		return errors.New("wait kernel: nil device")
	}
	if stream != nil {
		return stream.Synchronize(ctx)
	}
	return k.dev.WithContext(driver.CtxSynchronize)
}

func (k *nvccNativeHotSwapKernel) Close() error {
	if k == nil || k.module == nil {
		return nil
	}
	return k.module.Close()
}

func compileCubinO3AndVerify(t *testing.T, dir, nvcc, arch, cudaVer, prefix, src string) ([]byte, [32]byte) {
	t.Helper()
	cuPath := filepath.Join(dir, prefix+".cu")
	cubinPath := filepath.Join(dir, prefix+".cubin")
	if err := os.WriteFile(cuPath, []byte(src), 0o600); err != nil {
		t.Fatalf("write %s: %v", cuPath, err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, nvcc, "-std=c++17", "-O3", "-lineinfo", "--cubin", "-arch="+arch, cuPath, "-o", cubinPath)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("nvcc --cubin %s (%s, CUDA %s): %v\n%s", prefix, arch, cudaVer, err, out)
	}
	cubin, err := os.ReadFile(cubinPath)
	if err != nil {
		t.Fatalf("read %s: %v", cubinPath, err)
	}
	want := sha256.Sum256(cubin)
	if err := verifyArtifactDigest(cubin, want); err != nil {
		t.Fatalf("verifyArtifactDigest(%s): %v", prefix, err)
	}
	return cubin, want
}

func verifyArtifactDigest(data []byte, want [32]byte) error {
	if len(data) == 0 {
		return errors.New("verify artifact digest: empty data")
	}
	if want == ([32]byte{}) {
		return errors.New("verify artifact digest: zero expected digest")
	}
	if got := sha256.Sum256(data); got != want {
		return fmt.Errorf("verify artifact digest: sha256=%x, want %x", got, want)
	}
	return nil
}

func buildAutonomousStagePCBatch(t *testing.T, mega hopper.Megakernel) (map[int][2]uint64, cuptipcsampling.Batch) {
	t.Helper()
	_, spans, err := mega.LineInfoSource()
	if err != nil || len(spans) == 0 {
		t.Fatalf("mega.LineInfoSource: len=%d err=%v", len(spans), err)
	}
	lineToPC := make(map[int][2]uint64, len(spans))
	for i, sp := range spans {
		base := uint64(i+1) * 0x1000
		lineToPC[sp.StartLine] = [2]uint64{base, base + 0x200}
	}
	batch := cuptipcsampling.Batch{
		PCs: []cuptipcsampling.PC{
			{
				Name:   mega.Name,
				Offset: 0x1040, // Stage 0 (GEMM)
				Reasons: []cuptipcsampling.StallSamples{
					{Index: 5, Samples: 220}, // LongScoreboard
				},
			},
			{
				Name:   mega.Name,
				Offset: 0x2040, // Stage 1 (RMSNorm)
				Reasons: []cuptipcsampling.StallSamples{
					{Index: 3, Samples: 60}, // MIOThrottle
				},
			},
		},
	}
	return lineToPC, batch
}

// buildAutonomousFourLayerStack constructs a 4-layer decoder stack at M=2, D=256
// where each layer has post (MatMul(attn, w_down) + residual x -> RMSNorm -> Cast(BF16) -> xn)
// and attnIn (MatMul(xn, w_qkv) + b_qkv -> qkv), and fuses the 3 inter-layer
// boundaries (0->1, 1->2, 2->3) into persistent cross-layer megakernels with
// shared arena packing via hopper.FuseDecoderLayerStack.
func buildAutonomousFourLayerStack(t *testing.T) (*egraph.EGraph, egraph.Id, []hopper.DecoderLayerPlans, *hopper.CrossLayerStack) {
	t.Helper()
	const (
		numLayers = 4
		m         = 2
		d         = 256
		eps       = 1e-5
	)

	var (
		firstPostGraph *egraph.EGraph
		firstPostRoot  egraph.Id
		layers         = make([]hopper.DecoderLayerPlans, numLayers)
	)

	for l := range numLayers {
		// Layer[l].Post: MatMul(attn, w_down) + x -> RMSNorm(next_norm) -> Cast(BF16) -> xn
		bPost := hopper.NewBuilder()
		attn := bPost.Input("attn", hopper.BF16, m, d)
		x := bPost.Input("x", hopper.F32, m, d)
		wDown := bPost.Input("w_down", hopper.BF16, d, d)
		nextNorm := bPost.Input("next_norm", hopper.F32, d)
		down := bPost.Map(hopper.MustLambda("(add $0 $1)"), bPost.MatMul(attn, wDown), x)
		xnOut := bPost.Cast(hopper.BF16, bPost.RMSNorm(down, nextNorm, d, eps))

		if l == 0 {
			firstPostGraph = bPost.G.Clone()
			firstPostRoot = xnOut
		}

		postExpr, err := hopper.Optimize(bPost.G, xnOut)
		if err != nil {
			t.Fatalf("layer %d optimize post: %v", l, err)
		}
		postPlan, err := hopper.Compile(postExpr, fmt.Sprintf("l%d_post", l), hopper.BF16)
		if err != nil {
			t.Fatalf("layer %d compile post: %v", l, err)
		}
		if len(postPlan.Kernels) != 2 {
			t.Fatalf("layer %d post has %d kernels, want 2", l, len(postPlan.Kernels))
		}
		postPlan.Kernels[0].GEMM.Name = fmt.Sprintf("l%d_down_proj", l)
		postPlan.Kernels[0].GEMM.RowsPerBlock = 32
		postPlan.Kernels[0].GEMM.Warps = 8
		postPlan.Kernels[1].Row.Name = fmt.Sprintf("l%d_post_norm", l)

		// Layer[l].AttnIn: MatMul(xn, w_qkv) + b_qkv -> qkv
		bAttn := hopper.NewBuilder()
		xnIn := bAttn.Input("xn", hopper.BF16, m, d)
		wQKV := bAttn.Input("w_qkv", hopper.BF16, d, d)
		bQKV := bAttn.Input("b_qkv", hopper.F32, d)
		qkv := bAttn.Map(hopper.MustLambda("(add $0 $1)"), bAttn.MatMul(xnIn, wQKV), bQKV)

		attnExpr, err := hopper.Optimize(bAttn.G, qkv)
		if err != nil {
			t.Fatalf("layer %d optimize attnIn: %v", l, err)
		}
		attnPlan, err := hopper.Compile(attnExpr, fmt.Sprintf("l%d_attn_in", l), hopper.F32)
		if err != nil {
			t.Fatalf("layer %d compile attnIn: %v", l, err)
		}
		if len(attnPlan.Kernels) != 1 {
			t.Fatalf("layer %d attnIn has %d kernels, want 1", l, len(attnPlan.Kernels))
		}
		attnPlan.Kernels[0].GEMM.Name = fmt.Sprintf("l%d_qkv_proj", l)
		attnPlan.Kernels[0].GEMM.RowsPerBlock = 32
		attnPlan.Kernels[0].GEMM.Warps = 8

		layers[l] = hopper.DecoderLayerPlans{
			LayerIndex: l,
			AttnIn:     attnPlan,
			Post:       postPlan,
		}
	}

	stack, err := hopper.FuseDecoderLayerStack(layers, hopper.CrossLayerStackConfig{
		NamePrefix:         "cross",
		BridgeInput:        "xn",
		RetainBridgeOutput: true,
		Mega:               true,
		PackArena:          true,
	})
	if err != nil {
		t.Fatalf("FuseDecoderLayerStack: %v", err)
	}
	return firstPostGraph, firstPostRoot, layers, stack
}

// buildAutonomousFourLayerNativeHarness emits a self-contained CUDA C++
// benchmark binary that runs the 4-layer stack across boundaries (0->1), (1->2),
// and (2->3):
//   - Unfused baseline: 3 launches per boundary = 9 sequential kernel launches
//   - Fused cross-layer megakernels: 1 persistent launch per boundary = 3 launches
//   - Verifies numerical accuracy against the exact 9-stage CPU reference
//   - Emits greedy token IDs (REF_TOKENS, CAND_TOKENS) and mutant token IDs (MUT_TOKENS)
//   - Runs 3 A/A null blocks + 3 ABBA screen blocks + 6 ABBA holdout blocks via CUDA events
func buildAutonomousFourLayerNativeHarness(t *testing.T, stack *hopper.CrossLayerStack) (fullHarness, unfusedUnit, megaUnit string) {
	t.Helper()
	const (
		numBoundaries = 3
		m             = 2
		d             = 256
	)
	if len(stack.CrossBoundaries) != numBoundaries || len(stack.CrossMegakernels) != numBoundaries {
		t.Fatalf("stack has %d boundaries and %d megakernels, want %d",
			len(stack.CrossBoundaries), len(stack.CrossMegakernels), numBoundaries)
	}

	stdHeader := "#include <cuda_runtime.h>\n#include <cstdio>\n#include <cmath>\n#include <cstdint>\n#include <cstring>\n"

	var (
		unfusedBuilder strings.Builder
		megaBuilder    strings.Builder
		fullBuilder    strings.Builder
	)
	unfusedBuilder.WriteString(stdHeader)
	megaBuilder.WriteString(stdHeader)
	fullBuilder.WriteString(stdHeader)

	for bIdx := range numBoundaries {
		plan := stack.CrossBoundaries[bIdx]
		mega := stack.CrossMegakernels[bIdx]

		uSrc, err := plan.Source()
		if err != nil {
			t.Fatalf("boundary %d plan.Source: %v", bIdx, err)
		}
		mLineSrc, _, err := mega.LineInfoSource()
		if err != nil {
			t.Fatalf("boundary %d mega.LineInfoSource: %v", bIdx, err)
		}
		mCUDA, _, err := mega.LineInfoCUDA()
		if err != nil {
			t.Fatalf("boundary %d mega.LineInfoCUDA: %v", bIdx, err)
		}
		mLaunch, err := mega.Launcher()
		if err != nil {
			t.Fatalf("boundary %d mega.Launcher: %v", bIdx, err)
		}

		unfusedBuilder.WriteString(uSrc + "\n")
		megaBuilder.WriteString(mLineSrc + "\n")
		fullBuilder.WriteString(uSrc + "\n")
		fullBuilder.WriteString(mCUDA + "\n" + mLaunch + "\n")
	}

	// Generate deterministic inputs and weights across all 3 boundaries (4 layers)
	// and compute the exact 9-stage CPU reference output.
	rng := rand.New(rand.NewPCG(20261031, 4121))
	attn0BF16 := randAutonomousBF16(rng, m*d, 0.8)
	x0F32 := make([]float32, m*d)
	for i := range x0F32 {
		x0F32[i] = float32(rng.NormFloat64() * 0.5)
	}

	var (
		wDownBF16   [numBoundaries][]uint16
		nextNormF32 [numBoundaries][]float32
		wQKVBF16    [numBoundaries][]uint16
		bQKVF32     [numBoundaries][]float32
	)

	curAttnBF16 := attn0BF16
	curXF32 := x0F32

	for bIdx := range numBoundaries {
		wDownBF16[bIdx] = randAutonomousBF16(rng, d*d, 0.22)
		nextNormF32[bIdx] = make([]float32, d)
		for i := range nextNormF32[bIdx] {
			nextNormF32[bIdx][i] = 0.75 + 0.25*float32(rng.Float64())
		}
		wQKVBF16[bIdx] = randAutonomousBF16(rng, d*d, 0.22)
		bQKVF32[bIdx] = make([]float32, d)
		for i := range bQKVF32[bIdx] {
			bQKVF32[bIdx][i] = float32(rng.NormFloat64() * 0.15)
		}

		plan := stack.CrossBoundaries[bIdx]
		g0, err := plan.Kernels[0].GEMM.Norm()
		if err != nil {
			t.Fatalf("boundary %d stage 0 GEMM.Norm: %v", bIdx, err)
		}
		t1Out, err := g0.Reference(m, curAttnBF16, hopper.GEMMWeights{BF16: wDownBF16[bIdx]}, [][]float32{curXF32})
		if err != nil {
			t.Fatalf("boundary %d stage 0 Reference: %v", bIdx, err)
		}

		rk1 := *plan.Kernels[1].Row
		t1Name := plan.Temps[0].Name
		xnName := plan.Temps[1].Name
		normName := plan.Inputs[3].Name
		rowBufs := map[string][]float64{
			t1Name:   autonomousF64s(t1Out),
			normName: autonomousF64s(nextNormF32[bIdx]),
			xnName:   make([]float64, m*d),
		}
		if err := rk1.Reference(m, rowBufs); err != nil {
			t.Fatalf("boundary %d stage 1 RowKernel.Reference: %v", bIdx, err)
		}
		nextAttnBF16 := make([]uint16, m*d)
		for i, v := range rowBufs[xnName] {
			nextAttnBF16[i] = hopper.BF16Bits(float32(v))
		}

		g2, err := plan.Kernels[2].GEMM.Norm()
		if err != nil {
			t.Fatalf("boundary %d stage 2 GEMM.Norm: %v", bIdx, err)
		}
		nextXF32, err := g2.Reference(m, nextAttnBF16, hopper.GEMMWeights{BF16: wQKVBF16[bIdx]}, [][]float32{bQKVF32[bIdx]})
		if err != nil {
			t.Fatalf("boundary %d stage 2 Reference: %v", bIdx, err)
		}

		curAttnBF16 = nextAttnBF16
		curXF32 = nextXF32
	}
	wantOut := curXF32
	readyBytes := stack.CrossMegakernels[0].ReadyBytes()

	fmt.Fprintf(&fullBuilder, `
#define CK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::printf("CUDA error %%s at line %%d\n", cudaGetErrorString(err__), __LINE__); \
        return 2; \
    } \
} while (0)

static const uint16_t h_attn0[] = {%s};
static const uint32_t h_x0_bits[] = {%s};

static const uint16_t h_w_down_0[] = {%s};
static const uint32_t h_norm_0_bits[] = {%s};
static const uint16_t h_w_qkv_1[] = {%s};
static const uint32_t h_b_qkv_1_bits[] = {%s};

static const uint16_t h_w_down_1[] = {%s};
static const uint32_t h_norm_1_bits[] = {%s};
static const uint16_t h_w_qkv_2[] = {%s};
static const uint32_t h_b_qkv_2_bits[] = {%s};

static const uint16_t h_w_down_2[] = {%s};
static const uint32_t h_norm_2_bits[] = {%s};
static const uint16_t h_w_qkv_3[] = {%s};
static const uint32_t h_b_qkv_3_bits[] = {%s};

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
    uint16_t *d_attn0 = nullptr;
    float *d_x0 = nullptr;

    uint16_t *d_w_down[3] = {};
    float *d_norm[3] = {};
    uint16_t *d_w_qkv[3] = {};
    float *d_b_qkv[3] = {};

    float *d_t1_ref[3] = {}, *d_t1_cand[3] = {};
    uint16_t *d_xn_ref[3] = {}, *d_xn_cand[3] = {};
    float *d_out_ref[3] = {}, *d_out_cand[3] = {};
    unsigned long long *d_ready[3] = {};

    CK(cudaMalloc(&d_attn0, sizeof(h_attn0)));
    CK(cudaMalloc(&d_x0, sizeof(h_x0_bits)));
    CK(cudaMemcpy(d_attn0, h_attn0, sizeof(h_attn0), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_x0, h_x0_bits, sizeof(h_x0_bits), cudaMemcpyHostToDevice));

    const uint16_t* h_wd[3] = {h_w_down_0, h_w_down_1, h_w_down_2};
    const uint32_t* h_nm[3] = {h_norm_0_bits, h_norm_1_bits, h_norm_2_bits};
    const uint16_t* h_wq[3] = {h_w_qkv_1, h_w_qkv_2, h_w_qkv_3};
    const uint32_t* h_bq[3] = {h_b_qkv_1_bits, h_b_qkv_2_bits, h_b_qkv_3_bits};

    for (int b = 0; b < 3; ++b) {
        CK(cudaMalloc(&d_w_down[b], D * D * sizeof(uint16_t)));
        CK(cudaMalloc(&d_norm[b], D * sizeof(float)));
        CK(cudaMalloc(&d_w_qkv[b], D * D * sizeof(uint16_t)));
        CK(cudaMalloc(&d_b_qkv[b], D * sizeof(float)));

        CK(cudaMemcpy(d_w_down[b], h_wd[b], D * D * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_norm[b], h_nm[b], D * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_w_qkv[b], h_wq[b], D * D * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_b_qkv[b], h_bq[b], D * sizeof(float), cudaMemcpyHostToDevice));

        CK(cudaMalloc(&d_t1_ref[b], M * D * sizeof(float)));
        CK(cudaMalloc(&d_t1_cand[b], M * D * sizeof(float)));
        CK(cudaMalloc(&d_xn_ref[b], M * D * sizeof(uint16_t)));
        CK(cudaMalloc(&d_xn_cand[b], M * D * sizeof(uint16_t)));
        CK(cudaMalloc(&d_out_ref[b], M * D * sizeof(float)));
        CK(cudaMalloc(&d_out_cand[b], M * D * sizeof(float)));
        CK(cudaMalloc(&d_ready[b], %d));
        CK(cudaMemset(d_ready[b], 0, %d));
    }

    unsigned int epoch[3] = {1u, 1u, 1u};
    auto runUnfused4Layer = [&]() {
        cross_l0_l1_run(d_attn0, d_w_down[0], d_x0, d_norm[0], d_w_qkv[0], d_b_qkv[0], d_t1_ref[0], d_xn_ref[0], d_out_ref[0], 0);
        cross_l1_l2_run(d_xn_ref[0], d_w_down[1], d_out_ref[0], d_norm[1], d_w_qkv[1], d_b_qkv[1], d_t1_ref[1], d_xn_ref[1], d_out_ref[1], 0);
        cross_l2_l3_run(d_xn_ref[1], d_w_down[2], d_out_ref[1], d_norm[2], d_w_qkv[2], d_b_qkv[2], d_t1_ref[2], d_xn_ref[2], d_out_ref[2], 0);
    };
    auto runFusedMega4Layer = [&]() {
        cross_l0_l1_mega_run(d_attn0, d_w_down[0], d_x0, d_norm[0], d_w_qkv[0], d_b_qkv[0], d_t1_cand[0], d_xn_cand[0], d_out_cand[0], d_ready[0], epoch[0]++, 0);
        cross_l1_l2_mega_run(d_xn_cand[0], d_w_down[1], d_out_cand[0], d_norm[1], d_w_qkv[1], d_b_qkv[1], d_t1_cand[1], d_xn_cand[1], d_out_cand[1], d_ready[1], epoch[1]++, 0);
        cross_l2_l3_mega_run(d_xn_cand[1], d_w_down[2], d_out_cand[1], d_norm[2], d_w_qkv[2], d_b_qkv[2], d_t1_cand[2], d_xn_cand[2], d_out_cand[2], d_ready[2], epoch[2]++, 0);
    };

    runUnfused4Layer();
    runFusedMega4Layer();
    CK(cudaDeviceSynchronize());

    float h_ref[M * D] = {};
    float h_cand[M * D] = {};
    CK(cudaMemcpy(h_ref, d_out_ref[2], sizeof(h_ref), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_cand, d_out_cand[2], sizeof(h_cand), cudaMemcpyDeviceToHost));

    float logit_max_abs = 0.0f;
    for (int i = 0; i < M * D; ++i) {
        float w = 0.0f;
        std::memcpy(&w, &want_out_bits[i], sizeof(float));
        float er = std::fabs(h_ref[i] - w);
        float ec = std::fabs(h_cand[i] - w);
        float dc = std::fabs(h_ref[i] - h_cand[i]);
        if (dc > logit_max_abs) logit_max_abs = dc;
        if (std::isnan(h_ref[i]) || std::isnan(h_cand[i]) || er > 5e-2f || ec > 5e-2f) {
            std::printf("FAIL 4layer idx=%%d ref=%%f cand=%%f want=%%f\n", i, h_ref[i], h_cand[i], w);
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

    // Device mutant control: perturb final boundary b_qkv[2] so an alternative column wins argmax.
    float h_b_mut[D] = {};
    for (int c = 0; c < D; ++c) {
        std::memcpy(&h_b_mut[c], &h_b_qkv_3_bits[c], sizeof(float));
    }
    int win0 = argmax_row(&h_ref[0], D);
    int alt0 = (win0 + 1) %% D;
    h_b_mut[alt0] += 100.0f;
    CK(cudaMemcpy(d_b_qkv[2], h_b_mut, sizeof(h_b_mut), cudaMemcpyHostToDevice));
    runFusedMega4Layer();
    CK(cudaDeviceSynchronize());

    float h_mut[M * D] = {};
    CK(cudaMemcpy(h_mut, d_out_cand[2], sizeof(h_mut), cudaMemcpyDeviceToHost));
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

    // Restore clean b_qkv[2] before timing blocks.
    CK(cudaMemcpy(d_b_qkv[2], h_b_qkv_3_bits, sizeof(h_b_qkv_3_bits), cudaMemcpyHostToDevice));

    cudaEvent_t ev0, ev1;
    CK(cudaEventCreate(&ev0));
    CK(cudaEventCreate(&ev1));
    const int reps = 40;

    for (int i = 0; i < 8; ++i) {
        runUnfused4Layer();
        runFusedMega4Layer();
    }
    CK(cudaDeviceSynchronize());

    for (int blk = 0; blk < 3; ++blk) {
        float a1 = 0, b1 = 0, b2 = 0, a2 = 0;
        if (measure_abba_block(ev0, ev1, reps, runUnfused4Layer, runUnfused4Layer, &a1, &b1, &b2, &a2) != 0) return 5;
        std::printf("AA_BLOCK %%d a1=%%.6f b1=%%.6f b2=%%.6f a2=%%.6f\n", blk, a1 * 1e6, b1 * 1e6, b2 * 1e6, a2 * 1e6);
    }
    for (int blk = 0; blk < 3; ++blk) {
        float a1 = 0, b1 = 0, b2 = 0, a2 = 0;
        if (measure_abba_block(ev0, ev1, reps, runUnfused4Layer, runFusedMega4Layer, &a1, &b1, &b2, &a2) != 0) return 6;
        std::printf("SCREEN_BLOCK %%d a1=%%.6f b1=%%.6f b2=%%.6f a2=%%.6f\n", blk, a1 * 1e6, b1 * 1e6, b2 * 1e6, a2 * 1e6);
    }
    for (int blk = 0; blk < 6; ++blk) {
        float a1 = 0, b1 = 0, b2 = 0, a2 = 0;
        if (measure_abba_block(ev0, ev1, reps, runUnfused4Layer, runFusedMega4Layer, &a1, &b1, &b2, &a2) != 0) return 7;
        std::printf("HOLDOUT_BLOCK %%d a1=%%.6f b1=%%.6f b2=%%.6f a2=%%.6f\n", blk, a1 * 1e6, b1 * 1e6, b2 * 1e6, a2 * 1e6);
    }

    std::printf("AUTONOMOUS_4LAYER_NATIVE_OK\n");
    return 0;
}
`,
		autonomousFormatU16Array(attn0BF16),
		autonomousCF32s(x0F32),
		autonomousFormatU16Array(wDownBF16[0]),
		autonomousCF32s(nextNormF32[0]),
		autonomousFormatU16Array(wQKVBF16[0]),
		autonomousCF32s(bQKVF32[0]),
		autonomousFormatU16Array(wDownBF16[1]),
		autonomousCF32s(nextNormF32[1]),
		autonomousFormatU16Array(wQKVBF16[1]),
		autonomousCF32s(bQKVF32[1]),
		autonomousFormatU16Array(wDownBF16[2]),
		autonomousCF32s(nextNormF32[2]),
		autonomousFormatU16Array(wQKVBF16[2]),
		autonomousCF32s(bQKVF32[2]),
		autonomousCF32s(wantOut),
		m, d,
		readyBytes, readyBytes,
	)

	return fullBuilder.String(), unfusedBuilder.String(), megaBuilder.String()
}

type parsedAutonomousHarnessOutput struct {
	deviceLine    string
	aaBlocks      []pairedtiming.Block
	screenBlocks  []pairedtiming.Block
	holdoutBlocks []pairedtiming.Block
	refTokens     []int
	candTokens    []int
	mutTokens     []int
	logitMaxAbs   float64
}

func parseAutonomousHarnessOutput(out string) (parsedAutonomousHarnessOutput, error) {
	var p parsedAutonomousHarnessOutput
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		switch {
		case strings.HasPrefix(line, "device="):
			p.deviceLine = line
		case strings.HasPrefix(line, "AA_BLOCK "):
			b, err := parseAutonomousTimingBlock(line)
			if err != nil {
				return p, err
			}
			p.aaBlocks = append(p.aaBlocks, b)
		case strings.HasPrefix(line, "SCREEN_BLOCK "):
			b, err := parseAutonomousTimingBlock(line)
			if err != nil {
				return p, err
			}
			p.screenBlocks = append(p.screenBlocks, b)
		case strings.HasPrefix(line, "HOLDOUT_BLOCK "):
			b, err := parseAutonomousTimingBlock(line)
			if err != nil {
				return p, err
			}
			p.holdoutBlocks = append(p.holdoutBlocks, b)
		case strings.HasPrefix(line, "REF_TOKENS="):
			toks, err := parseAutonomousTokenCSV(strings.TrimPrefix(line, "REF_TOKENS="))
			if err != nil {
				return p, err
			}
			p.refTokens = toks
		case strings.HasPrefix(line, "CAND_TOKENS="):
			toks, err := parseAutonomousTokenCSV(strings.TrimPrefix(line, "CAND_TOKENS="))
			if err != nil {
				return p, err
			}
			p.candTokens = toks
		case strings.HasPrefix(line, "MUT_TOKENS="):
			toks, err := parseAutonomousTokenCSV(strings.TrimPrefix(line, "MUT_TOKENS="))
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

func parseAutonomousTimingBlock(line string) (pairedtiming.Block, error) {
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

func parseAutonomousTokenCSV(s string) ([]int, error) {
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

func randAutonomousBF16(rng *rand.Rand, n int, scale float32) []uint16 {
	out := make([]uint16, n)
	for i := range out {
		v := float32(rng.NormFloat64()) * scale
		out[i] = hopper.BF16Bits(v)
	}
	return out
}

func autonomousF64s(xs []float32) []float64 {
	out := make([]float64, len(xs))
	for i, x := range xs {
		out[i] = float64(x)
	}
	return out
}

func autonomousFormatU16Array(xs []uint16) string {
	var sb strings.Builder
	for i, x := range xs {
		if i > 0 {
			sb.WriteString(", ")
		}
		fmt.Fprintf(&sb, "0x%04xu", x)
	}
	return sb.String()
}

func autonomousCF32s(xs []float32) string {
	var sb strings.Builder
	for i, x := range xs {
		if i > 0 {
			sb.WriteString(", ")
		}
		fmt.Fprintf(&sb, "0x%08xu", math.Float32bits(x))
	}
	return sb.String()
}
