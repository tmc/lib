package specdecode

import (
	"context"
	"errors"
	"math"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

type bucketTestKernel struct {
	id         int
	lastLaunch cudalaunch.Config
	launches   int
}

func (k *bucketTestKernel) Launch(cfg cudalaunch.Config, _ ...any) error {
	k.lastLaunch = cfg
	k.launches++
	return nil
}

func (k *bucketTestKernel) Wait(context.Context, *cuda.Stream) error { return nil }
func (k *bucketTestKernel) Raw() driver.Function                     { return driver.Function(k.id) }
func (k *bucketTestKernel) Close() error                             { return nil }

type bucketTestLoader struct {
	nextID   int
	compiles int
	loads    int
	kernels  map[int]*bucketTestKernel
}

func newBucketTestLoader() *bucketTestLoader {
	return &bucketTestLoader{kernels: make(map[int]*bucketTestKernel)}
}

func (l *bucketTestLoader) Compile(_ context.Context, src cudahotswap.Source) (*cudahotswap.Compilation, error) {
	l.compiles++
	entry := src.Entry
	if entry == "" {
		entry = "specdecode_bucket_kernel"
	}
	return &cudahotswap.Compilation{
		Entry: entry,
		Key:   entry,
		PTX:   ".version 8.5\n.target sm_90a\n.address_size 64\n.visible .entry " + entry + "(.param .u64 p0)\n{\n\tret;\n}\n",
	}, nil
}

func (l *bucketTestLoader) Load(_ context.Context, _ *cudahotswap.Compilation) (cudahotswap.Kernel, error) {
	l.loads++
	l.nextID++
	k := &bucketTestKernel{id: l.nextID}
	l.kernels[l.nextID] = k
	return k, nil
}

func TestAdaptiveBucketEMATrackingAndRegimeSelection(t *testing.T) {
	ctrl, err := NewAdaptiveBucketController(AdaptiveBucketConfig{
		EMAAlpha:                0.25,
		HysteresisSteps:         3,
		LowAcceptanceThreshold:  0.30,
		HighAcceptanceThreshold: 0.65,
	})
	if err != nil {
		t.Fatalf("NewAdaptiveBucketController: %v", err)
	}

	// Step 1: 0/4 draft tokens accepted -> EMA initializes to 0.0 (< 0.30).
	ema := ctrl.RecordStep(1, 4, 0)
	if math.Abs(ema-0.0) > 1e-9 {
		t.Fatalf("step 1 EMA = %v, want 0.0", ema)
	}
	recLow1 := ctrl.RecommendBucket(1)
	if recLow1.Regime != RegimeSkinnyDecode || recLow1.BucketM != 1 {
		t.Fatalf("low acceptance (EMA=0.0, batch=1) recommended %+v, want RegimeSkinnyDecode M=1", recLow1)
	}

	// Step 2: 3/4 draft tokens accepted (0.75) -> EMA = 0.25*0.75 + 0.75*0.0 = 0.1875.
	// Still < LowAcceptanceThreshold (0.30), in upper skinny range (>= 0.15) -> M=2 RegimeSkinnyDecode.
	ema = ctrl.RecordStep(1, 4, 3)
	if math.Abs(ema-0.1875) > 1e-9 {
		t.Fatalf("step 2 EMA = %v, want 0.1875", ema)
	}
	recLow2 := ctrl.RecommendBucket(1)
	if recLow2.Regime != RegimeSkinnyDecode || recLow2.BucketM != 2 {
		t.Fatalf("low-upper acceptance (EMA=0.1875, batch=1) recommended %+v, want RegimeSkinnyDecode M=2", recLow2)
	}

	// Step 3: 4/4 draft tokens accepted (1.0) -> EMA = 0.25*1.0 + 0.75*0.1875 = 0.390625.
	// Now in [0.30, 0.65) -> medium acceptance -> RegimeMTPTreeVerify M=4.
	ema = ctrl.RecordStep(1, 4, 4)
	if math.Abs(ema-0.390625) > 1e-9 {
		t.Fatalf("step 3 EMA = %v, want 0.390625", ema)
	}
	recMed := ctrl.RecommendBucket(1)
	if recMed.Regime != RegimeMTPTreeVerify || recMed.BucketM != 4 {
		t.Fatalf("medium acceptance (EMA=%v, batch=1) recommended %+v, want RegimeMTPTreeVerify M=4", ema, recMed)
	}
	if recMed.Topology == nil || recMed.Topology.NumNodes() != 4 {
		t.Fatalf("medium bucket Topology = %+v, want 4-node DraftTreeTopology", recMed.Topology)
	}
	if recMed.MTPAttention == nil || recMed.MTPAttention.QTokensPerSeq != 4 {
		t.Fatalf("medium bucket MTPAttention = %+v, want QTokensPerSeq=4", recMed.MTPAttention)
	}
	if _, err := recMed.MTPAttention.Source(); err != nil {
		t.Fatalf("medium bucket MTPAttention.Source(): %v", err)
	}

	// Feed high acceptance steps until EMA >= 0.65:
	// Step 4 (4/4): EMA = 0.25 + 0.75*0.390625 = 0.54296875
	// Step 5 (4/4): EMA = 0.25 + 0.75*0.54296875 = 0.6572265625 (>= 0.65)
	ctrl.RecordStep(1, 4, 4)
	ema = ctrl.RecordStep(1, 4, 4)
	if ema < 0.65 {
		t.Fatalf("step 5 EMA = %v, want >= 0.65", ema)
	}
	recHigh := ctrl.RecommendBucket(1)
	if recHigh.Regime != RegimeMTPTreeVerify || recHigh.BucketM != 8 {
		t.Fatalf("high acceptance (EMA=%v, batch=1) recommended %+v, want RegimeMTPTreeVerify M=8", ema, recHigh)
	}
	if recHigh.Topology == nil || recHigh.Topology.NumNodes() != 8 {
		t.Fatalf("high bucket Topology = %+v, want 8-node DraftTreeTopology", recHigh.Topology)
	}
	if recHigh.MTPAttention == nil || recHigh.MTPAttention.QTokensPerSeq != 8 {
		t.Fatalf("high bucket MTPAttention = %+v, want QTokensPerSeq=8", recHigh.MTPAttention)
	}

	// Large effective batch: when ActiveBatch * QTokensPerSeq >= 16, regime
	// transitions to RegimeTilePrefill (M >= 16).
	recTile16 := ctrl.RecommendBucket(2) // 2 * 8 = 16
	if recTile16.Regime != RegimeTilePrefill || recTile16.BucketM != 16 {
		t.Fatalf("large batch (batch=2, QTokensPerSeq=8 -> M=16) recommended %+v, want RegimeTilePrefill M=16", recTile16)
	}
	recTile64 := ctrl.RecommendBucket(8) // 8 * 8 = 64
	if recTile64.Regime != RegimeTilePrefill || recTile64.BucketM != 64 {
		t.Fatalf("large batch (batch=8, QTokensPerSeq=8 -> M=64) recommended %+v, want RegimeTilePrefill M=64", recTile64)
	}
}

func TestAdaptiveBucketHysteresisAntiThrashing(t *testing.T) {
	t.Setenv("GPU_LOCKDIR", filepath.Join(t.TempDir(), "gpu-lock"))
	ctx := context.Background()
	loader := newBucketTestLoader()
	svc, err := cudahotswap.NewService(loader)
	if err != nil {
		t.Fatalf("cudahotswap.NewService: %v", err)
	}
	defer func() {
		if err := svc.Close(ctx); err != nil {
			t.Errorf("svc.Close: %v", err)
		}
	}()

	ctrl, err := NewAdaptiveBucketController(AdaptiveBucketConfig{
		EMAAlpha:                0.90, // responsive EMA to test hysteresis gate under rapid oscillation
		HysteresisSteps:         3,
		LowAcceptanceThreshold:  0.30,
		HighAcceptanceThreshold: 0.65,
		Service:                 svc,
		KernelName:              "specdecode_thrash_test",
	})
	if err != nil {
		t.Fatalf("NewAdaptiveBucketController: %v", err)
	}
	if ctrl.CurrentBucket().BucketM != 1 {
		t.Fatalf("initial BucketM = %d, want 1", ctrl.CurrentBucket().BucketM)
	}

	// Oscillate between 2 consecutive high-acceptance steps (recommending M=8)
	// and 1 low-acceptance step (recommending M=1). Because HysteresisSteps=3,
	// no hot-swap may occur.
	for cycle := range 4 {
		// High step 1 -> recommends M=8 (consecutive=1)
		ctrl.RecordStep(1, 8, 8)
		if gotM, gotSteps := ctrl.PendingBucket(); gotM != 8 || gotSteps != 1 {
			t.Fatalf("cycle %d high step 1: pending=(%d, %d), want (8, 1)", cycle, gotM, gotSteps)
		}
		// High step 2 -> recommends M=8 (consecutive=2 < 3)
		ctrl.RecordStep(1, 8, 8)
		if gotM, gotSteps := ctrl.PendingBucket(); gotM != 8 || gotSteps != 2 {
			t.Fatalf("cycle %d high step 2: pending=(%d, %d), want (8, 2)", cycle, gotM, gotSteps)
		}
		if ctrl.Transitions() != 0 || ctrl.CurrentBucket().BucketM != 1 {
			t.Fatalf("cycle %d: unexpected early transition: transitions=%d currentM=%d",
				cycle, ctrl.Transitions(), ctrl.CurrentBucket().BucketM)
		}

		// Drop acceptance on step 3 -> recommends M=1 (current bucket), resetting hysteresis counter.
		ctrl.RecordStep(1, 8, 0)
		if gotM, gotSteps := ctrl.PendingBucket(); gotM != 0 || gotSteps != 0 {
			t.Fatalf("cycle %d low reset step: pending=(%d, %d), want (0, 0)", cycle, gotM, gotSteps)
		}
		if ctrl.Transitions() != 0 || ctrl.CurrentBucket().BucketM != 1 {
			t.Fatalf("cycle %d after reset: transitions=%d currentM=%d, want 0 and 1",
				cycle, ctrl.Transitions(), ctrl.CurrentBucket().BucketM)
		}
	}

	// Now sustain 3 consecutive high-acceptance steps -> must hot-swap on step 3.
	ctrl.RecordStep(1, 8, 8)
	ctrl.RecordStep(1, 8, 8)
	if ctrl.Transitions() != 0 {
		t.Fatalf("before 3rd consecutive step: transitions=%d, want 0", ctrl.Transitions())
	}
	ctrl.RecordStep(1, 8, 8)
	if ctrl.LastError() != nil {
		t.Fatalf("RecordStep hot-swap error: %v", ctrl.LastError())
	}
	if ctrl.Transitions() != 1 || ctrl.CurrentBucket().BucketM != 8 || ctrl.CurrentBucket().Regime != RegimeMTPTreeVerify {
		t.Fatalf("after 3rd consecutive high step: transitions=%d bucket=%+v, want transitions=1 M=8 RegimeMTPTreeVerify",
			ctrl.Transitions(), ctrl.CurrentBucket())
	}
}

func TestAdaptiveBucketLiveHotswapAndRollback(t *testing.T) {
	t.Setenv("GPU_LOCKDIR", filepath.Join(t.TempDir(), "gpu-lock"))
	ctx := context.Background()

	loader := newBucketTestLoader()
	svc, err := cudahotswap.NewService(loader)
	if err != nil {
		t.Fatalf("cudahotswap.NewService: %v", err)
	}
	defer func() {
		if err := svc.Close(ctx); err != nil {
			t.Errorf("svc.Close: %v", err)
		}
	}()

	const kernelName = "specdecode_flight_kernel"
	ctrl, err := NewAdaptiveBucketController(AdaptiveBucketConfig{
		EMAAlpha:                0.50,
		HysteresisSteps:         3,
		LowAcceptanceThreshold:  0.30,
		HighAcceptanceThreshold: 0.65,
		Service:                 svc,
		KernelName:              kernelName,
	})
	if err != nil {
		t.Fatalf("NewAdaptiveBucketController: %v", err)
	}

	// Verify initial state (Bucket M=1, RegimeSkinnyDecode, Version 1).
	stV1, err := svc.Status(kernelName)
	if err != nil || stV1.Version != 1 {
		t.Fatalf("initial service status = %+v, err = %v; want Version=1", stV1, err)
	}
	geomM1 := ctrl.CurrentGeometry()
	wantGeomM1 := cudahotswap.LaunchGeometry{
		Grid:      cudashape.Grid3D{X: 1, Y: 8, Z: 1},
		Block:     driver.Dims(128, 1, 1),
		SharedMem: 0,
	}
	if geomM1 != wantGeomM1 {
		t.Fatalf("initial geometry = %+v, want %+v", geomM1, wantGeomM1)
	}

	// Launch through service and verify kernel 1 received M=1 3D geometry.
	p, err := svc.Launch(ctx, kernelName, cudalaunch.Config{}, uint64(0))
	if err != nil {
		t.Fatalf("svc.Launch v1: %v", err)
	}
	if err := p.Wait(ctx); err != nil {
		t.Fatalf("p.Wait v1: %v", err)
	}
	if loader.kernels[1].lastLaunch.Grid != wantGeomM1.Grid || loader.kernels[1].lastLaunch.Block != wantGeomM1.Block {
		t.Fatalf("v1 launch config = %+v, want grid=%+v block=%+v",
			loader.kernels[1].lastLaunch, wantGeomM1.Grid, wantGeomM1.Block)
	}

	// 1. Transition M=1 (RegimeSkinnyDecode) -> M=8 (RegimeMTPTreeVerify) via 3 high-acceptance steps.
	for i := range 3 {
		trans, err := ctrl.Step(ctx, 1, 8, 8)
		if err != nil {
			t.Fatalf("M=8 step %d: %v", i+1, err)
		}
		if (i < 2 && trans.Swapped) || (i == 2 && !trans.Swapped) {
			t.Fatalf("M=8 step %d: Swapped=%v, want %v", i+1, trans.Swapped, i == 2)
		}
	}
	if ctrl.CurrentBucket().BucketM != 8 || ctrl.CurrentBucket().Regime != RegimeMTPTreeVerify {
		t.Fatalf("after M=8 transition: bucket = %+v, want M=8 RegimeMTPTreeVerify", ctrl.CurrentBucket())
	}
	stV2, err := svc.Status(kernelName)
	if err != nil || stV2.Version != 2 {
		t.Fatalf("after M=8 transition: status = %+v, err = %v; want Version=2", stV2, err)
	}
	geomM8 := ctrl.CurrentGeometry()
	if geomM8.Grid != (cudashape.Grid3D{X: 1, Y: 8, Z: 1}) || geomM8.Block != driver.Dims(128, 1, 1) || geomM8.SharedMem == 0 {
		t.Fatalf("M=8 geometry = %+v, want Grid=(1,8,1) Block=(128,1,1) SharedMem>0", geomM8)
	}

	// Launch through service and verify kernel 2 received M=8 3D geometry + dynamic shared memory.
	p, err = svc.Launch(ctx, kernelName, cudalaunch.Config{}, uint64(0))
	if err != nil {
		t.Fatalf("svc.Launch v2: %v", err)
	}
	if err := p.Wait(ctx); err != nil {
		t.Fatalf("p.Wait v2: %v", err)
	}
	if loader.kernels[2].lastLaunch.Grid != geomM8.Grid ||
		loader.kernels[2].lastLaunch.Block != geomM8.Block ||
		loader.kernels[2].lastLaunch.SharedMem != geomM8.SharedMem {
		t.Fatalf("v2 launch config = %+v, want %+v", loader.kernels[2].lastLaunch, geomM8)
	}

	// 2. Transition M=8 (RegimeMTPTreeVerify) -> M=64 (RegimeTilePrefill) via 3 large-batch steps (batch=8, 8*8=64).
	for i := range 3 {
		trans, err := ctrl.Step(ctx, 8, 64, 56)
		if err != nil {
			t.Fatalf("M=64 step %d: %v", i+1, err)
		}
		if (i < 2 && trans.Swapped) || (i == 2 && !trans.Swapped) {
			t.Fatalf("M=64 step %d: Swapped=%v, want %v", i+1, trans.Swapped, i == 2)
		}
	}
	if ctrl.CurrentBucket().BucketM != 64 || ctrl.CurrentBucket().Regime != RegimeTilePrefill {
		t.Fatalf("after M=64 transition: bucket = %+v, want M=64 RegimeTilePrefill", ctrl.CurrentBucket())
	}
	stV3, err := svc.Status(kernelName)
	if err != nil || stV3.Version != 3 {
		t.Fatalf("after M=64 transition: status = %+v, err = %v; want Version=3", stV3, err)
	}
	geomM64 := ctrl.CurrentGeometry()
	wantGeomM64 := cudahotswap.LaunchGeometry{
		Grid:      cudashape.Grid3D{X: 8, Y: 8, Z: 1},
		Block:     driver.Dims(128, 2, 1),
		SharedMem: 49152,
	}
	if geomM64 != wantGeomM64 {
		t.Fatalf("M=64 geometry = %+v, want %+v", geomM64, wantGeomM64)
	}

	p, err = svc.Launch(ctx, kernelName, cudalaunch.Config{}, uint64(0))
	if err != nil {
		t.Fatalf("svc.Launch v3: %v", err)
	}
	if err := p.Wait(ctx); err != nil {
		t.Fatalf("p.Wait v3: %v", err)
	}
	if loader.kernels[3].lastLaunch.Grid != wantGeomM64.Grid ||
		loader.kernels[3].lastLaunch.Block != wantGeomM64.Block ||
		loader.kernels[3].lastLaunch.SharedMem != wantGeomM64.SharedMem {
		t.Fatalf("v3 launch config = %+v, want %+v", loader.kernels[3].lastLaunch, wantGeomM64)
	}

	// 3. Simulate a post-switch token verification failure on M=64 using
	// VerifyAndRollbackIfFailed (which calls RollbackBucket / RestorePrevious).
	// Must restore immediate prior bucket M=8 (version 2) and geomM8, NOT version 1.
	rolledBack, verifyErr := ctrl.VerifyAndRollbackIfFailed(ctx, func() error {
		return errors.New("post-switch greedy token verification mismatch at draft node 5")
	})
	if !rolledBack || verifyErr == nil {
		t.Fatalf("VerifyAndRollbackIfFailed: rolledBack=%v err=%v, want true and non-nil error", rolledBack, verifyErr)
	}
	if ctrl.CurrentBucket().BucketM != 8 || ctrl.CurrentBucket().Regime != RegimeMTPTreeVerify {
		t.Fatalf("after RollbackBucket: current bucket = %+v, want M=8 RegimeMTPTreeVerify", ctrl.CurrentBucket())
	}
	if ctrl.CurrentGeometry() != geomM8 {
		t.Fatalf("after RollbackBucket: geometry = %+v, want %+v", ctrl.CurrentGeometry(), geomM8)
	}
	stAfterRB1, err := svc.Status(kernelName)
	if err != nil || stAfterRB1.Version != 2 {
		t.Fatalf("service status after first rollback = %+v (err=%v), want Version=2", stAfterRB1, err)
	}

	// Verify launching after rollback executes kernel 2 with M=8's exact 3D launch geometry.
	p, err = svc.Launch(ctx, kernelName, cudalaunch.Config{}, uint64(0))
	if err != nil {
		t.Fatalf("svc.Launch after rollback 1: %v", err)
	}
	if err := p.Wait(ctx); err != nil {
		t.Fatalf("p.Wait after rollback 1: %v", err)
	}
	if loader.kernels[2].launches != 2 ||
		loader.kernels[2].lastLaunch.Grid != geomM8.Grid ||
		loader.kernels[2].lastLaunch.Block != geomM8.Block ||
		loader.kernels[2].lastLaunch.SharedMem != geomM8.SharedMem {
		t.Fatalf("kernel 2 after rollback: launches=%d cfg=%+v, want launches=2 cfg=%+v",
			loader.kernels[2].launches, loader.kernels[2].lastLaunch, geomM8)
	}

	// 4. Call RestorePrevious directly to roll back one more step from M=8 (v2) -> M=1 (v1).
	stAfterRB2, err := ctrl.RestorePrevious(ctx)
	if err != nil {
		t.Fatalf("RestorePrevious to M=1: %v", err)
	}
	if stAfterRB2.Version != 1 || ctrl.CurrentBucket().BucketM != 1 || ctrl.CurrentGeometry() != wantGeomM1 {
		t.Fatalf("after second rollback: state=%+v bucket=%+v geom=%+v, want Version=1 M=1 geom=%+v",
			stAfterRB2, ctrl.CurrentBucket(), ctrl.CurrentGeometry(), wantGeomM1)
	}

	// 5. Re-select M=8 after it was already compiled and retained in svc:
	// must use svc.Select (UsedSelect=true) without invoking loader.Compile again.
	compilesBefore := loader.compiles
	for range 3 {
		ctrl.RecordStep(1, 8, 8)
	}
	if ctrl.LastError() != nil {
		t.Fatalf("re-transition to M=8 error: %v", ctrl.LastError())
	}
	if ctrl.CurrentBucket().BucketM != 8 || ctrl.CurrentState().Version != 2 {
		t.Fatalf("re-transition to M=8: bucket=%+v state=%+v, want M=8 Version=2",
			ctrl.CurrentBucket(), ctrl.CurrentState())
	}
	if loader.compiles != compilesBefore {
		t.Fatalf("re-transition compiled %d times, want 0 (retained Select)", loader.compiles-compilesBefore)
	}
}

func TestAdaptiveBucketConfigValidation(t *testing.T) {
	badConfigs := []AdaptiveBucketConfig{
		{EMAAlpha: -0.1},
		{EMAAlpha: 1.5},
		{HysteresisSteps: -1},
		{LowAcceptanceThreshold: 0.70, HighAcceptanceThreshold: 0.50},
		{
			Buckets: []BucketSpec{
				{BucketM: 4, Regime: RegimeMTPTreeVerify},
				{BucketM: 4, Regime: RegimeMTPTreeVerify},
			},
		},
		{
			Buckets: []BucketSpec{
				{BucketM: 0},
			},
		},
	}
	for i, cfg := range badConfigs {
		if _, err := NewAdaptiveBucketController(cfg); err == nil {
			t.Fatalf("badConfigs[%d] (%+v) succeeded, want error", i, cfg)
		}
	}
}

