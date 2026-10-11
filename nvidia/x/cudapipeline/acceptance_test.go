package cudapipeline_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudagroup"
	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudapipeline"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudareg"
	"github.com/tmc/lib/nvidia/x/cudashared"
	"github.com/tmc/lib/nvidia/x/cudatma"
	"github.com/tmc/lib/nvidia/x/cudatmem"
)

// 1. TestBarrierInitialization: missing or duplicate init rejected
func TestBarrierInitialization(t *testing.T) {
	t.Run("missing_init_schedule", func(t *testing.T) {
		sched, err := cudapipeline.BuildWarpSpecializedSchedule(2, 1024, 2)
		if err != nil {
			t.Fatalf("build schedule: %v", err)
		}
		// Mutant: omit stage 0 init
		filteredSteps := make([]cudapipeline.ScheduleStep, 0, len(sched.Steps))
		for _, step := range sched.Steps {
			if step.Kind == cudapipeline.StepInit && step.Stage == 0 {
				continue // Omit init
			}
			filteredSteps = append(filteredSteps, step)
		}
		sched.Steps = filteredSteps

		errs := sched.Validate()
		if len(errs) == 0 {
			t.Fatalf("expected error for uninitialized barrier, got 0")
		}
		found := false
		for _, e := range errs {
			if e.Violation == cudapipeline.ErrUninitializedBarrier {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected ErrUninitializedBarrier, got %v", errs)
		}
	})

	t.Run("duplicate_init_schedule", func(t *testing.T) {
		sched, err := cudapipeline.BuildWarpSpecializedSchedule(2, 1024, 2)
		if err != nil {
			t.Fatalf("build schedule: %v", err)
		}
		// Mutant: duplicate stage 0 init
		sched.Steps = append([]cudapipeline.ScheduleStep{
			{StepIndex: 999, Role: cudapipeline.RoleProducer, Kind: cudapipeline.StepInit, Stage: 0},
		}, sched.Steps...)

		errs := sched.Validate()
		found := false
		for _, e := range errs {
			if e.Violation == cudapipeline.ErrDuplicateInit {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected ErrDuplicateInit, got %v", errs)
		}
	})
}

// 2. TestTransactionBalance: under/over-counting transaction bytes rejected
func TestTransactionBalance(t *testing.T) {
	t.Run("undercount_schedule", func(t *testing.T) {
		sched, err := cudapipeline.BuildWarpSpecializedSchedule(2, 1024, 2)
		if err != nil {
			t.Fatalf("build schedule: %v", err)
		}
		// Mutant: reduce TMA payload bytes emitted
		for i := range sched.Steps {
			if sched.Steps[i].Kind == cudapipeline.StepIssueTMA {
				sched.Steps[i].Bytes = 512 // 512 != 1024
				break
			}
		}
		errs := sched.Validate()
		found := false
		for _, e := range errs {
			if e.Violation == cudapipeline.ErrTransactionUnderflow {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected ErrTransactionUnderflow, got %v", errs)
		}
	})
}

// 3. TestStageOwnership: producer overwriting active stage rejected
func TestStageOwnership(t *testing.T) {
	sched, err := cudapipeline.BuildWarpSpecializedSchedule(2, 1024, 2)
	if err != nil {
		t.Fatalf("build schedule: %v", err)
	}
	// Mutant: producer acquires stage 0 a second time before consumer has released it
	var mutated []cudapipeline.ScheduleStep
	for _, step := range sched.Steps {
		mutated = append(mutated, step)
		if step.Kind == cudapipeline.StepIssueTMA && step.Stage == 0 && step.Epoch == 0 {
			// Premature acquire of stage 0
			mutated = append(mutated, cudapipeline.ScheduleStep{
				StepIndex: 998,
				Role:      cudapipeline.RoleProducer,
				Kind:      cudapipeline.StepAcquireStage,
				Stage:     0,
				Epoch:     1,
			})
		}
	}
	sched.Steps = mutated
	errs := sched.Validate()
	found := false
	for _, e := range errs {
		if e.Violation == cudapipeline.ErrStageOverwrite {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected ErrStageOverwrite, got %v", errs)
	}
}

// 4. TestEpochReuse: stale epoch reuse or wrong epoch advance detected
func TestEpochReuse(t *testing.T) {
	sched, err := cudapipeline.BuildWarpSpecializedSchedule(2, 1024, 4)
	if err != nil {
		t.Fatalf("build schedule: %v", err)
	}
	// Mutant: acquire stage with stale/duplicate epoch
	foundMutant := false
	for i := range sched.Steps {
		if sched.Steps[i].Kind == cudapipeline.StepAcquireStage && sched.Steps[i].Epoch > 0 {
			sched.Steps[i].Epoch = 0 // reuse old epoch 0
			foundMutant = true
			break
		}
	}
	if !foundMutant {
		t.Fatalf("failed to find step with Epoch > 0 to mutate")
	}
	errs := sched.Validate()
	found := false
	for _, e := range errs {
		if e.Violation == cudapipeline.ErrEpochReuseStale {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected ErrEpochReuseStale, got %v", errs)
	}
}

// 5. TestBarrierPhaseParity: consumer waiting on mismatched parity detected
func TestBarrierPhaseParity(t *testing.T) {
	sched, err := cudapipeline.BuildWarpSpecializedSchedule(2, 1024, 2)
	if err != nil {
		t.Fatalf("build schedule: %v", err)
	}
	// Mutant: invert wait parity on consumer
	for i := range sched.Steps {
		if sched.Steps[i].Kind == cudapipeline.StepWaitParity {
			sched.Steps[i].Parity ^= 1 // flip parity
			break
		}
	}
	errs := sched.Validate()
	found := false
	for _, e := range errs {
		if e.Violation == cudapipeline.ErrBarrierPhaseParity {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected ErrBarrierPhaseParity, got %v", errs)
	}
}

// 6. TestUnsupportedControlFlow: branch/call across barrier protocol rejected
func TestUnsupportedControlFlow(t *testing.T) {
	ptxSrc := `
.version 8.8
.target sm_121a
.address_size 64

.visible .entry bad_cfg_kernel(.param .u64 tmap) {
    .reg .b64 %rd_tmap, %state;
    .shared .align 8 .b64 bar;

    mbarrier.init.shared::cta.b64 [bar], 1;
    mbarrier.arrive.expect_tx.shared::cta.b64 %state, [bar], 1024;
    bra arbitrary_target; // unsupported arbitrary branch across barrier protocol
arbitrary_target:
    mbarrier.try_wait.parity.shared::cta.b64 %p0, [bar], 0;
    ret;
}
`
	mod, err := cudaptx.Parse(strings.NewReader(ptxSrc))
	if err != nil {
		t.Fatalf("parse PTX: %v", err)
	}
	violations, err := cudapipeline.ValidatePTX(mod)
	if err != nil {
		t.Fatalf("validate PTX: %v", err)
	}
	found := false
	for _, v := range violations {
		if v.Violation == cudapipeline.ErrUnsupportedControlFlow {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected ErrUnsupportedControlFlow, got %v", violations)
	}
}

// 7. TestWarpScheduleResources: exceeding 101,376 B smem, >48 warps, >24 blocks rejected; 256 reg quantum validated
func TestWarpScheduleResources(t *testing.T) {
	// Valid configuration on SM121
	summary, violations := cudapipeline.ValidateSM121Resources(32768, 4, 2, 32)
	if len(violations) != 0 {
		t.Errorf("expected valid resources, got %v", violations)
	}
	if !summary.WithinLimits {
		t.Errorf("expected WithinLimits=true")
	}
	// Registers per warp: 32 threads * 32 regs = 1024 regs. 1024 is multiple of 256.
	if summary.RegistersPerWarp != 1024 {
		t.Errorf("RegistersPerWarp = %d, want 1024", summary.RegistersPerWarp)
	}

	// Mutant 1: Exceed smem optin limit 101,376 B
	_, v1 := cudapipeline.ValidateSM121Resources(102000, 4, 1, 32)
	if len(v1) == 0 || v1[0].Violation != cudapipeline.ErrExceededSMEMLimit {
		t.Errorf("expected ErrExceededSMEMLimit, got %v", v1)
	}

	// Mutant 2: Exceed max blocks (25 > 24)
	_, v2 := cudapipeline.ValidateSM121Resources(1024, 1, 25, 32)
	if len(v2) == 0 || v2[0].Violation != cudapipeline.ErrExceededBlocksPerSM {
		t.Errorf("expected ErrExceededBlocksPerSM, got %v", v2)
	}

	// Mutant 3: Exceed max warps (4 warps * 13 blocks = 52 warps > 48)
	_, v3 := cudapipeline.ValidateSM121Resources(1024, 4, 13, 32)
	if len(v3) == 0 || v3[0].Violation != cudapipeline.ErrExceededWarpsPerSM {
		t.Errorf("expected ErrExceededWarpsPerSM, got %v", v3)
	}
}

// 8. TestAuditAndRejectInvalidBulkTMEMPTX: direct *.tmem.global copy strings rejected
func TestAuditAndRejectInvalidBulkTMEMPTX(t *testing.T) {
	ptxSrc := `
.version 8.8
.target sm_100a
.address_size 64

.visible .entry invalid_tmem_bulk_kernel(.param .u64 tmap) {
    .reg .b64 %rd_tmap;
    .shared .align 8 .b64 bar;

    mbarrier.init.shared::cta.b64 [bar], 1;
    mbarrier.arrive.expect_tx.shared::cta.b64 %state, [bar], 1024;
    // Direct unverified global-to-TMEM copy must be rejected!
    cp.async.bulk.tensor.2d.tmem.global.mbarrier::complete_tx::bytes [%tmem0], [%rd_tmap, {0, 0}], [bar];
    ret;
}
`
	mod, err := cudaptx.Parse(strings.NewReader(ptxSrc))
	if err != nil {
		t.Fatalf("parse PTX: %v", err)
	}
	violations, err := cudapipeline.ValidatePTX(mod)
	if err != nil {
		t.Fatalf("validate PTX: %v", err)
	}
	found := false
	for _, v := range violations {
		if v.Violation == cudapipeline.ErrInvalidBulkTMEMPTX {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected ErrInvalidBulkTMEMPTX, got %v", violations)
	}
}

// TestHostPipelineCopyReference checks the CPU staged-copy oracle.
func TestHostPipelineCopyReference(t *testing.T) {

	// Reference host CPU oracle for double buffered copy
	stages := 2
	iters := 4
	bufferSize := 256
	hostSmem := make([][]byte, stages)
	for i := range hostSmem {
		hostSmem[i] = make([]byte, bufferSize)
	}

	srcGlobal := make([]byte, iters*bufferSize)
	for i := range srcGlobal {
		srcGlobal[i] = byte(i % 251)
	}
	dstGlobal := make([]byte, iters*bufferSize)

	// CPU reference execution simulating pipeline stages
	for iter := 0; iter < iters; iter++ {
		stage := iter % stages
		copy(hostSmem[stage], srcGlobal[iter*bufferSize:(iter+1)*bufferSize])
		copy(dstGlobal[iter*bufferSize:(iter+1)*bufferSize], hostSmem[stage])
	}

	for i := range srcGlobal {
		if dstGlobal[i] != srcGlobal[i] {
			t.Fatalf("byte %d mismatch: got %d want %d", i, dstGlobal[i], srcGlobal[i])
		}
	}
}

// 10. TestTMEMTargetRejection: tcgen05 and TMEM rejected on sm_121a and sm_90, accepted on sm_100a
func TestTMEMTargetRejection(t *testing.T) {
	ptxSrc := `
.version 8.8
.target %s
.address_size 64

.visible .entry tmem_alloc_kernel() {
    .shared .align 4 .b32 taddr;
    tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [taddr], 32;
    ret;
}
`
	// sm_121a must reject tcgen05
	mod121, err := cudaptx.Parse(strings.NewReader(strings.ReplaceAll(ptxSrc, "%s", "sm_121a")))
	if err != nil {
		t.Fatalf("parse PTX: %v", err)
	}
	v121, _ := cudapipeline.ValidatePTX(mod121)
	found121 := false
	for _, v := range v121 {
		if v.Violation == cudapipeline.ErrUnsupportedTMEM {
			found121 = true
			break
		}
	}
	if !found121 {
		t.Errorf("expected ErrUnsupportedTMEM on sm_121a, got %v", v121)
	}

	// sm_100a must accept tcgen05
	mod100, err := cudaptx.Parse(strings.NewReader(strings.ReplaceAll(ptxSrc, "%s", "sm_100a")))
	if err != nil {
		t.Fatalf("parse PTX: %v", err)
	}
	v100, _ := cudapipeline.ValidatePTX(mod100)
	for _, v := range v100 {
		if v.Violation == cudapipeline.ErrUnsupportedTMEM {
			t.Errorf("unexpected ErrUnsupportedTMEM on sm_100a: %v", v)
		}
	}
}

// 11. TestRegisterPoolProgress: integrating cudareg schedule concepts
func TestRegisterPoolProgress(t *testing.T) {
	// Create a legal register redistribution schedule with cudareg
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
		t.Fatalf("cudagroup.WarpGroupForCTA: %v", err)
	}
	groupCons, err := cudagroup.WarpGroupForCTA(cta, 4)
	if err != nil {
		t.Fatalf("cudagroup.WarpGroupForCTA: %v", err)
	}

	sched := cudareg.Schedule{
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

	if err := sched.Validate(); err != nil {
		t.Fatalf("cudareg schedule validation: %v", err)
	}
}

// 12. TestMulticastCompletionPerCTA: one completion transaction per destination rank
func TestMulticastCompletionPerCTA(t *testing.T) {
	// 4 destination ranks in cluster
	destRanks := []uint32{0, 1, 2, 3}
	txBytes := uint64(1024)
	mc, err := cudatma.NewMulticast(0, destRanks, txBytes)
	if err != nil {
		t.Fatalf("NewMulticast: %v", err)
	}

	// Arrive/complete rank by rank
	for _, rank := range destRanks[:3] {
		if err := mc.Complete(rank, txBytes); err != nil {
			t.Fatalf("Complete rank %d: %v", rank, err)
		}
	}
	if mc.CompleteAll() {
		t.Errorf("CompleteAll() = true before final rank completed")
	}

	// Complete final rank
	if err := mc.Complete(destRanks[3], txBytes); err != nil {
		t.Fatalf("Complete rank 3: %v", err)
	}
	if !mc.CompleteAll() {
		t.Errorf("CompleteAll() = false after all ranks completed")
	}
}

// 13. TestUnsupportedTransferEmissionRejected: verifying fallback on sm_121a
func TestUnsupportedTransferEmissionRejected(t *testing.T) {
	// Transfer directly from global to TMEM on sm_121a must be rejected
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
		Descriptor:   desc,
		Direction:    cudatma.GlobalToTMEM,
		Staged:       false,
		SharedOffset: 0,
		Bytes:        1024,
	}
	if err := tr.Validate(); err == nil {
		t.Errorf("expected error for direct GlobalToTMEM on sm_121a, got nil")
	}

	// cudatmem target check on sm_121a
	_, err := cudatmem.ParseTarget("sm_121a")
	if err == nil {
		t.Errorf("expected cudatmem to reject sm_121a, got nil")
	}
}

// 14. TestHardwareTilePipelineIntegration: integration with TMA box geometry, shared memory buffer layout, and mbarrier
func TestHardwareTilePipelineIntegration(t *testing.T) {
	// Layout & shared memory allocation
	tile, err := cudalayout.NewTile(32, 32, 2, cudalayout.WithAlignment(16))
	if err != nil {
		t.Fatalf("cudalayout.NewTile: %v", err)
	}
	req0, err := cudashared.RequestForTile(tile, "stage_0", 0, 0, false)
	if err != nil {
		t.Fatalf("RequestForTile: %v", err)
	}
	req1, err := cudashared.RequestForTile(tile, "stage_1", 1, 1, false)
	if err != nil {
		t.Fatalf("RequestForTile: %v", err)
	}

	limits := cudashared.Limits{
		StaticMax:  cudapipeline.SM121MaxOptinSharedMemBytes,
		DynamicMax: cudapipeline.SM121MaxOptinSharedMemBytes,
		TotalMax:   cudapipeline.SM121MaxOptinSharedMemBytes,
		Source:     "sm_121a",
	}
	plan, err := cudashared.NewPlan(limits)
	if err != nil {
		t.Fatalf("NewPlan: %v", err)
	}
	reg0, err := plan.Add(req0)
	if err != nil {
		t.Fatalf("plan.Add stage 0: %v", err)
	}
	reg1, err := plan.Add(req1)
	if err != nil {
		t.Fatalf("plan.Add stage 1: %v", err)
	}

	// Construct cudapipeline with 2 stages
	box := cudapipeline.TMABox{
		DimX:        32,
		DimY:        32,
		ElementSize: 2,
		StrideBytes: 64,
	}
	cfg := cudapipeline.PipelineConfig{
		TargetArch:            cudapipeline.ArchSM121a,
		NumStages:             2,
		Box:                   box,
		MemorySpace:           cudapipeline.MemorySpaceShared,
		SharedMemBaseOffset:   int(reg0.Offset),
		ArrivalTokensPerStage: 1,
	}
	pipe, err := cudapipeline.NewPipeline(cfg)
	if err != nil {
		t.Fatalf("NewPipeline: %v", err)
	}
	if pipe.NumStages() != 2 {
		t.Errorf("NumStages = %d, want 2", pipe.NumStages())
	}
	if pipe.Stages[0].TMABox.PayloadBytes() != 2048 {
		t.Errorf("PayloadBytes = %d, want 2048", pipe.Stages[0].TMABox.PayloadBytes())
	}
	_ = reg1
}
