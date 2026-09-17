package cudapipeline

import (
	"fmt"
)

// Role identifies the functional responsibility of a warp or warpgroup in
// a specialized producer-consumer schedule.
type Role string

const (
	// RoleProducer issues asynchronous memory copies (e.g. TMA bulk loads)
	// into shared memory stages and signals mbarrier transactions.
	RoleProducer Role = "producer"

	// RoleConsumer waits on mbarrier phase arrival and consumes stage data
	// (e.g. via MMA/HMMA math or SIMT kernels) before releasing stage ownership.
	RoleConsumer Role = "consumer"
)

// StageState represents the lifecycle status of a pipeline buffer stage.
type StageState string

const (
	// StageIdle indicates the stage buffer is empty and unallocated for current epoch.
	StageIdle StageState = "idle"

	// StageProducing indicates the producer is actively issuing transfers into this stage.
	StageProducing StageState = "producing"

	// StageReady indicates all transaction bytes and arrival tokens have completed;
	// data is ready for consumer consumption.
	StageReady StageState = "ready"

	// StageConsuming indicates the consumer is actively reading/processing this stage.
	StageConsuming StageState = "consuming"
)

// StageOwnership models explicit ownership, buffer lifecycle, and epoch tracking
// for a single pipeline stage.
type StageOwnership struct {
	StageIndex int
	Owner      Role
	State      StageState
	Epoch      int  // 0-indexed epoch counter; stage can only advance after consumer release
	Parity     int  // Phase parity (Epoch & 1)
}

// BarrierExpectation models static mbarrier transaction and arrival counts
// expected for stage synchronization.
type BarrierExpectation struct {
	BarrierOffset    int
	ArrivalCount     int   // Expected thread arrival tokens (e.g. 1 for warp-specialized leader)
	TransactionBytes int64 // Expected transaction byte count from TMA transfer
	PhaseParity      int   // Expected phase parity bit (0 or 1)
}

// StepKind identifies the discrete protocol step in a warp-specialized schedule.
type StepKind string

const (
	StepInit            StepKind = "init"             // mbarrier.init
	StepAcquireStage    StepKind = "acquire_stage"    // Producer acquires idle stage for epoch
	StepArriveExpectTx  StepKind = "arrive_expect_tx" // mbarrier.arrive.expect_tx
	StepIssueTMA        StepKind = "issue_tma"        // cp.async.bulk.tensor
	StepWaitParity      StepKind = "wait_parity"      // mbarrier.try_wait.parity
	StepConsumeStage    StepKind = "consume_stage"    // Consumer processes stage data
	StepReleaseStage    StepKind = "release_stage"    // Consumer releases stage; epoch advances
)

// ScheduleStep records a single protocol step executed by a role.
type ScheduleStep struct {
	StepIndex int
	Role      Role
	Kind      StepKind
	Stage     int
	Epoch     int
	Bytes     int64
	Parity    int
	Desc      string
}

// PipelineSchedule represents an explicit, finite producer-consumer schedule
// modeling stages, buffer ownership, barrier expectations, and epoch reuse.
type PipelineSchedule struct {
	Stages       int
	Roles        []Role
	Steps        []ScheduleStep
	Expectations []BarrierExpectation
	PayloadBytes int64
}

// BuildWarpSpecializedSchedule generates an explicit finite schedule for a given
// stage count and payload byte size across numIterations iterations (epochs).
// It models producer prefetching, consumer waiting, and epoch reuse.
func BuildWarpSpecializedSchedule(numStages int, payloadBytes int64, numIterations int) (*PipelineSchedule, error) {
	if numStages < 2 {
		return nil, fmt.Errorf("cudapipeline: schedule requires at least 2 stages, got %d", numStages)
	}
	if payloadBytes <= 0 {
		return nil, fmt.Errorf("cudapipeline: payload bytes must be positive, got %d", payloadBytes)
	}
	if numIterations < 1 {
		return nil, fmt.Errorf("cudapipeline: numIterations must be at least 1, got %d", numIterations)
	}

	sched := &PipelineSchedule{
		Stages:       numStages,
		Roles:        []Role{RoleProducer, RoleConsumer},
		PayloadBytes: payloadBytes,
	}

	for s := 0; s < numStages; s++ {
		sched.Expectations = append(sched.Expectations, BarrierExpectation{
			BarrierOffset:    s * 8,
			ArrivalCount:     1, // 1 producer leader thread arriving per stage
			TransactionBytes: payloadBytes,
			PhaseParity:      0,
		})
	}

	stepIdx := 0
	addStep := func(role Role, kind StepKind, stage, epoch int, bytes int64, parity int, desc string) {
		sched.Steps = append(sched.Steps, ScheduleStep{
			StepIndex: stepIdx,
			Role:      role,
			Kind:      kind,
			Stage:     stage,
			Epoch:     epoch,
			Bytes:     bytes,
			Parity:    parity,
			Desc:      desc,
		})
		stepIdx++
	}

	// 1. Initialization: producer initializes all stage mbarriers
	for s := 0; s < numStages; s++ {
		addStep(RoleProducer, StepInit, s, 0, 0, 0, fmt.Sprintf("init mbarrier stage %d", s))
	}

	// 2. Prolog: Producer pre-fills the pipeline stages up to min(numStages-1, numIterations)
	prologStages := numStages - 1
	if prologStages > numIterations {
		prologStages = numIterations
	}

	for i := 0; i < prologStages; i++ {
		stage := i % numStages
		epoch := i / numStages
		parity := epoch & 1
		addStep(RoleProducer, StepAcquireStage, stage, epoch, 0, parity, fmt.Sprintf("acquire stage %d epoch %d", stage, epoch))
		addStep(RoleProducer, StepArriveExpectTx, stage, epoch, payloadBytes, parity, fmt.Sprintf("arrive expect_tx stage %d (%d bytes)", stage, payloadBytes))
		addStep(RoleProducer, StepIssueTMA, stage, epoch, payloadBytes, parity, fmt.Sprintf("issue TMA copy stage %d", stage))
	}

	// 3. Steady-state loop: Consumer consumes stage i, while Producer produces stage i + prologStages
	for i := 0; i < numIterations; i++ {
		cStage := i % numStages
		cEpoch := i / numStages
		cParity := cEpoch & 1

		// Producer fetches next stage if within bounds
		pIter := i + prologStages
		if pIter < numIterations {
			pStage := pIter % numStages
			pEpoch := pIter / numStages
			pParity := pEpoch & 1
			addStep(RoleProducer, StepAcquireStage, pStage, pEpoch, 0, pParity, fmt.Sprintf("acquire stage %d epoch %d", pStage, pEpoch))
			addStep(RoleProducer, StepArriveExpectTx, pStage, pEpoch, payloadBytes, pParity, fmt.Sprintf("arrive expect_tx stage %d (%d bytes)", pStage, payloadBytes))
			addStep(RoleProducer, StepIssueTMA, pStage, pEpoch, payloadBytes, pParity, fmt.Sprintf("issue TMA copy stage %d", pStage))
		}

		// Consumer waits on cStage parity, consumes, and releases
		addStep(RoleConsumer, StepWaitParity, cStage, cEpoch, 0, cParity, fmt.Sprintf("wait parity %d on stage %d epoch %d", cParity, cStage, cEpoch))
		addStep(RoleConsumer, StepConsumeStage, cStage, cEpoch, payloadBytes, cParity, fmt.Sprintf("consume stage %d epoch %d", cStage, cEpoch))
		addStep(RoleConsumer, StepReleaseStage, cStage, cEpoch, 0, cParity, fmt.Sprintf("release stage %d epoch %d", cStage, cEpoch))
	}

	return sched, nil
}

// Validate verifies the pipeline schedule against:
//   - Barrier initialization before use
//   - Transaction balance (expect_tx bytes == issued TMA bytes)
//   - Stage ownership and no overwrite before release
//   - Epoch reuse correctness (no stale epochs, strictly monotonic per stage)
//   - Phase parity alternating properly
func (s *PipelineSchedule) Validate() []ValidationError {
	if s == nil {
		return []ValidationError{{Violation: "nil schedule", Message: "schedule is nil"}}
	}
	var errs []ValidationError

	stageOwners := make([]StageOwnership, s.Stages)
	stageInited := make([]bool, s.Stages)
	stageExpectedBytes := make([]int64, s.Stages)
	stageIssuedBytes := make([]int64, s.Stages)

	for i := 0; i < s.Stages; i++ {
		stageOwners[i] = StageOwnership{
			StageIndex: i,
			Owner:      RoleProducer,
			State:      StageIdle,
			Epoch:      -1,
			Parity:     0,
		}
	}

	for _, step := range s.Steps {
		if step.Stage < 0 || step.Stage >= s.Stages {
			errs = append(errs, ValidationError{
				Line:      step.StepIndex,
				Violation: ErrStageIndexOutOfBounds,
				Message:   fmt.Sprintf("step %d references invalid stage %d (max %d)", step.StepIndex, step.Stage, s.Stages-1),
			})
			continue
		}

		switch step.Kind {
		case StepInit:
			if stageInited[step.Stage] {
				errs = append(errs, ValidationError{
					Line:      step.StepIndex,
					Violation: ErrDuplicateInit,
					Message:   fmt.Sprintf("barrier for stage %d already initialized", step.Stage),
				})
			}
			stageInited[step.Stage] = true

		case StepAcquireStage:
			if !stageInited[step.Stage] {
				errs = append(errs, ValidationError{
					Line:      step.StepIndex,
					Violation: ErrUninitializedBarrier,
					Message:   fmt.Sprintf("stage %d acquired before barrier initialization", step.Stage),
				})
			}
			cur := &stageOwners[step.Stage]
			// Cannot overwrite an active stage that hasn't been released by consumer
			if cur.State == StageProducing || cur.State == StageReady || cur.State == StageConsuming {
				errs = append(errs, ValidationError{
					Line:      step.StepIndex,
					Violation: ErrStageOverwrite,
					Message:   fmt.Sprintf("producer overwrote stage %d while in state %s (epoch %d)", step.Stage, cur.State, cur.Epoch),
				})
			}
			// Epoch must strictly advance
			if step.Epoch <= cur.Epoch {
				errs = append(errs, ValidationError{
					Line:      step.StepIndex,
					Violation: ErrEpochReuseStale,
					Message:   fmt.Sprintf("stale or duplicate epoch %d on stage %d (current epoch %d)", step.Epoch, step.Stage, cur.Epoch),
				})
			}
			cur.Owner = RoleProducer
			cur.State = StageProducing
			cur.Epoch = step.Epoch
			cur.Parity = step.Parity
			stageExpectedBytes[step.Stage] = 0
			stageIssuedBytes[step.Stage] = 0

		case StepArriveExpectTx:
			if !stageInited[step.Stage] {
				errs = append(errs, ValidationError{
					Line:      step.StepIndex,
					Violation: ErrUninitializedBarrier,
					Message:   fmt.Sprintf("arrive expect_tx on uninitialized barrier for stage %d", step.Stage),
				})
			}
			stageExpectedBytes[step.Stage] += step.Bytes

		case StepIssueTMA:
			if !stageInited[step.Stage] {
				errs = append(errs, ValidationError{
					Line:      step.StepIndex,
					Violation: ErrUninitializedBarrier,
					Message:   fmt.Sprintf("TMA copy issued to uninitialized barrier for stage %d", step.Stage),
				})
			}
			stageIssuedBytes[step.Stage] += step.Bytes
			// If expected bytes matches issued bytes, stage transitions to ready
			if stageIssuedBytes[step.Stage] == stageExpectedBytes[step.Stage] {
				stageOwners[step.Stage].State = StageReady
			}

		case StepWaitParity:
			if !stageInited[step.Stage] {
				errs = append(errs, ValidationError{
					Line:      step.StepIndex,
					Violation: ErrUninitializedBarrier,
					Message:   fmt.Sprintf("wait on uninitialized barrier for stage %d", step.Stage),
				})
			}
			cur := &stageOwners[step.Stage]
			expectedParity := cur.Epoch & 1
			if step.Parity != expectedParity {
				errs = append(errs, ValidationError{
					Line:      step.StepIndex,
					Violation: ErrBarrierPhaseParity,
					Message:   fmt.Sprintf("consumer waiting on mismatched parity %d for stage %d (epoch %d expects parity %d)", step.Parity, step.Stage, cur.Epoch, expectedParity),
				})
			}
			// Check transaction balance before consumer proceeds
			if stageExpectedBytes[step.Stage] != stageIssuedBytes[step.Stage] {
				errs = append(errs, ValidationError{
					Line:      step.StepIndex,
					Violation: ErrTransactionUnderflow,
					Message:   fmt.Sprintf("transaction balance mismatch on stage %d: expected %d bytes, issued %d bytes", step.Stage, stageExpectedBytes[step.Stage], stageIssuedBytes[step.Stage]),
				})
			}
			cur.State = StageConsuming
			cur.Owner = RoleConsumer

		case StepConsumeStage:
			cur := &stageOwners[step.Stage]
			if cur.State != StageConsuming {
				errs = append(errs, ValidationError{
					Line:      step.StepIndex,
					Violation: ErrStageNotReady,
					Message:   fmt.Sprintf("consumer consumed stage %d when not in consuming state (state %s)", step.Stage, cur.State),
				})
			}

		case StepReleaseStage:
			cur := &stageOwners[step.Stage]
			if cur.State != StageConsuming {
				errs = append(errs, ValidationError{
					Line:      step.StepIndex,
					Violation: ErrStageNotReady,
					Message:   fmt.Sprintf("stage %d released when not in consuming state (state %s)", step.Stage, cur.State),
				})
			}
			cur.State = StageIdle
		}
	}

	return errs
}
