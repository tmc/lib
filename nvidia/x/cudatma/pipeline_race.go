package cudatma

import (
	"fmt"
)

// PipelineHazard represents an ordering or synchronization defect in an asynchronous copy pipeline.
type PipelineHazard struct {
	StageID     int
	Kind        string
	Description string
}

// StageState tracks the lifecycle of an individual buffer stage in an async pipeline.
type StageState struct {
	StageID          int
	BufferBytes      uint64
	InFlight         bool
	GroupCommitted   bool
	ExpectedTxBytes  uint64
	CompletedTxBytes uint64
	BarrierArrived   bool
	BarrierPhase     int
	HasPendingWrites bool
}

// AsyncCopyPipeline models multi-stage producer-consumer asynchronous copy buffers and barriers.
type AsyncCopyPipeline struct {
	numStages     int
	stageBytes    uint64
	stages        []StageState
	inFlightGroup int
	hazards       []PipelineHazard
}

// NewAsyncCopyPipeline initializes an async copy pipeline state machine with the given number of stages.
func NewAsyncCopyPipeline(stages int, stageBytes uint64) (*AsyncCopyPipeline, error) {
	if stages <= 0 {
		return nil, fmt.Errorf("pipeline stages must be positive, got %d", stages)
	}
	if stageBytes == 0 {
		return nil, fmt.Errorf("pipeline stage bytes must be positive")
	}

	st := make([]StageState, stages)
	for i := 0; i < stages; i++ {
		st[i] = StageState{
			StageID:     i,
			BufferBytes: stageBytes,
		}
	}

	return &AsyncCopyPipeline{
		numStages:  stages,
		stageBytes: stageBytes,
		stages:     st,
		hazards:    make([]PipelineHazard, 0),
	}, nil
}

// NumStages returns the number of stages in the pipeline.
func (p *AsyncCopyPipeline) NumStages() int {
	if p == nil {
		return 0
	}
	return p.numStages
}

// IssueCopy marks an async copy issued for the given stage buffer.
func (p *AsyncCopyPipeline) IssueCopy(stage int, bytes uint64) error {
	if p == nil {
		return fmt.Errorf("nil async copy pipeline")
	}
	if stage < 0 || stage >= p.numStages {
		return fmt.Errorf("invalid stage %d (numStages=%d)", stage, p.numStages)
	}
	if bytes == 0 || bytes > p.stageBytes {
		return fmt.Errorf("invalid copy size %d for stage capacity %d", bytes, p.stageBytes)
	}

	s := &p.stages[stage]
	if s.InFlight {
		p.hazards = append(p.hazards, PipelineHazard{
			StageID:     stage,
			Kind:        "WAW",
			Description: fmt.Sprintf("write-after-write hazard: issued async copy into stage %d while prior copy is still in flight", stage),
		})
	}

	s.InFlight = true
	s.GroupCommitted = false
	s.HasPendingWrites = true
	return nil
}

// CommitGroup commits in-flight cp.async operations into a completion group.
func (p *AsyncCopyPipeline) CommitGroup() error {
	if p == nil {
		return fmt.Errorf("nil async copy pipeline")
	}
	committedAny := false
	for i := range p.stages {
		if p.stages[i].InFlight && !p.stages[i].GroupCommitted {
			p.stages[i].GroupCommitted = true
			committedAny = true
		}
	}
	if committedAny {
		p.inFlightGroup++
	}
	return nil
}

// WaitGroup waits until at most remaining commit groups are in flight.
// For cp.async.wait_group N, all groups beyond N are waited on and retired.
func (p *AsyncCopyPipeline) WaitGroup(remaining int) error {
	if p == nil {
		return fmt.Errorf("nil async copy pipeline")
	}
	if remaining < 0 {
		return fmt.Errorf("wait_group count cannot be negative: %d", remaining)
	}

	for p.inFlightGroup > remaining {
		// Retire the oldest committed stages.
		for i := range p.stages {
			if p.stages[i].InFlight && p.stages[i].GroupCommitted {
				p.stages[i].InFlight = false
				p.stages[i].HasPendingWrites = false
				p.stages[i].BarrierArrived = true
			}
		}
		p.inFlightGroup--
	}
	return nil
}

// ArriveExpectTx signals mbarrier transaction arrival for TMA / cp.async.bulk.
func (p *AsyncCopyPipeline) ArriveExpectTx(stage int, txBytes uint64) error {
	if p == nil {
		return fmt.Errorf("nil async copy pipeline")
	}
	if stage < 0 || stage >= p.numStages {
		return fmt.Errorf("invalid stage %d", stage)
	}
	s := &p.stages[stage]
	s.ExpectedTxBytes += txBytes
	s.InFlight = true
	s.BarrierArrived = false
	return nil
}

// CompleteTx signals completion of transferred transaction bytes by TMA.
func (p *AsyncCopyPipeline) CompleteTx(stage int, txBytes uint64) error {
	if p == nil {
		return fmt.Errorf("nil async copy pipeline")
	}
	if stage < 0 || stage >= p.numStages {
		return fmt.Errorf("invalid stage %d", stage)
	}
	s := &p.stages[stage]
	s.CompletedTxBytes += txBytes
	if s.ExpectedTxBytes > 0 && s.CompletedTxBytes >= s.ExpectedTxBytes {
		s.BarrierArrived = true
	}
	return nil
}

// WaitParity waits for the mbarrier phase parity.
func (p *AsyncCopyPipeline) WaitParity(stage int, phase int) error {
	if p == nil {
		return fmt.Errorf("nil async copy pipeline")
	}
	if stage < 0 || stage >= p.numStages {
		return fmt.Errorf("invalid stage %d", stage)
	}
	s := &p.stages[stage]
	if !s.BarrierArrived || s.CompletedTxBytes < s.ExpectedTxBytes {
		p.hazards = append(p.hazards, PipelineHazard{
			StageID:     stage,
			Kind:        "WaitFailure",
			Description: fmt.Sprintf("mbarrier wait on stage %d before transaction completed (expected %d, got %d)", stage, s.ExpectedTxBytes, s.CompletedTxBytes),
		})
	}
	s.InFlight = false
	s.HasPendingWrites = false
	s.BarrierPhase = phase
	return nil
}

// ReadStage models a consumer thread reading from a stage buffer.
func (p *AsyncCopyPipeline) ReadStage(stage int) error {
	if p == nil {
		return fmt.Errorf("nil async copy pipeline")
	}
	if stage < 0 || stage >= p.numStages {
		return fmt.Errorf("invalid stage %d", stage)
	}
	s := &p.stages[stage]
	if s.InFlight || s.HasPendingWrites {
		p.hazards = append(p.hazards, PipelineHazard{
			StageID:     stage,
			Kind:        "RAW",
			Description: fmt.Sprintf("read-after-write hazard: consumer read from stage %d while async copy is still in flight", stage),
		})
	}
	return nil
}

// WriteStage models a producer or thread writing into a stage buffer.
func (p *AsyncCopyPipeline) WriteStage(stage int) error {
	if p == nil {
		return fmt.Errorf("nil async copy pipeline")
	}
	if stage < 0 || stage >= p.numStages {
		return fmt.Errorf("invalid stage %d", stage)
	}
	s := &p.stages[stage]
	if s.InFlight {
		p.hazards = append(p.hazards, PipelineHazard{
			StageID:     stage,
			Kind:        "WAR",
			Description: fmt.Sprintf("write-after-read hazard: concurrent write into stage %d while transfer is in flight", stage),
		})
	}
	s.HasPendingWrites = true
	return nil
}

// CheckHazards returns a list of all hazards detected during pipeline execution.
func (p *AsyncCopyPipeline) CheckHazards() []PipelineHazard {
	if p == nil {
		return nil
	}
	return append([]PipelineHazard(nil), p.hazards...)
}

// PipelineScenario describes a configured test sequence evaluating async copy races.
type PipelineScenario struct {
	Name        string
	Fixed       bool
	IsTMA       bool
	NumStages   int
	StageBytes  uint64
	HasHazards  bool
	Hazards     []PipelineHazard
	Description string
}

// GenerateAsyncCopyScenario constructs a synthetic cp.async pipeline sequence modeling cuHadron memcpy races.
// When fixed is false, the required commit/wait_group synchronization is omitted, triggering a RAW hazard.
// When fixed is true, proper synchronization is included, resulting in 0 hazards.
func GenerateAsyncCopyScenario(fixed bool) (PipelineScenario, error) {
	pipe, err := NewAsyncCopyPipeline(2, 256)
	if err != nil {
		return PipelineScenario{}, err
	}

	// Producer issues copy into stage 0
	if err := pipe.IssueCopy(0, 256); err != nil {
		return PipelineScenario{}, err
	}

	if fixed {
		// FIXED=1: properly commit and wait for copy to complete before consumer reads
		if err := pipe.CommitGroup(); err != nil {
			return PipelineScenario{}, err
		}
		if err := pipe.WaitGroup(0); err != nil {
			return PipelineScenario{}, err
		}
	} else {
		// FIXED=0: omit wait_group! Consumer immediately reads un-synchronized buffer
	}

	// Consumer reads stage 0
	if err := pipe.ReadStage(0); err != nil {
		return PipelineScenario{}, err
	}

	hazards := pipe.CheckHazards()
	return PipelineScenario{
		Name:        "memcpy/shared_readwrite_pipeline",
		Fixed:       fixed,
		IsTMA:       false,
		NumStages:   2,
		StageBytes:  256,
		HasHazards:  len(hazards) > 0,
		Hazards:     hazards,
		Description: "cp.async producer-consumer synchronization verification",
	}, nil
}

// GenerateTMAScenario constructs a synthetic TMA cp.async.bulk pipeline sequence modeling cuHadron bulkcpy races.
// When fixed is false, mbarrier parity wait is omitted before consumer reads, triggering a hazard.
// When fixed is true, mbarrier transaction expect, completion, and wait are executed cleanly.
func GenerateTMAScenario(fixed bool) (PipelineScenario, error) {
	pipe, err := NewAsyncCopyPipeline(2, 512)
	if err != nil {
		return PipelineScenario{}, err
	}

	// Stage 0: expect 512 bytes
	if err := pipe.ArriveExpectTx(0, 512); err != nil {
		return PipelineScenario{}, err
	}
	if err := pipe.CompleteTx(0, 512); err != nil {
		return PipelineScenario{}, err
	}

	if fixed {
		// FIXED=1: Wait for transaction barrier parity before reading
		if err := pipe.WaitParity(0, 1); err != nil {
			return PipelineScenario{}, err
		}
	} else {
		// FIXED=0: Omit wait parity! Reader consumes stage 0 while barrier is un-waited
	}

	// Consumer reads stage 0
	if err := pipe.ReadStage(0); err != nil {
		return PipelineScenario{}, err
	}

	hazards := pipe.CheckHazards()
	return PipelineScenario{
		Name:        "bulkcpy/tma_transaction_barrier",
		Fixed:       fixed,
		IsTMA:       true,
		NumStages:   2,
		StageBytes:  512,
		HasHazards:  len(hazards) > 0,
		Hazards:     hazards,
		Description: "TMA mbarrier transaction barrier hazard verification",
	}, nil
}
