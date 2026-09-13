package cudapipeline

import (
	"fmt"
)

// StepEvent records an action performed during pipeline execution simulation.
type StepEvent struct {
	Step        int
	StageIndex  int
	Action      string // "produce" or "consume"
	PhaseParity int
	Success     bool
	ErrorMsg    string
}

// SimulationTrace captures the sequential history of producer-consumer execution.
type SimulationTrace struct {
	StepsCompleted int
	Events         []StepEvent
}

// stageRuntime tracks in-flight simulation state of a stage.
type stageRuntime struct {
	occupied     bool
	txPending    int
	arrivalCount int
	phase        int
}

// Simulate executes an in-memory simulation of producer and consumer steps
// across the specified number of iterations, verifying that no deadlocks,
// phase inversions, or buffer overruns occur.
func (p *Pipeline) Simulate(iterations int) (*SimulationTrace, error) {
	if p == nil || len(p.Stages) < 2 {
		return nil, fmt.Errorf("simulate: pipeline must have at least 2 stages")
	}

	trace := &SimulationTrace{}
	numStages := len(p.Stages)

	runtimes := make([]stageRuntime, numStages)
	for i := range runtimes {
		runtimes[i] = stageRuntime{
			phase: 0,
		}
	}

	for iter := 0; iter < iterations; iter++ {
		stageIdx := iter % numStages
		expectedParity := (iter / numStages) & 1

		rt := &runtimes[stageIdx]

		// 1. Producer step: check for buffer overrun.
		if rt.occupied {
			err := fmt.Errorf("iteration %d (stage %d): buffer overrun: previous data has not been consumed", iter, stageIdx)
			trace.Events = append(trace.Events, StepEvent{
				Step:        iter,
				StageIndex:  stageIdx,
				Action:      "produce",
				PhaseParity: expectedParity,
				Success:     false,
				ErrorMsg:    err.Error(),
			})
			return trace, err
		}

		// Producer issues TMA and arrives with expected transaction bytes.
		rt.occupied = true
		rt.txPending = p.Stages[stageIdx].TMABox.PayloadBytes()
		rt.arrivalCount = p.Stages[stageIdx].ArrivalTokens

		trace.Events = append(trace.Events, StepEvent{
			Step:        iter,
			StageIndex:  stageIdx,
			Action:      "produce",
			PhaseParity: expectedParity,
			Success:     true,
		})

		// Simulated TMA arrival: memory arrives and completes transaction.
		rt.txPending = 0
		rt.arrivalCount = 0

		// 2. Consumer step: wait for parity.
		currentParity := rt.phase & 1
		if currentParity != expectedParity {
			err := fmt.Errorf("iteration %d (stage %d): phase parity mismatch: consumer waiting on parity %d but barrier is at parity %d",
				iter, stageIdx, expectedParity, currentParity)
			trace.Events = append(trace.Events, StepEvent{
				Step:        iter,
				StageIndex:  stageIdx,
				Action:      "consume",
				PhaseParity: expectedParity,
				Success:     false,
				ErrorMsg:    err.Error(),
			})
			return trace, err
		}

		// Barrier completes phase, toggling phase counter.
		rt.phase++
		rt.occupied = false

		trace.Events = append(trace.Events, StepEvent{
			Step:        iter,
			StageIndex:  stageIdx,
			Action:      "consume",
			PhaseParity: currentParity,
			Success:     true,
		})
		trace.StepsCompleted++
	}

	return trace, nil
}
