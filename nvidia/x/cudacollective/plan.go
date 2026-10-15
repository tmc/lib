package cudacollective

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
)

// CollectiveStep represents an explicit collective transfer step within a plan.
type CollectiveStep struct {
	StepID     int
	Name       string
	Descriptor OperationDescriptor
	SendBuffer driver.DevicePtr
	RecvBuffer driver.DevicePtr
	Slot       int
	InFlightTr *InFlightTransfer
}

// Plan coordinates execution across distributed steps and kernel execution.
type Plan struct {
	mu       sync.Mutex
	comm     *Communicator
	pipeline *PipelineManager
	steps    []CollectiveStep
	closed   bool
}

// ErrPipelineCommunicator reports a PipelineManager created for a different
// Communicator than the Plan it is given to.
var ErrPipelineCommunicator = errors.New("cudacollective: pipeline bound to a different communicator")

// NewPlan creates a distributed execution plan bound to a Communicator and PipelineManager.
// A non-nil pipeline must have been created for comm.
func NewPlan(comm *Communicator, pipeline *PipelineManager) (*Plan, error) {
	if comm == nil {
		return nil, ErrCommunicatorClosed
	}
	if pipeline != nil && pipeline.comm != comm {
		return nil, ErrPipelineCommunicator
	}
	return &Plan{
		comm:     comm,
		pipeline: pipeline,
	}, nil
}

// AddCollectiveStep appends an explicit collective operation to the plan.
func (p *Plan) AddCollectiveStep(name string, op OperationDescriptor, send, recv driver.DevicePtr) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return ErrCommunicatorClosed
	}
	if err := op.Validate(p.comm.Membership().WorldSize()); err != nil {
		return err
	}
	if send == 0 || recv == 0 {
		return ErrNilBuffer
	}

	step := CollectiveStep{
		StepID:     len(p.steps),
		Name:       name,
		Descriptor: op,
		SendBuffer: send,
		RecvBuffer: recv,
		Slot:       op.PipelineSlot,
	}
	p.steps = append(p.steps, step)
	return nil
}

// ExecuteSteps runs the collective steps sequentially, validating backpressure and generation liveness.
func (p *Plan) ExecuteSteps(ctx context.Context) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return ErrCommunicatorClosed
	}
	if err := p.comm.CheckAlive(); err != nil {
		return err
	}

	for i := range p.steps {
		step := &p.steps[i]
		if p.pipeline != nil {
			tr, err := p.pipeline.SubmitStage(ctx, step.Descriptor)
			if err != nil {
				return fmt.Errorf("step %s: %w", step.Name, err)
			}
			step.InFlightTr = tr

			// Execute via communicator transport
			err = p.comm.Transport().Execute(ctx, p.comm, step.Descriptor, step.SendBuffer, step.RecvBuffer)
			p.pipeline.CompleteStage(tr, err)
			if err != nil {
				p.comm.MarkFailed(err)
				return fmt.Errorf("step %s execute: %w", step.Name, err)
			}
		} else {
			err := p.comm.Transport().Execute(ctx, p.comm, step.Descriptor, step.SendBuffer, step.RecvBuffer)
			if err != nil {
				p.comm.MarkFailed(err)
				return fmt.Errorf("step %s execute: %w", step.Name, err)
			}
		}
	}
	return nil
}

// Close closes the plan. The caller owns the PipelineManager and Communicator
// passed to NewPlan; Close does not close or drain them.
func (p *Plan) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return nil
	}
	p.closed = true
	return nil
}
