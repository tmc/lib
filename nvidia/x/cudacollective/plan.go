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

func (p *Plan) validateStep(op OperationDescriptor, send, recv driver.DevicePtr) (sendBytes, recvBytes int, err error) {
	if p.comm == nil {
		return 0, 0, ErrCommunicatorClosed
	}
	worldSize := p.comm.Membership().WorldSize()
	rank := p.comm.Rank()
	if err := op.Validate(worldSize); err != nil {
		return 0, 0, err
	}
	if send == 0 || recv == 0 {
		return 0, 0, ErrNilBuffer
	}
	sendBytes, recvBytes, err = op.transferBytes(worldSize)
	if err != nil {
		return 0, 0, err
	}
	sLo, sN := uint64(send), uint64(sendBytes)
	rLo, rN := uint64(recv), uint64(recvBytes)
	if sLo > ^uint64(0)-sN || rLo > ^uint64(0)-rN {
		return 0, 0, fmt.Errorf("%w: step buffer range wraps the address space", ErrBufferLengthMismatch)
	}
	sHi := sLo + sN
	rHi := rLo + rN

	wantSend, wantRecv := send, recv
	if op.Kind == KindAllGather || op.Kind == KindGather {
		wantSend = recv + driver.DevicePtr(uint64(rank)*sN)
	}
	if op.Kind == KindReduceScatter || op.Kind == KindScatter {
		wantRecv = send + driver.DevicePtr(uint64(rank)*rN)
	}
	if op.InPlace {
		if send != wantSend || recv != wantRecv || (op.Kind != KindAllGather && op.Kind != KindGather && op.Kind != KindReduceScatter && op.Kind != KindScatter && send != recv) {
			return 0, 0, fmt.Errorf("%w: invalid in-place buffer binding send=0x%x recv=0x%x", ErrBufferOverlap, send, recv)
		}
	} else if sLo < rHi && rLo < sHi {
		return 0, 0, fmt.Errorf("%w: out-of-place send [0x%x, 0x%x) and recv [0x%x, 0x%x) overlap", ErrBufferOverlap, sLo, sHi, rLo, rHi)
	}

	if p.pipeline != nil {
		slot := op.PipelineSlot
		if !op.InPlace && sN+rN > uint64(p.pipeline.buf.Bytes) {
			return 0, 0, fmt.Errorf("%w: out-of-place step requires %d bytes, exceeding slot %d capacity %d",
				ErrBufferLengthMismatch, sN+rN, slot, p.pipeline.buf.Bytes)
		}
		slotTr := InFlightTransfer{
			Slot:       slot,
			BufferAddr: uintptr(p.pipeline.buf.Slots[slot]),
			Bytes:      p.pipeline.buf.Bytes,
		}
		if err := slotTr.ValidateRange(uintptr(send), sendBytes); err != nil {
			return 0, 0, fmt.Errorf("send buffer: %w", err)
		}
		if err := slotTr.ValidateRange(uintptr(recv), recvBytes); err != nil {
			return 0, 0, fmt.Errorf("recv buffer: %w", err)
		}
	}
	return sendBytes, recvBytes, nil
}

// AddCollectiveStep appends an explicit collective operation to the plan.
func (p *Plan) AddCollectiveStep(name string, op OperationDescriptor, send, recv driver.DevicePtr) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return ErrCommunicatorClosed
	}
	if _, _, err := p.validateStep(op, send, recv); err != nil {
		return err
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
		if step.Slot != step.Descriptor.PipelineSlot {
			return fmt.Errorf("step %s: %w: step slot %d does not match descriptor pipeline slot %d",
				step.Name, ErrBufferUnbound, step.Slot, step.Descriptor.PipelineSlot)
		}
		sendBytes, recvBytes, err := p.validateStep(step.Descriptor, step.SendBuffer, step.RecvBuffer)
		if err != nil {
			return fmt.Errorf("step %s: %w", step.Name, err)
		}
		if p.pipeline != nil {
			p.pipeline.mu.Lock()
			prevActive := p.pipeline.activeSlot
			p.pipeline.activeSlot = step.Slot
			p.pipeline.mu.Unlock()

			tr, err := p.pipeline.SubmitStage(ctx, step.Descriptor)
			if err != nil {
				p.pipeline.mu.Lock()
				p.pipeline.activeSlot = prevActive
				p.pipeline.mu.Unlock()
				return fmt.Errorf("step %s: %w", step.Name, err)
			}
			if tr.Slot != step.Slot {
				p.pipeline.CompleteStage(tr, ErrBufferUnbound)
				return fmt.Errorf("step %s: %w: reserved slot %d does not match step slot %d", step.Name, ErrBufferUnbound, tr.Slot, step.Slot)
			}
			if err := tr.ValidateRange(uintptr(step.SendBuffer), sendBytes); err != nil {
				p.pipeline.CompleteStage(tr, err)
				return fmt.Errorf("step %s send buffer: %w", step.Name, err)
			}
			if err := tr.ValidateRange(uintptr(step.RecvBuffer), recvBytes); err != nil {
				p.pipeline.CompleteStage(tr, err)
				return fmt.Errorf("step %s recv buffer: %w", step.Name, err)
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
