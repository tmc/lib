package cudacollective

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/tmc/lib/nvidia/driver"
)

// DoubleBuffer holds two ping-pong slots for pipelined transfers.
type DoubleBuffer struct {
	Slots [2]driver.DevicePtr
	Bytes int
}

// PipelineManager coordinates double-buffered pipelined transfers with backpressure control.
type PipelineManager struct {
	mu           sync.Mutex
	comm         *Communicator
	buf          DoubleBuffer
	activeSlot   int // 0 or 1
	inFlight     [2]*InFlightTransfer
	completed    [2]bool
	maxQueueSize int
	pending      int32
}

// NewPipelineManager creates a PipelineManager for comm with double buffering on buf.
// The slot ranges [Slots[i], Slots[i]+Bytes) must not wrap the address space
// and must not intersect; adjacent ranges are allowed.
func NewPipelineManager(comm *Communicator, buf DoubleBuffer, maxQueueSize int) (*PipelineManager, error) {
	if comm == nil {
		return nil, ErrCommunicatorClosed
	}
	if buf.Slots[0] == 0 || buf.Slots[1] == 0 {
		return nil, ErrNilBuffer
	}
	if buf.Bytes <= 0 {
		return nil, ErrBufferLengthMismatch
	}
	n := uint64(buf.Bytes)
	s0, s1 := uint64(buf.Slots[0]), uint64(buf.Slots[1])
	if s0 > ^uint64(0)-n || s1 > ^uint64(0)-n {
		return nil, fmt.Errorf("%w: slot range of %d bytes wraps the address space", ErrBufferLengthMismatch, buf.Bytes)
	}
	if s0 < s1+n && s1 < s0+n {
		return nil, fmt.Errorf("%w: slots [%#x, %#x) and [%#x, %#x) intersect", ErrBufferOverlap, s0, s0+n, s1, s1+n)
	}
	if maxQueueSize <= 0 {
		maxQueueSize = 2 // Default double-buffer queue limit
	}

	return &PipelineManager{
		comm:         comm,
		buf:          buf,
		maxQueueSize: maxQueueSize,
	}, nil
}

// SubmitStage schedules a transfer stage into the next double-buffer slot.
// If the target slot is currently in flight or maxQueueSize is exceeded,
// it fails with ErrBackpressureExceeded or ErrBufferInUse rather than corrupting memory.
func (p *PipelineManager) SubmitStage(ctx context.Context, op OperationDescriptor) (*InFlightTransfer, error) {
	if p == nil || p.comm == nil {
		return nil, ErrCommunicatorClosed
	}
	p.mu.Lock()
	defer p.mu.Unlock()

	if err := p.comm.CheckAlive(); err != nil {
		return nil, err
	}
	if err := op.Validate(p.comm.Membership().WorldSize()); err != nil {
		return nil, err
	}

	// Backpressure check
	currPending := atomic.LoadInt32(&p.pending)
	if int(currPending) >= p.maxQueueSize {
		return nil, fmt.Errorf("%w: pending transfers (%d) reached capacity %d", ErrBackpressureExceeded, currPending, p.maxQueueSize)
	}

	slot := p.activeSlot
	slotAddr := uintptr(p.buf.Slots[slot])

	// Check if this slot's previous transfer is still in flight
	if prev := p.inFlight[slot]; prev != nil {
		select {
		case <-prev.Done:
			// Completed, safe to recycle slot
		default:
			return nil, fmt.Errorf("%w: cannot overwrite active transfer buffer in slot %d", ErrBufferInUse, slot)
		}
	}

	tr, err := p.comm.AcquireTransferBuffer(slotAddr, p.buf.Bytes, slot)
	if err != nil {
		return nil, err
	}

	p.inFlight[slot] = tr
	p.completed[slot] = false
	atomic.AddInt32(&p.pending, 1)

	// Advance ping-pong slot
	p.activeSlot = 1 - p.activeSlot

	return tr, nil
}

// CompleteStage marks a completed transfer stage, decrementing pending backpressure.
func (p *PipelineManager) CompleteStage(tr *InFlightTransfer, err error) {
	if tr == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if tr.Slot < 0 || tr.Slot >= len(p.inFlight) || p.inFlight[tr.Slot] != tr || p.completed[tr.Slot] {
		return
	}
	p.completed[tr.Slot] = true
	p.comm.ReleaseTransferBuffer(tr, err)
	atomic.AddInt32(&p.pending, -1)
}

// Drain blocks until all in-flight pipeline transfers have completed.
func (p *PipelineManager) Drain(ctx context.Context) error {
	p.mu.Lock()
	transfers := [2]*InFlightTransfer{p.inFlight[0], p.inFlight[1]}
	p.mu.Unlock()

	for slot, tr := range transfers {
		if tr == nil {
			continue
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-tr.Done:
			if tr.Err != nil {
				return fmt.Errorf("pipeline slot %d: %w", slot, tr.Err)
			}
		}
	}
	return nil
}
