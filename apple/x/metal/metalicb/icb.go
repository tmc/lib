//go:build darwin

package metalicb

import (
	"fmt"
	"runtime"
	"sync"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
)

// Sequence is a fixed sequence of compute dispatches. Launch creates and CPU-
// encodes a new command buffer, waits for it to complete, and serializes with
// AddDispatch and other launches.
//
// Sequence does not own its device, pipelines, or buffers. The caller must
// keep them alive until the Sequence is no longer used.
type Sequence struct {
	dev     *metal.Device
	queue   apple.MTLCommandQueue
	mu      sync.Mutex
	replays int
	stages  []Dispatch
}

// Dispatch describes one compute dispatch in a Sequence.
type Dispatch struct {
	PSO     apple.MTLComputePipelineState
	Buffers []BufferBinding
	Grid    apple.MTLSize
	Group   apple.MTLSize
}

// GraphExec is the historical name for Sequence.
//
// Deprecated: use Sequence. Launches are not Metal indirect command buffers.
type GraphExec = Sequence

// Stage is the historical name for Dispatch.
//
// Deprecated: use Dispatch.
type Stage = Dispatch

// BufferBinding associates an MTLBuffer with an encoder index.
type BufferBinding struct {
	Buffer apple.MTLBuffer
	Offset uint
	Index  uint
}

// NewSequence creates an empty dispatch sequence.
func NewSequence(dev *metal.Device) (*Sequence, error) {
	if dev == nil {
		return nil, fmt.Errorf("metalicb: nil device")
	}
	return &Sequence{
		dev:   dev,
		queue: dev.Queue(),
	}, nil
}

// NewGraphExec creates an empty dispatch sequence.
//
// Deprecated: use NewSequence, which reports a nil device as an error.
func NewGraphExec(dev *metal.Device) *GraphExec {
	g, _ := NewSequence(dev)
	return g
}

// AddDispatch appends a compute dispatch to the sequence. It copies bindings,
// so later changes to the argument slice do not change the sequence.
func (g *Sequence) AddDispatch(pso apple.MTLComputePipelineState, grid, group apple.MTLSize, bindings ...BufferBinding) error {
	if grid.Width == 0 || grid.Height == 0 || grid.Depth == 0 {
		return fmt.Errorf("metalicb: grid dimensions must be positive")
	}
	if group.Width == 0 || group.Height == 0 || group.Depth == 0 {
		return fmt.Errorf("metalicb: threadgroup dimensions must be positive")
	}
	g.mu.Lock()
	defer g.mu.Unlock()

	g.stages = append(g.stages, Dispatch{
		PSO:     pso,
		Buffers: append([]BufferBinding(nil), bindings...),
		Grid:    grid,
		Group:   group,
	})
	return nil
}

// AddStage appends a compute dispatch to the sequence.
//
// Deprecated: use AddDispatch.
func (g *Sequence) AddStage(pso apple.MTLComputePipelineState, grid, group apple.MTLSize, bindings ...BufferBinding) {
	_ = g.AddDispatch(pso, grid, group, bindings...)
}

// Launch synchronously CPU-encodes and executes the sequence in one new Metal
// command buffer. It does not create or execute a Metal indirect command buffer.
func (g *Sequence) Launch() error {
	if g == nil || g.dev == nil {
		return fmt.Errorf("metalicb: nil device")
	}
	g.mu.Lock()
	defer g.mu.Unlock()

	if len(g.stages) == 0 {
		return fmt.Errorf("metalicb: no dispatches in sequence")
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	cb := g.queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()

	for _, st := range g.stages {
		enc.SetComputePipelineState(st.PSO)
		for _, b := range st.Buffers {
			enc.SetBufferWithOffsetAtIndex(b.Buffer, b.Offset, b.Index)
		}
		enc.DispatchThreadsThreadsPerThreadgroup(st.Grid, st.Group)
	}

	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()

	g.replays++
	return nil
}

// Replays returns the total number of times this graph has been replayed.
func (g *Sequence) Replays() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.replays
}
