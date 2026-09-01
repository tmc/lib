//go:build darwin

package metalicb

import (
	"fmt"
	"runtime"
	"sync"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
)

// GraphExec captures repeated GPU forward passes and replays them via reusable command encoders.
type GraphExec struct {
	dev        *metal.Device
	queue      apple.MTLCommandQueue
	mu         sync.Mutex
	replays    int
	stages     []Stage
}

// Stage represents one recorded dispatch step in an execution sequence.
type Stage struct {
	PSO     apple.MTLComputePipelineState
	Buffers []BufferBinding
	Grid    apple.MTLSize
	Group   apple.MTLSize
}

// BufferBinding associates an MTLBuffer with an encoder index.
type BufferBinding struct {
	Buffer apple.MTLBuffer
	Offset uint
	Index  uint
}

// NewGraphExec creates a new hardware execution graph.
func NewGraphExec(dev *metal.Device) *GraphExec {
	return &GraphExec{
		dev:   dev,
		queue: dev.Queue(),
	}
}

// AddStage appends a compute dispatch stage to the execution sequence.
func (g *GraphExec) AddStage(pso apple.MTLComputePipelineState, grid, group apple.MTLSize, bindings ...BufferBinding) {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.stages = append(g.stages, Stage{
		PSO:     pso,
		Buffers: bindings,
		Grid:    grid,
		Group:   group,
	})
}

// Launch replays the entire multi-layer execution graph within a single command buffer.
func (g *GraphExec) Launch() error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if len(g.stages) == 0 {
		return fmt.Errorf("metalicb: no stages recorded in graph")
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
func (g *GraphExec) Replays() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.replays
}
