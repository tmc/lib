package main

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

// PipelineStep encapsulates a configured kernel launcher and its device argument pointers.
type PipelineStep struct {
	Name     string
	Launcher *cudalaunch.Launcher
	Args     []any

	// Optional direct configuration overrides or standalone kernel parameters.
	Func           driver.Function
	GridDimX       uint32
	GridDimY       uint32
	GridDimZ       uint32
	BlockDimX      uint32
	BlockDimY      uint32
	BlockDimZ      uint32
	SharedMemBytes uint32
}

// graphResult records statistics about CUDA Graph creation and execution.
type graphResult struct {
	Mode      string `json:"mode"`
	NodeCount int    `json:"node_count"`
	Passes    int    `json:"passes"`
	PDLEdges  int    `json:"pdl_edges,omitempty"`
}

var (
	launcherType          = reflect.TypeOf(cudalaunch.Launcher{})
	fnOffset, hasFn       = launcherFieldOffset("fn")
	ownedOffset, hasOwned = launcherFieldOffset("owned")
	gridOffset, hasGrid   = launcherFieldOffset("grid")
	blockOffset, hasBlock = launcherFieldOffset("block")
	shmemOffset, hasShmem = launcherFieldOffset("shmem")
)

func launcherFieldOffset(name string) (uintptr, bool) {
	if f, ok := launcherType.FieldByName(name); ok {
		return f.Offset, true
	}
	return 0, false
}

func inspectLauncher(l *cudalaunch.Launcher) (driver.Function, cudalaunch.Dim3, cudalaunch.Dim3, uint32, error) {
	if l == nil {
		return 0, cudalaunch.Dim3{}, cudalaunch.Dim3{}, 0, errors.New("nil launcher")
	}

	var fn driver.Function
	var grid cudalaunch.Dim3
	var block cudalaunch.Dim3
	var shmem uint32

	// Check interface methods if present
	if p, ok := any(l).(interface{ Function() driver.Function }); ok {
		fn = p.Function()
	} else if p, ok := any(l).(interface{ Function() *driver.Function }); ok {
		if ptr := p.Function(); ptr != nil {
			fn = *ptr
		}
	} else if p, ok := any(l).(interface{ Raw() driver.Function }); ok {
		fn = p.Raw()
	}

	if p, ok := any(l).(interface{ Grid() cudalaunch.Dim3 }); ok {
		grid = p.Grid()
	}
	if p, ok := any(l).(interface{ Block() cudalaunch.Dim3 }); ok {
		block = p.Block()
	}
	if p, ok := any(l).(interface{ SharedMem() uint32 }); ok {
		shmem = p.SharedMem()
	}

	p := unsafe.Pointer(l)
	if fn == 0 && hasOwned {
		owned := *(**cuda.Kernel)(unsafe.Add(p, ownedOffset))
		if owned != nil {
			fn = owned.Raw()
		}
	}
	if fn == 0 && hasFn {
		fn = *(*driver.Function)(unsafe.Add(p, fnOffset))
	}
	if grid == (cudalaunch.Dim3{}) && hasGrid {
		grid = *(*cudalaunch.Dim3)(unsafe.Add(p, gridOffset))
	}
	if block == (cudalaunch.Dim3{}) && hasBlock {
		block = *(*cudalaunch.Dim3)(unsafe.Add(p, blockOffset))
	}
	if shmem == 0 && hasShmem {
		shmem = *(*uint32)(unsafe.Add(p, shmemOffset))
	}

	return fn, grid, block, shmem, nil
}

// stepToKernelParams converts a PipelineStep into driver.KernelNodeParams for cudagraph.AddKernelNode.
func stepToKernelParams(step PipelineStep) (*driver.KernelNodeParams, error) {
	kernelParams, err := cudalaunch.PackArgs(step.Args...)
	if err != nil {
		return nil, fmt.Errorf("pack args: %w", err)
	}

	var kparamsPtr uintptr
	if len(kernelParams) > 0 {
		kparamsPtr = uintptr(unsafe.Pointer(&kernelParams[0]))
	}

	fn := step.Func
	gridX := step.GridDimX
	gridY := step.GridDimY
	gridZ := step.GridDimZ
	blockX := step.BlockDimX
	blockY := step.BlockDimY
	blockZ := step.BlockDimZ
	shmem := step.SharedMemBytes

	if step.Launcher != nil {
		lFn, lGrid, lBlock, lShmem, err := inspectLauncher(step.Launcher)
		if err != nil && fn == 0 {
			return nil, err
		}
		if fn == 0 {
			fn = lFn
		}
		if gridX == 0 && gridY == 0 && gridZ == 0 {
			gridX, gridY, gridZ = lGrid.X, lGrid.Y, lGrid.Z
		}
		if blockX == 0 && blockY == 0 && blockZ == 0 {
			blockX, blockY, blockZ = lBlock.X, lBlock.Y, lBlock.Z
		}
		if shmem == 0 {
			shmem = lShmem
		}
	}

	if fn == 0 {
		return nil, errors.New("kernel function is null")
	}

	if gridX == 0 {
		gridX = 1
	}
	if gridY == 0 {
		gridY = 1
	}
	if gridZ == 0 {
		gridZ = 1
	}
	if blockX == 0 {
		blockX = 1
	}
	if blockY == 0 {
		blockY = 1
	}
	if blockZ == 0 {
		blockZ = 1
	}

	params := &driver.KernelNodeParams{
		Func:           fn,
		GridDimX:       gridX,
		GridDimY:       gridY,
		GridDimZ:       gridZ,
		BlockDimX:      blockX,
		BlockDimY:      blockY,
		BlockDimZ:      blockZ,
		SharedMemBytes: shmem,
		KernelParams:   kparamsPtr,
	}
	return params, nil
}

// runWorkloadInGraph executes the DeepSeek-V4 pipeline inside a captured CUDA Graph
// or Programmatic Dependent Launch (PDL) execution DAG.
func runWorkloadInGraph(
	ctx context.Context,
	dev *cuda.Device,
	stream *cuda.Stream,
	mode string,
	passes int,
	steps []PipelineStep,
) (*graphResult, []int64, error) {
	if mode != "capture" && mode != "pdl" {
		return nil, nil, fmt.Errorf("unsupported graph mode: %q", mode)
	}
	if len(steps) == 0 {
		return nil, nil, errors.New("graph workload: no pipeline steps")
	}
	if stream == nil {
		return nil, nil, errors.New("graph workload: nil stream")
	}

	start, err := dev.NewEvent()
	if err != nil {
		return nil, nil, err
	}
	defer func() { _ = start.Destroy() }()

	end, err := dev.NewEvent()
	if err != nil {
		return nil, nil, err
	}
	defer func() { _ = end.Destroy() }()

	var g *cudagraph.Graph
	defer func() {
		if g != nil {
			_ = g.Destroy()
		}
	}()

	var nodeCount int
	var pdlCount int

	switch mode {
	case "capture":
		var captureErr error
		g, err = cudagraph.Capture(stream, func() {
			for _, step := range steps {
				if lErr := step.Launcher.On(stream).Launch(step.Args...); lErr != nil && captureErr == nil {
					captureErr = lErr
				}
			}
		})
		if err != nil {
			return nil, nil, fmt.Errorf("cuda graph stream capture: %w", err)
		}
		if captureErr != nil {
			return nil, nil, fmt.Errorf("launch pipeline inside capture: %w", captureErr)
		}
		nodeCount = len(steps)

	case "pdl":
		// Build an execution DAG with Programmatic Dependent Launch (PDL) edges
		var gErr error
		g, gErr = cudagraph.New(dev)
		if gErr != nil {
			return nil, nil, fmt.Errorf("create pdl graph: %w", gErr)
		}
		// Create kernel nodes for each pipeline step and connect them with ProgrammaticEdgeData
		nodes := make([]*cudagraph.Node, len(steps))
		for i, step := range steps {
			params, err := stepToKernelParams(step)
			if err != nil {
				return nil, nil, fmt.Errorf("step %d (%s) kernel params: %w", i, step.Name, err)
			}
			n, err := g.AddKernelNode(params)
			if err != nil {
				return nil, nil, fmt.Errorf("add kernel node %d (%s): %w", i, step.Name, err)
			}
			nodes[i] = n
			if i > 0 {
				if err := g.AddDependencies([]*cudagraph.Node{nodes[i-1]}, []*cudagraph.Node{nodes[i]}, cudagraph.ProgrammaticEdgeData()); err != nil {
					return nil, nil, fmt.Errorf("add pdl edge %d->%d: %w", i-1, i, err)
				}
				pdlCount++
			}
		}
		nodeCount = len(nodes)

	default:
		return nil, nil, fmt.Errorf("unsupported graph mode: %q", mode)
	}

	exec, err := g.Instantiate()
	if err != nil {
		return nil, nil, fmt.Errorf("instantiate graph: %w", err)
	}
	defer func() { _ = exec.Destroy() }()

	// Warmup execution
	if err := exec.Launch(stream); err != nil {
		return nil, nil, fmt.Errorf("warmup graph launch: %w", err)
	}
	if err := stream.Synchronize(ctx); err != nil {
		return nil, nil, fmt.Errorf("warmup graph sync: %w", err)
	}

	// Measured passes
	intervals := make([]int64, 0, passes)
	for p := 0; p < passes; p++ {
		if err := start.Record(stream); err != nil {
			return nil, nil, err
		}
		if err := exec.Launch(stream); err != nil {
			return nil, nil, fmt.Errorf("pass %d graph launch: %w", p, err)
		}
		if err := end.Record(stream); err != nil {
			return nil, nil, err
		}
		if err := end.Synchronize(ctx); err != nil {
			return nil, nil, err
		}
		dur, err := end.ElapsedSince(start)
		if err != nil {
			return nil, nil, err
		}
		intervals = append(intervals, dur.Nanoseconds())
	}

	res := &graphResult{
		Mode:      mode,
		NodeCount: nodeCount,
		Passes:    passes,
		PDLEdges:  pdlCount,
	}

	return res, intervals, nil
}
