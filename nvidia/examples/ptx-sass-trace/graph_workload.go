package main

import (
	"context"
	"errors"
	"fmt"

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

// stepDescription resolves the launch a PipelineStep performs. Explicit step
// fields override the launcher's configuration; zero fields defer to it.
func stepDescription(step PipelineStep) (cudalaunch.Description, error) {
	var desc cudalaunch.Description
	if step.Launcher != nil {
		d, err := step.Launcher.Describe()
		if err != nil && step.Func == 0 {
			return cudalaunch.Description{}, err
		}
		desc = d
	}
	if step.Func != 0 {
		desc.Function = step.Func
	}
	if step.GridDimX != 0 || step.GridDimY != 0 || step.GridDimZ != 0 {
		desc.Grid = cudalaunch.Dim3{X: step.GridDimX, Y: step.GridDimY, Z: step.GridDimZ}
	}
	if step.BlockDimX != 0 || step.BlockDimY != 0 || step.BlockDimZ != 0 {
		desc.Block = cudalaunch.Dim3{X: step.BlockDimX, Y: step.BlockDimY, Z: step.BlockDimZ}
	}
	if step.SharedMemBytes != 0 {
		desc.SharedMem = step.SharedMemBytes
	}
	if desc.Function == 0 {
		return cudalaunch.Description{}, errors.New("kernel function is null")
	}
	return desc, nil
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
) (_ *graphResult, _ []int64, err error) {
	// retainCleanup keeps a failed teardown visible: a leaked event, graph or
	// executable graph is a real error even when the workload itself ran.
	retainCleanup := func(what string, closeErr error) {
		if closeErr != nil && err == nil {
			err = fmt.Errorf("destroy %s: %w", what, closeErr)
		}
	}
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
	defer func() { retainCleanup("start event", start.Destroy()) }()

	end, err := dev.NewEvent()
	if err != nil {
		return nil, nil, err
	}
	defer func() { retainCleanup("end event", end.Destroy()) }()

	var g *cudagraph.Graph
	defer func() {
		if g != nil {
			retainCleanup("graph", g.Destroy())
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
			desc, err := stepDescription(step)
			if err != nil {
				return nil, nil, fmt.Errorf("step %d (%s) launch description: %w", i, step.Name, err)
			}
			// AddLaunch retains the packed arguments and the module for the
			// node's native lifetime, so nothing here has to stay reachable.
			n, err := g.AddLaunch(desc, step.Args)
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
	defer func() { retainCleanup("executable graph", exec.Destroy()) }()

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
