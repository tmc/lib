package main

import (
	"context"
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

// PipelineStep encapsulates a configured kernel launcher and its device argument pointers.
type PipelineStep struct {
	Name     string
	Launcher *cudalaunch.Launcher
	Args     []any
}

// graphResult records statistics about CUDA Graph creation and execution.
type graphResult struct {
	Mode      string `json:"mode"`
	NodeCount int    `json:"node_count"`
	Passes    int    `json:"passes"`
	PDLEdges  int    `json:"pdl_edges,omitempty"`
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
	if stream == nil {
		return nil, nil, errors.New("graph workload: nil stream")
	}
	if len(steps) == 0 {
		return nil, nil, errors.New("graph workload: no pipeline steps")
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
		g, gErr = cudagraph.New()
		if gErr != nil {
			return nil, nil, fmt.Errorf("create pdl graph: %w", gErr)
		}
		// Create empty nodes for each pipeline step and connect them with ProgrammaticEdgeData
		nodes := make([]*cudagraph.Node, len(steps))
		for i := range steps {
			n, err := g.AddEmptyNode()
			if err != nil {
				return nil, nil, fmt.Errorf("add node %d: %w", i, err)
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
	defer func() { _ = g.Destroy() }()

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
