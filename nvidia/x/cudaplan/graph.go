package cudaplan

import (
	"errors"
	"fmt"
	"maps"
	"math"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

func graphAvailable() bool {
	return driver.HasGraphCreate() && driver.HasGraphAddKernelNode() &&
		driver.HasGraphInstantiate() && driver.HasGraphLaunch() &&
		driver.HasGraphExecDestroy() && driver.HasGraphDestroy()
}

// prepareGraph runs with p.mu held and before any invocation is submitted.
func (p *ExecutablePlan) prepareGraph(pointers map[string]driver.DevicePtr) (err error) {
	return p.prepareGraphOn(nil, pointers)
}

func (p *ExecutablePlan) prepareGraphOn(stream *cuda.Stream, pointers map[string]driver.DevicePtr) error {
	ops := graphBuildOps{
		create:       func() (*cudagraph.Graph, error) { return cudagraph.New(p.dev) },
		destroyGraph: (*cudagraph.Graph).Destroy,
		destroyExec:  (*cudagraph.Exec).Destroy,
		instantiate:  (*cudagraph.Graph).Instantiate,
		add:          p.addGraphStep,
	}
	if p.graphCapture {
		if stream == nil {
			return ErrExecutionUnavailable
		}
		ops.captured = true
		ops.create = func() (*cudagraph.Graph, error) {
			return capturePlan(func(fn func()) (*cudagraph.Graph, error) { return cudagraph.Capture(stream, fn) }, func() error { return p.enqueue(stream, pointers) })
		}
	}
	return p.prepareGraphWith(pointers, ops)
}

func capturePlan(capture func(func()) (*cudagraph.Graph, error), enqueue func() error) (*cudagraph.Graph, error) {
	var enqueueErr error
	graph, err := capture(func() { enqueueErr = enqueue() })
	if enqueueErr != nil {
		enqueueErr = fmt.Errorf("cudaplan: capture operations: %w", enqueueErr)
	}
	return graph, errors.Join(err, enqueueErr)
}

type graphBuildOps struct {
	captured     bool
	create       func() (*cudagraph.Graph, error)
	destroyGraph func(*cudagraph.Graph) error
	destroyExec  func(*cudagraph.Exec) error
	instantiate  func(*cudagraph.Graph) (*cudagraph.Exec, error)
	add          func(*cudagraph.Graph, StepPlan, map[string]driver.DevicePtr, *cudagraph.Node) (*cudagraph.Node, error)
}

func (p *ExecutablePlan) prepareGraphWith(pointers map[string]driver.DevicePtr, ops graphBuildOps) (err error) {
	if p.graphExec != nil && !p.rebuildGraph {
		if !maps.Equal(p.graphPointers, pointers) {
			return ErrStaleBinding
		}
		return nil
	}
	// A construction or teardown failure leaves all owners on the plan for
	// Close to retry. No partial graph can subsequently be executed.
	defer func() {
		if err != nil {
			p.closed = true
		}
	}()
	if p.graphExec != nil {
		if err := ops.destroyExec(p.graphExec); err != nil {
			return fmt.Errorf("cudaplan: replace graph executable: %w", err)
		}
		p.graphExec = nil
	}
	if p.graphObj != nil {
		if err := ops.destroyGraph(p.graphObj); err != nil {
			return fmt.Errorf("cudaplan: replace graph: %w", err)
		}
		p.graphObj = nil
	}
	p.graphObj, err = ops.create()
	if err != nil {
		return err
	}
	var previous *cudagraph.Node
	steps := p.steps
	if ops.captured {
		steps = nil
	}
	for _, step := range steps {
		previous, err = ops.add(p.graphObj, step, pointers, previous)
		if err != nil {
			return fmt.Errorf("cudaplan: add graph step %s: %w", step.NodeName, err)
		}
	}
	p.graphExec, err = ops.instantiate(p.graphObj)
	if err != nil {
		return err
	}
	p.graphPointers = maps.Clone(pointers)
	p.rebuildGraph = false
	return nil
}

func (p *ExecutablePlan) addGraphStep(graph *cudagraph.Graph, step StepPlan, pointers map[string]driver.DevicePtr, previous *cudagraph.Node) (*cudagraph.Node, error) {
	if step.Backend != BackendGraph || step.Artifact == nil {
		return nil, fmt.Errorf("%w: graph step %s", ErrUnsupportedBackend, step.NodeName)
	}
	mod := p.modules[step.Artifact]
	if mod == nil {
		return nil, fmt.Errorf("cudaplan: missing graph module for %s", step.NodeName)
	}
	kernel, err := mod.Function(step.Artifact.EntryPoint)
	if err != nil {
		return nil, err
	}
	args, blocks, err := p.kernelArguments(step, pointers)
	if err != nil {
		return nil, err
	}
	var deps []*cudagraph.Node
	// Preserve the ordinary execution order, including workspace reuse
	// edges that are not tensor dependencies.
	if previous != nil {
		deps = append(deps, previous)
	}
	return graph.AddLauncher(cudalaunch.OwnedKernel(kernel).Grid(blocks, 1, 1).Block(256, 1, 1), args, deps...)
}

// kernelArguments defines the pointer ABI and geometry for both backends.
// Reduction artifacts receive the full input pointer and use output extents
// for launch geometry; their reduction axes and input shape are compiled in.
func (p *ExecutablePlan) kernelArguments(step StepPlan, pointers map[string]driver.DevicePtr) ([]any, uint32, error) {
	if p.snap == nil || step.Artifact == nil {
		return nil, 0, fmt.Errorf("%w: missing kernel snapshot or artifact", ErrUnsupportedBackend)
	}
	node := p.snap.NodeByID(step.NodeID)
	if node == nil || node.NumElements() < 0 || uint64(node.NumElements()) > math.MaxUint32 {
		return nil, 0, fmt.Errorf("%w: node %s extent", ErrUnsupportedBackend, step.NodeName)
	}
	blocks := uint32((uint64(node.NumElements()) + 255) / 256)
	if blocks == 0 {
		blocks = 1
	}
	args := make([]any, 0, len(step.InputNames)+1)
	for _, name := range append(append([]string(nil), step.InputNames...), step.OutputName) {
		ptr := pointers[name]
		if ptr == 0 {
			return nil, 0, fmt.Errorf("%w: kernel %s tensor %s", ErrMissingBinding, step.NodeName, name)
		}
		args = append(args, ptr)
	}
	if _, err := step.Artifact.Schema.Pack(args...); err != nil {
		return nil, 0, fmt.Errorf("cudaplan: arguments for %s: %w", step.NodeName, err)
	}
	return args, blocks, nil
}
