package cudaplan

import (
	"errors"
	"fmt"
	"maps"
	"math"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudakernel"
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
	mod := p.modules[*step.Artifact]
	if mod == nil {
		return nil, fmt.Errorf("cudaplan: missing graph module for %s", step.NodeName)
	}
	kernel, err := mod.Function(step.Artifact.EntryPoint())
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
	var scratch driver.DevicePtr
	w := step.Artifact.Resources().Workspace
	if w.Bytes != 0 {
		if p.workspace == nil {
			return nil, 0, fmt.Errorf("%w: kernel %s workspace owner", ErrMissingBinding, step.NodeName)
		}
		buffer := p.workspace.Buffer()
		var err error
		scratch, err = artifactScratchPointerAt(buffer.Ptr(), buffer.Len(), step.ScratchOffset, step.ScratchCapacity, w)
		if err != nil {
			return nil, 0, err
		}
		// Retain WorkspaceManager's closed/range checks after proving its
		// base+offset arithmetic cannot wrap or leave the owned object.
		if _, err := p.workspace.SliceAt(step.ScratchOffset, step.ScratchCapacity); err != nil {
			return nil, 0, err
		}

	}
	args, err := artifactKernelArguments(step, pointers, scratch)
	if err != nil {
		return nil, 0, err
	}

	return args, blocks, nil
}

func artifactScratchPointerAt(base driver.DevicePtr, total, offset, capacity int, w cudakernel.WorkspaceRequirements) (driver.DevicePtr, error) {
	if total < 0 || offset < 0 || capacity < 0 || offset > total || capacity > total-offset {
		return 0, fmt.Errorf("cudaplan: scratch reservation is outside workspace")
	}
	if base == 0 || uint64(offset) > ^uint64(0)-uint64(base) {
		return 0, fmt.Errorf("cudaplan: scratch workspace base plus offset overflows")
	}
	return artifactScratchPointer(base+driver.DevicePtr(offset), capacity, w)
}

// artifactScratchPointer checks the whole reservation before aligning its
// device address. A false kernel mask cannot repair an invalid pointer.
func artifactScratchPointer(base driver.DevicePtr, capacity int, w cudakernel.WorkspaceRequirements) (driver.DevicePtr, error) {
	if err := w.Validate(); err != nil {
		return 0, err
	}
	if w.Bytes == 0 {
		return 0, nil
	}
	if base == 0 || capacity < 0 {
		return 0, ErrMissingBinding
	}
	b := uint64(base)
	if uint64(capacity) > ^uint64(0)-b {
		return 0, fmt.Errorf("cudaplan: scratch reservation address overflows")
	}
	padding := (-b) & (uint64(w.Alignment) - 1)
	if padding > uint64(capacity) || w.Bytes > uint64(capacity)-padding || padding > ^uint64(0)-b {
		return 0, fmt.Errorf("cudaplan: scratch reservation does not cover aligned demand")
	}
	return driver.DevicePtr(b + padding), nil
}

// artifactKernelArguments shares ABI construction between graph and ordinary
// launches. The caller owns scratch until actual completion; this helper only
// validates and packs fixed-width device addresses.
func artifactKernelArguments(step StepPlan, pointers map[string]driver.DevicePtr, scratch driver.DevicePtr) ([]any, error) {
	if step.Artifact == nil {
		return nil, ErrUnsupportedBackend
	}
	if err := validateKernelSchema(step.Artifact, len(step.InputNames)); err != nil {
		return nil, err
	}
	w := step.Artifact.Resources().Workspace
	if w.Bytes == 0 && (step.ScratchOffset != 0 || step.ScratchCapacity != 0 || scratch != 0) {
		return nil, fmt.Errorf("cudaplan: unexpected scratch for zero-workspace artifact")
	}
	if w.Bytes != 0 && (scratch == 0 || uint64(scratch)%uint64(w.Alignment) != 0 || step.ScratchOffset < 0 || step.ScratchCapacity < 0 || uint64(step.ScratchCapacity) < w.Bytes) {
		return nil, fmt.Errorf("%w: invalid artifact scratch binding", ErrMissingBinding)
	}
	names := append(append([]string(nil), step.InputNames...), step.OutputName)
	args := make([]any, 0, len(step.Artifact.Schema().Params))
	tensor := 0
	for _, param := range step.Artifact.Schema().Params {
		if w.Bytes != 0 && param.Name == w.Parameter {
			args = append(args, scratch)
			continue
		}
		if tensor >= len(names) {
			return nil, ErrUnsupportedBackend
		}
		name := names[tensor]
		tensor++
		ptr := pointers[name]
		if ptr == 0 {
			return nil, fmt.Errorf("%w: kernel %s tensor %s", ErrMissingBinding, step.NodeName, name)
		}
		args = append(args, ptr)
	}
	if tensor != len(names) {
		return nil, ErrUnsupportedBackend
	}
	if _, err := step.Artifact.Pack(args...); err != nil {
		return nil, fmt.Errorf("cudaplan: arguments for %s: %w", step.NodeName, err)
	}
	return args, nil
}
