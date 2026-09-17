package cudaplan

import (
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cublasltx"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

// PlanOption configures plan compilation.
type PlanOption func(*planConfig) error

type planConfig struct {
	dev         *cuda.Device
	peakBudget  int
	enableGraph bool
	registry    *KernelRegistry
	// Injected failures for testing
	injectCompileErrorAfterResource bool
	injectResourceCleanupHook       func()
}

// ErrExecutionUnavailable indicates that native execution has no device or stream.
var ErrExecutionUnavailable = errors.New("cudaplan: native execution requires a device and stream")

// ErrUnsupportedBackend indicates an operation without an executable lowering.
var ErrUnsupportedBackend = errors.New("cudaplan: backend is unsupported")

// WithDevice sets the target CUDA execution device.
func WithDevice(dev *cuda.Device) PlanOption {
	return func(c *planConfig) error {
		c.dev = dev
		return nil
	}
}

// WithPeakWorkspaceBudget limits the maximum workspace memory in bytes.
func WithPeakWorkspaceBudget(bytes int) PlanOption {
	return func(c *planConfig) error {
		if bytes <= 0 {
			return fmt.Errorf("cudaplan: invalid peak workspace budget: %d", bytes)
		}
		c.peakBudget = bytes
		return nil
	}
}

// WithGraphReplay requests graph replay. Compile currently rejects an enabled request.
func WithGraphReplay(enable bool) PlanOption {
	return func(c *planConfig) error {
		c.enableGraph = enable
		return nil
	}
}

// WithKernelRegistry configures custom precompiled kernel artifacts for nodes.
func WithKernelRegistry(r *KernelRegistry) PlanOption {
	return func(c *planConfig) error {
		c.registry = r
		return nil
	}
}

// WithInjectedCompileFailure configures an intentional failure during compilation
// after at least one hardware resource has been acquired, verifying cleanup.
func WithInjectedCompileFailure(inject bool, onResourceAcquired func()) PlanOption {
	return func(c *planConfig) error {
		c.injectCompileErrorAfterResource = inject
		c.injectResourceCleanupHook = onResourceAcquired
		return nil
	}
}

// ExecutablePlan represents an instantiated execution plan with explicit ownership
// of modules, workspace buffers, and graph executables.
type ExecutablePlan struct {
	mu               sync.Mutex
	dev              *cuda.Device
	snap             *cudair.Snapshot
	steps            []StepPlan
	workspace        *WorkspaceManager
	modules          map[*cudakernel.Artifact]*cuda.Module
	cublasEngine     *cublasltx.Engine
	graphExec        *cudagraph.Exec
	graphObj         *cudagraph.Graph
	requiredInputs   map[string]int
	requiredOutputs  map[string]int
	intermediateLoc  map[string]int // intermediate node name -> workspace offset
	inFlight         bool
	failedInvocation *Completion
	closed           bool
	staleGraph       bool
}

// Compile builds an ExecutablePlan from a validated cudair.Snapshot.
// If any stage of compilation fails, all acquired resources (modules, engines, workspaces)
// are strictly cleaned up without leaking device state.
func Compile(snap *cudair.Snapshot, opts ...PlanOption) (p *ExecutablePlan, err error) {
	if snap == nil {
		return nil, errors.New("cudaplan: nil snapshot")
	}

	cfg := planConfig{
		peakBudget: 512 * 1024 * 1024, // 512 MiB default peak budget
	}
	for _, opt := range opts {
		if opt != nil {
			if err := opt(&cfg); err != nil {
				return nil, err
			}
		}
	}
	if cfg.enableGraph {
		return nil, fmt.Errorf("%w: graph replay is not implemented", ErrUnsupportedBackend)
	}

	plan := &ExecutablePlan{
		dev:             cfg.dev,
		snap:            snap,
		modules:         make(map[*cudakernel.Artifact]*cuda.Module),
		requiredInputs:  make(map[string]int),
		requiredOutputs: make(map[string]int),
		intermediateLoc: make(map[string]int),
	}

	// Defer cleanup on any error during compilation
	defer func() {
		if err != nil {
			_ = plan.Close()
			p = nil
		}
	}()

	// Analyze inputs and outputs
	for _, inp := range snap.Inputs() {
		plan.requiredInputs[inp.Name] = inp.SizeBytes()
	}
	for _, out := range snap.Outputs() {
		plan.requiredOutputs[out.Name] = out.SizeBytes()
	}

	// Calculate workspace demands
	arena := snap.ArenaPlan()
	totalScratchBytes := 0
	if arena != nil {
		totalScratchBytes = arena.TotalBytes
		if totalScratchBytes < 0 || totalScratchBytes > cfg.peakBudget {
			return nil, fmt.Errorf("%w: required %d bytes exceeds peak budget %d bytes", ErrWorkspaceBudgetExceeded, totalScratchBytes, cfg.peakBudget)
		}
		for node, offset := range arena.Offsets {
			if offset < 0 || node.SizeBytes() < 0 || offset > int(^uint(0)>>1)-node.SizeBytes() || offset+node.SizeBytes() > totalScratchBytes {
				return nil, fmt.Errorf("cudaplan: invalid workspace range for %s", node.Name)
			}
			plan.intermediateLoc[node.Name] = offset
		}
	}

	// Initialize workspace manager if device is provided
	if cfg.dev != nil {
		wm, err := NewWorkspaceManager(cfg.dev, cfg.peakBudget)
		if err != nil {
			return nil, fmt.Errorf("cudaplan: init workspace manager: %w", err)
		}
		plan.workspace = wm

		if totalScratchBytes > 0 {
			if err := wm.EnsureSize(totalScratchBytes); err != nil {
				return nil, fmt.Errorf("cudaplan: allocate workspace: %w", err)
			}
		}
	}

	if cfg.injectResourceCleanupHook != nil {
		cfg.injectResourceCleanupHook()
	}

	if cfg.injectCompileErrorAfterResource {
		return nil, errors.New("cudaplan: injected compile failure after acquiring resource")
	}

	// Check if any matmul exists; if so, create cuBLASLt engine if device is present
	hasMatMul := false
	for _, n := range snap.Order() {
		if n.Kind == cudair.KindMatMul {
			hasMatMul = true
			break
		}
	}

	if hasMatMul && cfg.dev != nil {
		engine, err := cublasltx.NewEngine(cfg.dev, cublasltx.WithWorkspaceBytes(16*1024*1024))
		if err != nil {
			return nil, fmt.Errorf("cudaplan: init cublaslt engine: %w", err)
		}
		plan.cublasEngine = engine
	}

	// Build step plans
	stepIdx := 0
	nodeStepMap := make(map[*cudair.Node]int)

	for _, n := range snap.Order() {
		if n.Kind == cudair.KindInput {
			continue
		}

		var deps []int
		for _, inp := range n.Inputs {
			if s, ok := nodeStepMap[inp]; ok {
				deps = append(deps, s)
			}
		}

		backend := SelectBackend(n.Op, cfg.enableGraph)
		var art *cudakernel.Artifact
		if cfg.registry != nil {
			art = cfg.registry.Get(n.Op)
			if art == nil {
				art = cfg.registry.Get(n.Name)
			}
		}
		if cfg.dev != nil && backend == BackendLaunch && art == nil {
			return nil, fmt.Errorf("%w: no kernel artifact for %s", ErrUnsupportedBackend, n.Name)
		}
		if cfg.dev != nil && art != nil && backend == BackendLaunch {
			if len(art.Schema.Params) != len(n.Inputs)+1 {
				return nil, fmt.Errorf("%w: artifact schema for %s has %d parameters, want %d", ErrUnsupportedBackend, n.Name, len(art.Schema.Params), len(n.Inputs)+1)
			}
			for i, param := range art.Schema.Params {
				if !param.IsPointer || param.Type != "ptr" {
					return nil, fmt.Errorf("%w: artifact schema for %s parameter %d is not a device pointer", ErrUnsupportedBackend, n.Name, i)
				}
			}
		}

		// Load module if artifact is present and device is bound
		if art != nil && cfg.dev != nil {
			if _, exists := plan.modules[art]; !exists {
				rawMod, err := art.Load(cfg.dev)
				if err != nil {
					return nil, fmt.Errorf("cudaplan: load module for %s: %w", n.Name, err)
				}
				mod, err := cuda.OwnModule(cfg.dev, rawMod)
				if err != nil {
					_ = cfg.dev.WithContext(rawMod.Unload)
					return nil, fmt.Errorf("cudaplan: own module for %s: %w", n.Name, err)
				}
				plan.modules[art] = mod
			}
		}

		inputNames := make([]string, len(n.Inputs))
		for i, inp := range n.Inputs {
			inputNames[i] = inp.Name
		}

		isOut := false
		for _, out := range snap.Outputs() {
			if out == n {
				isOut = true
				break
			}
		}

		step := StepPlan{
			NodeID:          n.ID,
			NodeName:        n.Name,
			Backend:         backend,
			Artifact:        art,
			InputNames:      inputNames,
			OutputName:      n.Name,
			IsOutput:        isOut,
			WorkspaceOffset: plan.intermediateLoc[n.Name],
			WorkspaceBytes:  n.SizeBytes(),
			Dependencies:    deps,
		}

		plan.steps = append(plan.steps, step)
		nodeStepMap[n] = stepIdx
		stepIdx++
	}

	return plan, nil
}

// RequiredInputs returns the map of input node names to required buffer byte lengths.
func (p *ExecutablePlan) RequiredInputs() map[string]int {
	if p == nil {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	res := make(map[string]int, len(p.requiredInputs))
	for k, v := range p.requiredInputs {
		res[k] = v
	}
	return res
}

// RequiredOutputs returns the map of output node names to required buffer byte lengths.
func (p *ExecutablePlan) RequiredOutputs() map[string]int {
	if p == nil {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	res := make(map[string]int, len(p.requiredOutputs))
	for k, v := range p.requiredOutputs {
		res[k] = v
	}
	return res
}

// Execute enqueues the plan's operations on the stream using the provided buffer bindings.
// It returns a Completion that tracks the asynchronous GPU execution and retains buffer
// leases until stream execution has completed.
//
// Invariant: An ExecutablePlan admits only one active invocation at a time.
// A submission error may return a non-nil Completion for work already queued.
// Wait on that completion or call Close to drain it before releasing bindings.
func (p *ExecutablePlan) Execute(stream *cuda.Stream, bindings *Bindings) (completion *Completion, retErr error) {
	if p == nil {
		return nil, errors.New("cudaplan: nil plan")
	}
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return nil, ErrClosed
	}
	if p.inFlight {
		return nil, ErrInvocationInFlight
	}
	if p.staleGraph {
		return nil, ErrStaleBinding
	}
	if bindings == nil {
		return nil, errors.New("cudaplan: nil bindings passed to Execute")
	}
	if p.dev == nil || stream == nil {
		return nil, ErrExecutionUnavailable
	}

	// Validate bindings against plan requirements
	if err := bindings.Validate(p.dev, p.requiredInputs, p.requiredOutputs); err != nil {
		return nil, fmt.Errorf("cudaplan: validate bindings: %w", err)
	}

	// Retain borrowed buffers
	p.inFlight = true
	releasePlan := func() {
		p.mu.Lock()
		p.inFlight = false
		p.mu.Unlock()
	}

	var releases []func()
	for _, b := range bindings.inputs {
		if b.Release != nil {
			releases = append(releases, b.Release)
		}
	}
	for _, b := range bindings.outputs {
		if b.Release != nil {
			releases = append(releases, b.Release)
		}
	}
	releases = append(releases, releasePlan)

	// Resolve buffer pointers for all nodes (inputs, outputs, and intermediates)
	nodePointers := make(map[string]driver.DevicePtr)
	for name, b := range bindings.inputs {
		nodePointers[name] = b.DevicePtr
	}
	for name, b := range bindings.outputs {
		nodePointers[name] = b.DevicePtr
	}

	// For non-output intermediates, resolve into workspace
	if p.workspace != nil {
		for name, offset := range p.intermediateLoc {
			if _, ok := nodePointers[name]; !ok {
				// Allocate in workspace
				size := 0
				for _, step := range p.steps {
					if step.NodeName == name {
						size = step.WorkspaceBytes
						break
					}
				}
				ptr, err := p.workspace.SliceAt(offset, size)
				if err != nil {
					p.inFlight = false
					return nil, fmt.Errorf("cudaplan: resolve workspace for %s: %w", name, err)
				}
				nodePointers[name] = ptr
			}
		}
	}

	// Record completion event
	event, err := p.dev.NewEvent()
	if err != nil {
		p.inFlight = false
		return nil, fmt.Errorf("cudaplan: create completion event: %w", err)
	}
	// A failed enqueue can leave earlier work on the stream. Preserve all
	// leases until that stream has drained, even if event recording failed.
	defer func() {
		if retErr != nil {
			p.inFlight = true
			completion = newHostCompletion(stream.Synchronize, nil, releases...)
			p.failedInvocation = completion
		}
	}()

	// Enqueue operations
	for _, step := range p.steps {
		switch step.Backend {
		case BackendCuBLASLt:
			if p.cublasEngine == nil {
				_ = event.Destroy()
				p.inFlight = false
				return nil, errors.New("cudaplan: cuBLASLt engine not initialized for matmul")
			}
			// Dispatch matmul
			// Look up input and output pointers
			aPtr := nodePointers[step.InputNames[0]]
			bPtr := nodePointers[step.InputNames[1]]
			dPtr := nodePointers[step.OutputName]

			// Find node shapes
			node := p.snap.NodeByID(step.NodeID)
			if node == nil {
				_ = event.Destroy()
				p.inFlight = false
				return nil, fmt.Errorf("cudaplan: node %d not found in snapshot", step.NodeID)
			}
			if err := validateMatMulNode(node); err != nil {
				_ = event.Destroy()
				p.inFlight = false
				return nil, err
			}
			m := node.Inputs[0].Shape[0]
			k := node.Inputs[0].Shape[1]
			n := node.Inputs[1].Shape[1]

			// cuBLASLt row-major matmul: D = A * B
			// Using MatMulRaw
			cfg := cublasltx.MatMulConfig{
				Alpha: 1.0,
			}
			// In row-major: D = A * B is D^T = B^T * A^T in column-major
			// aRows = n, aCols = k, lda = n (from B)
			// bRows = k, bCols = m, ldb = k (from A)
			// dRows = n, dCols = m, ldd = n (from D)
			if err := p.cublasEngine.MatMulRaw(stream, cfg, bPtr, n, k, n, aPtr, k, m, k, dPtr, n, m, n); err != nil {
				_ = event.Destroy()
				p.inFlight = false
				return nil, fmt.Errorf("cudaplan: execute matmul for %s: %w", step.NodeName, err)
			}

		case BackendLaunch:
			if step.Artifact != nil {
				mod := p.modules[step.Artifact]
				if mod == nil {
					_ = event.Destroy()
					p.inFlight = false
					return nil, fmt.Errorf("cudaplan: module not loaded for %s", step.NodeName)
				}
				kernel, err := mod.Function(step.Artifact.EntryPoint)
				if err != nil {
					_ = event.Destroy()
					p.inFlight = false
					return nil, fmt.Errorf("cudaplan: get function %s: %w", step.Artifact.EntryPoint, err)
				}

				// Pack arguments
				var args []any
				for _, inName := range step.InputNames {
					args = append(args, nodePointers[inName])
				}
				args = append(args, nodePointers[step.OutputName])

				node := p.snap.NodeByID(step.NodeID)
				numElems := uint32(1024)
				if node != nil {
					if node.NumElements() < 0 || uint64(node.NumElements()) > math.MaxUint32 {
						_ = event.Destroy()
						p.inFlight = false
						return nil, fmt.Errorf("%w: node %s element count overflows launch dimensions", ErrUnsupportedBackend, step.NodeName)
					}
					numElems = uint32(node.NumElements())
				}
				if len(step.Artifact.Schema.Params) != len(step.InputNames)+1 {
					_ = event.Destroy()
					p.inFlight = false
					return nil, fmt.Errorf("%w: kernel %s parameter schema has %d parameters, graph supplies %d pointers", ErrUnsupportedBackend, step.NodeName, len(step.Artifact.Schema.Params), len(step.InputNames)+1)
				}

				threads := uint32(256)
				blocks := (numElems + threads - 1) / threads
				if blocks == 0 {
					blocks = 1
				}

				launcher := cudalaunch.OwnedKernel(kernel).
					Grid(blocks).
					Block(threads).
					On(stream)

				if err := launcher.Launch(args...); err != nil {
					_ = event.Destroy()
					p.inFlight = false
					return nil, fmt.Errorf("cudaplan: launch %s: %w", step.NodeName, err)
				}
			}
		}
	}

	if err := event.Record(stream); err != nil {
		_ = event.Destroy()
		p.inFlight = false
		return nil, fmt.Errorf("cudaplan: record completion event: %w", err)
	}

	return newEventCompletion(p.dev, stream, event, releases)
}

func validateMatMulNode(node *cudair.Node) error {
	if node == nil || len(node.Inputs) != 2 || node.Inputs[0] == nil || node.Inputs[1] == nil || len(node.Inputs[0].Shape) != 2 || len(node.Inputs[1].Shape) != 2 || len(node.Shape) != 2 {
		return fmt.Errorf("%w: matmul requires two rank-2 inputs and output", ErrUnsupportedBackend)
	}
	if node.Inputs[0].Shape[1] != node.Inputs[1].Shape[0] || node.Shape[0] != node.Inputs[0].Shape[0] || node.Shape[1] != node.Inputs[1].Shape[1] {
		return fmt.Errorf("%w: matmul dimensions do not conform", ErrUnsupportedBackend)
	}
	if node.Inputs[0].Type != node.Inputs[1].Type || node.Type != node.Inputs[0].Type {
		return fmt.Errorf("%w: matmul dtypes do not match", ErrUnsupportedBackend)
	}
	if node.Type != cudair.Float32 {
		return fmt.Errorf("%w: matmul requires float32", ErrUnsupportedBackend)
	}
	return nil
}

// InvalidateGraphBindings marks instantiated CUDA Graph bindings as stale.
// Future submissions will fail with ErrStaleBinding until a new graph is captured/updated.
func (p *ExecutablePlan) InvalidateGraphBindings() {
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.staleGraph = true
}

// RebindGraph marks the graph bindings as refreshed and valid.
func (p *ExecutablePlan) RebindGraph() {
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.staleGraph = false
}

// Close destroys all owned modules, workspaces, cuBLASLt engines, and CUDA Graph executables.
func (p *ExecutablePlan) Close() error {
	if p == nil {
		return nil
	}
	p.mu.Lock()
	if pending := p.failedInvocation; pending != nil {
		p.mu.Unlock()
		if err := pending.Close(); err != nil {
			return fmt.Errorf("cudaplan: drain failed invocation: %w", err)
		}
		p.mu.Lock()
		if p.failedInvocation == pending {
			p.failedInvocation = nil
		}
	}

	if p.closed {
		p.mu.Unlock()
		return nil
	}
	if p.inFlight {
		p.mu.Unlock()
		return ErrInvocationInFlight
	}
	p.closed = true

	var errs []error

	if p.graphExec != nil {
		if err := p.graphExec.Destroy(); err != nil {
			errs = append(errs, fmt.Errorf("cudaplan: destroy graph exec: %w", err))
		}
		p.graphExec = nil
	}
	if p.graphObj != nil {
		if err := p.graphObj.Destroy(); err != nil {
			errs = append(errs, fmt.Errorf("cudaplan: destroy graph object: %w", err))
		}
		p.graphObj = nil
	}

	if p.cublasEngine != nil {
		if err := p.cublasEngine.Close(); err != nil {
			errs = append(errs, fmt.Errorf("cudaplan: close cublaslt engine: %w", err))
		}
		p.cublasEngine = nil
	}

	if p.workspace != nil {
		if err := p.workspace.Close(); err != nil {
			errs = append(errs, fmt.Errorf("cudaplan: close workspace: %w", err))
		}
		p.workspace = nil
	}

	if len(p.modules) > 0 {
		for _, mod := range p.modules {
			if mod != nil {
				if err := mod.Close(); err != nil {
					errs = append(errs, fmt.Errorf("cudaplan: close module: %w", err))
				}
			}
		}
		p.modules = nil
	}

	p.mu.Unlock()
	return errors.Join(errs...)
}
