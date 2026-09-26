package cudaplan

import (
	"errors"
	"fmt"
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

// WithGraphReplay requests replay of registered kernel steps as a CUDA graph.
// Matmul regions use stream capture. The first Execute fixes
// buffer addresses until RebindGraph requests rebuilding with new bindings.
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
	pendingModules   []driver.Module // Loaded modules whose context ownership was not adopted.
	cublasEngine     *cublasltx.Engine
	graphExec        *cudagraph.Exec
	graphObj         *cudagraph.Graph
	graphReplay      bool
	graphCapture     bool
	graphPointers    map[string]driver.DevicePtr
	rebuildGraph     bool
	requiredInputs   map[string]int
	requiredOutputs  map[string]int
	intermediateLoc  map[string]int // intermediate node name -> workspace offset
	inFlight         bool
	failedInvocation *Completion
	closed           bool
	staleGraph       bool
}

// Compile builds an ExecutablePlan from a validated cudair.Snapshot.
// On compilation failure, acquired resources are closed. If cleanup also fails,
// Compile returns the closed-to-execution plan with the error; the caller must
// retry its Close before releasing the device. Otherwise the error result is nil.
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
		if cfg.dev == nil || !graphAvailable() {
			return nil, fmt.Errorf("%w: graph replay requires a device and CUDA graph entry points", ErrUnsupportedBackend)
		}
		for _, node := range snap.Order() {
			if node.Kind == cudair.KindMatMul && (!driver.HasStreamBeginCapture() || !driver.HasStreamEndCapture()) {
				return nil, fmt.Errorf("%w: matmul graph replay requires stream capture entry points", ErrUnsupportedBackend)
			}
		}
	}

	plan := &ExecutablePlan{
		dev:             cfg.dev,
		graphReplay:     cfg.enableGraph,
		snap:            snap,
		modules:         make(map[*cudakernel.Artifact]*cuda.Module),
		requiredInputs:  make(map[string]int),
		requiredOutputs: make(map[string]int),
		intermediateLoc: make(map[string]int),
	}

	// Defer cleanup on any error during compilation
	defer func() {
		if err != nil {
			p, err = failedCompilation(plan, err, plan.Close)
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
		plan.graphCapture = cfg.enableGraph
		engine, err := cublasltx.NewEngine(cfg.dev, cublasltx.WithWorkspaceBytes(16*1024*1024))
		plan.cublasEngine = engine
		if err != nil {
			return nil, fmt.Errorf("cudaplan: init cublaslt engine: %w", err)
		}
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
		if cfg.dev != nil && (backend == BackendLaunch || backend == BackendGraph) && art == nil {
			return nil, fmt.Errorf("%w: no kernel artifact for %s", ErrUnsupportedBackend, n.Name)
		}
		if cfg.dev != nil && art != nil && (backend == BackendLaunch || backend == BackendGraph) {
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
				load := func() (driver.Module, error) { return art.Load(cfg.dev) }
				own := func(raw driver.Module) (*cuda.Module, error) { return cuda.OwnModule(cfg.dev, raw) }
				if err := plan.loadModule(art, load, own); err != nil {
					return nil, fmt.Errorf("cudaplan: module for %s: %w", n.Name, err)
				}
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

func (p *ExecutablePlan) loadModule(art *cudakernel.Artifact, load func() (driver.Module, error), own func(driver.Module) (*cuda.Module, error)) error {
	raw, err := load()
	if raw != 0 {
		p.pendingModules = append(p.pendingModules, raw)
	}
	if err != nil {
		return fmt.Errorf("load: %w", err)
	}
	if raw == 0 {
		return errors.New("cudaplan: loader returned no module")
	}
	mod, err := own(raw)
	if err != nil {
		return fmt.Errorf("retain context: %w", err)
	}
	p.modules[art] = mod
	p.pendingModules = p.pendingModules[:len(p.pendingModules)-1]
	return nil
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

// Steps returns the plan's dispatches in execution order.
//
// The slices each step holds are shared with the plan; treat the result as
// read-only. It is here so that a region of a compiled plan can be lowered by
// [LowerRegion] without the plan having to know what it is lowered into.
func (p *ExecutablePlan) Steps() []StepPlan {
	if p == nil {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]StepPlan(nil), p.steps...)
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

	if p.graphReplay {
		if err := p.prepareGraphOn(stream, nodePointers); err != nil {
			p.inFlight = false
			return nil, err
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

	if p.graphReplay {
		if err := p.graphExec.Launch(stream); err != nil {
			_ = event.Destroy()
			return nil, fmt.Errorf("cudaplan: replay graph: %w", err)
		}
	} else if err := p.enqueue(stream, nodePointers); err != nil {
		_ = event.Destroy()
		return nil, err
	}

	if err := event.Record(stream); err != nil {
		_ = event.Destroy()
		p.inFlight = false
		return nil, fmt.Errorf("cudaplan: record completion event: %w", err)
	}

	return newEventCompletion(p.dev, stream, event, releases)
}

func (p *ExecutablePlan) enqueue(stream *cuda.Stream, nodePointers map[string]driver.DevicePtr) error {
	for _, step := range p.steps {
		switch step.Backend {
		case BackendCuBLASLt:
			if p.cublasEngine == nil {
				return errors.New("cudaplan: cuBLASLt engine not initialized for matmul")
			}
			// Dispatch matmul
			// Look up input and output pointers
			aPtr := nodePointers[step.InputNames[0]]
			bPtr := nodePointers[step.InputNames[1]]
			dPtr := nodePointers[step.OutputName]

			// Find node shapes
			node := p.snap.NodeByID(step.NodeID)
			if node == nil {
				return fmt.Errorf("cudaplan: node %d not found in snapshot", step.NodeID)
			}
			if err := validateMatMulNode(node); err != nil {
				return err
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
				return fmt.Errorf("cudaplan: execute matmul for %s: %w", step.NodeName, err)
			}

		case BackendLaunch, BackendGraph:
			if step.Artifact != nil {
				mod := p.modules[step.Artifact]
				if mod == nil {
					return fmt.Errorf("cudaplan: module not loaded for %s", step.NodeName)
				}
				kernel, err := mod.Function(step.Artifact.EntryPoint)
				if err != nil {
					return fmt.Errorf("cudaplan: get function %s: %w", step.Artifact.EntryPoint, err)
				}

				args, blocks, err := p.kernelArguments(step, nodePointers)
				if err != nil {
					return err
				}

				launcher := cudalaunch.OwnedKernel(kernel).
					Grid(blocks, 1, 1).
					Block(256, 1, 1).
					On(stream)

				if err := launcher.Launch(args...); err != nil {
					return fmt.Errorf("cudaplan: launch %s: %w", step.NodeName, err)
				}
			}
		}
	}

	return nil
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

// RebindGraph requests rebuilding the graph with the next Execute's bindings.
// It does not modify an in-flight graph; Execute still requires completion of
// the previous invocation before replacing graph resources.
func (p *ExecutablePlan) RebindGraph() {
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.staleGraph = false
	p.rebuildGraph = true
}

// Close destroys owned resources after completion. A failed destruction retains
// that resource and its dependencies for another Close; further execution is
// forbidden once resource teardown begins. Active invocations return
// ErrInvocationInFlight and must be completed or drained first.
func (p *ExecutablePlan) Close() error {
	return p.closeWith(planCloseOps{
		run: func(fn func() error) error { return p.dev.WithContext(fn) }, unload: func(m driver.Module) error { return m.Close() },
		graphExec: (*cudagraph.Exec).Destroy, graph: (*cudagraph.Graph).Destroy,
		engine: (*cublasltx.Engine).Close, workspace: (*WorkspaceManager).Close,
		module: (*cuda.Module).Close,
	})
}

type planCloseOps struct {
	run       func(func() error) error
	unload    func(driver.Module) error
	graphExec func(*cudagraph.Exec) error
	graph     func(*cudagraph.Graph) error
	engine    func(*cublasltx.Engine) error
	workspace func(*WorkspaceManager) error
	module    func(*cuda.Module) error
}

func failedCompilation(p *ExecutablePlan, cause error, close func() error) (*ExecutablePlan, error) {
	if err := close(); err != nil {
		return p, errors.Join(cause, fmt.Errorf("cudaplan: cleanup after compile failure: %w", err))
	}
	return nil, cause
}

func (p *ExecutablePlan) closeWith(ops planCloseOps) error {
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

	if p.inFlight {
		p.mu.Unlock()
		return ErrInvocationInFlight
	}
	defer p.mu.Unlock()
	p.closed = true

	if p.graphExec != nil {
		if err := ops.graphExec(p.graphExec); err != nil {
			return fmt.Errorf("cudaplan: destroy graph exec: %w", err)
		}
		p.graphExec = nil
	}
	if p.graphObj != nil {
		if err := ops.graph(p.graphObj); err != nil {
			return fmt.Errorf("cudaplan: destroy graph object: %w", err)
		}
		p.graphObj = nil
	}

	if p.cublasEngine != nil {
		if err := ops.engine(p.cublasEngine); err != nil {
			return fmt.Errorf("cudaplan: close cublaslt engine: %w", err)
		}
		p.cublasEngine = nil
	}

	if p.workspace != nil {
		if err := ops.workspace(p.workspace); err != nil {
			return fmt.Errorf("cudaplan: close workspace: %w", err)
		}
		p.workspace = nil
	}

	if len(p.modules) > 0 {
		for artifact, mod := range p.modules {
			if mod != nil {
				if err := ops.module(mod); err != nil {
					return fmt.Errorf("cudaplan: close module: %w", err)
				}
			}
			delete(p.modules, artifact)
		}
		p.modules = nil
	}
	for len(p.pendingModules) > 0 {
		i := len(p.pendingModules) - 1
		if err := ops.run(func() error {
			if err := ops.unload(p.pendingModules[i]); err != nil {
				return err
			}
			p.pendingModules = p.pendingModules[:i]
			return nil
		}); err != nil {
			return fmt.Errorf("cudaplan: unload pending module: %w", err)
		}
	}

	return nil
}
