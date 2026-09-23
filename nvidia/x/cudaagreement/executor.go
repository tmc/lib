package cudaagreement

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudaplan"
)

// An Executor submits a plan and reports which backend executed each step.
// The report is built while submitting, from the path taken, and describes what
// ran rather than what was asked for.
type Executor interface {
	// Requested reports the backend the executor was asked to use.
	Requested() cudaplan.TargetBackend

	// Execute submits plan's steps on stream against the named buffers. The
	// returned record covers every step. Execute does not synchronize: the
	// caller must drain the stream before reading any output buffer.
	Execute(stream *cuda.Stream, plan *Plan, bufs map[string]Buffer) (*Record, error)
}

// Modules loads and owns one CUDA module per distinct artifact so that the
// arms of a run share the compiled code and differ only in how it is launched.
// The zero value is not usable; use NewModules.
type Modules struct {
	dev     *cuda.Device
	loaded  map[*cudakernel.Artifact]*cuda.Module
	pending []driver.Module
}

// NewModules returns a module cache bound to dev.
func NewModules(dev *cuda.Device) *Modules {
	return &Modules{dev: dev, loaded: make(map[*cudakernel.Artifact]*cuda.Module)}
}

// Function returns the entry point of art, loading its module on first use.
func (m *Modules) Function(art *cudakernel.Artifact) (*cuda.Kernel, error) {
	if m == nil || m.dev == nil {
		return nil, errors.New("cudaagreement: module cache has no device")
	}
	if art == nil {
		return nil, errors.New("cudaagreement: nil artifact")
	}
	mod, ok := m.loaded[art]
	if !ok {
		raw, err := art.Load(m.dev)
		if raw != 0 {
			m.pending = append(m.pending, raw)
		}
		if err != nil {
			return nil, fmt.Errorf("cudaagreement: load %s: %w", art.EntryPoint, err)
		}
		mod, err = cuda.OwnModule(m.dev, raw)
		if err != nil {
			return nil, fmt.Errorf("cudaagreement: own module %s: %w", art.EntryPoint, err)
		}
		m.loaded[art] = mod
		m.pending = m.pending[:len(m.pending)-1]
	}
	return mod.Function(art.EntryPoint)
}

// Close unloads every module the cache owns. A module that fails to unload is
// retained for another Close.
func (m *Modules) Close() error {
	if m == nil {
		return nil
	}
	for art, mod := range m.loaded {
		if err := mod.Close(); err != nil {
			return fmt.Errorf("cudaagreement: close module %s: %w", art.EntryPoint, err)
		}
		delete(m.loaded, art)
	}
	for len(m.pending) > 0 {
		i := len(m.pending) - 1
		if err := m.dev.WithContext(func() error { return m.pending[i].Unload() }); err != nil {
			return fmt.Errorf("cudaagreement: unload module: %w", err)
		}
		m.pending = m.pending[:i]
	}
	return nil
}

// stepArgs resolves a step's buffers into launch arguments in schema order and
// returns the output buffer it will be recorded against.
func stepArgs(s Step, bufs map[string]Buffer) ([]any, Buffer, error) {
	names := append(append(make([]string, 0, len(s.Inputs)+1), s.Inputs...), s.Output)
	args := make([]any, 0, len(names))
	for _, name := range names {
		b, ok := bufs[name]
		if !ok || b.Ptr == 0 || b.Bytes <= 0 {
			return nil, Buffer{}, fmt.Errorf("cudaagreement: step %s has no buffer bound for %s", s.Name, name)
		}
		args = append(args, b.Ptr)
	}
	if _, err := s.Artifact.Schema.Pack(args...); err != nil {
		return nil, Buffer{}, fmt.Errorf("cudaagreement: arguments for %s: %w", s.Name, err)
	}
	return args, bufs[s.Output], nil
}

func recordFor(s Step, requested, executed cudaplan.TargetBackend, declined string, out Buffer) StepRecord {
	return StepRecord{
		Step:       s.Name,
		Requested:  requested,
		Executed:   executed,
		Declined:   declined,
		Digest:     s.Artifact.Digest,
		EntryPoint: s.Artifact.EntryPoint,
		Grid:       s.Grid,
		Block:      s.Block,
		Output:     out,
	}
}

// LaunchExecutor dispatches every step with an ordinary stream launch.
type LaunchExecutor struct {
	Modules *Modules
}

// Requested reports [cudaplan.BackendLaunch].
func (e *LaunchExecutor) Requested() cudaplan.TargetBackend { return cudaplan.BackendLaunch }

// Execute submits each step in order on stream.
func (e *LaunchExecutor) Execute(stream *cuda.Stream, plan *Plan, bufs map[string]Buffer) (*Record, error) {
	if err := plan.Validate(); err != nil {
		return nil, err
	}
	if stream == nil {
		return nil, errors.New("cudaagreement: nil stream")
	}
	rec := &Record{Arm: plan.Name + "/launch", Requested: cudaplan.BackendLaunch}
	for _, s := range plan.Steps {
		out, err := e.launch(stream, s, bufs)
		if err != nil {
			return rec, err
		}
		rec.Steps = append(rec.Steps, recordFor(s, cudaplan.BackendLaunch, cudaplan.BackendLaunch, "", out))
	}
	return rec, nil
}

func (e *LaunchExecutor) launch(stream *cuda.Stream, s Step, bufs map[string]Buffer) (Buffer, error) {
	return launchStep(e.Modules, stream, s, bufs)
}

func launchStep(mods *Modules, stream *cuda.Stream, s Step, bufs map[string]Buffer) (Buffer, error) {
	args, out, err := stepArgs(s, bufs)
	if err != nil {
		return Buffer{}, err
	}
	fn, err := mods.Function(s.Artifact)
	if err != nil {
		return Buffer{}, err
	}
	if err := cudalaunch.OwnedKernel(fn).Grid(s.Grid).Block(s.Block).On(stream).Launch(args...); err != nil {
		return Buffer{}, fmt.Errorf("cudaagreement: launch %s: %w", s.Name, err)
	}
	return out, nil
}

// GraphExecutor admits each step to a CUDA graph and replays it. A step the
// graph will not take is run with an ordinary launch instead, in its place in
// the chain: the graph built so far is instantiated and launched first, the
// declining step is launched on the stream, and a fresh graph collects the
// rest. Order is preserved and the record says which steps took which path.
type GraphExecutor struct {
	Device  *cuda.Device
	Modules *Modules

	// Decline, when non-nil, reports why a step may not enter the graph. It
	// exists so that a caller can exercise the fallback path on demand;
	// without it a step declines only when the driver or the graph refuses
	// it. A nil error admits the step.
	Decline func(Step) error

	graphs []*cudagraph.Graph
	execs  []*cudagraph.Exec
}

// Requested reports [cudaplan.BackendGraph].
func (e *GraphExecutor) Requested() cudaplan.TargetBackend { return cudaplan.BackendGraph }

// graphAvailable reports whether the driver exposes the graph entry points
// this executor needs. Mirrors the check in cudaplan.
func graphAvailable() bool {
	return driver.HasGraphCreate() && driver.HasGraphAddKernelNode() &&
		driver.HasGraphInstantiate() && driver.HasGraphLaunch() &&
		driver.HasGraphExecDestroy() && driver.HasGraphDestroy()
}

// Execute submits plan on stream, replaying as much of it as the graph accepts.
func (e *GraphExecutor) Execute(stream *cuda.Stream, plan *Plan, bufs map[string]Buffer) (*Record, error) {
	if err := plan.Validate(); err != nil {
		return nil, err
	}
	if stream == nil {
		return nil, errors.New("cudaagreement: nil stream")
	}
	if e.Device == nil {
		return nil, errors.New("cudaagreement: graph executor has no device")
	}
	rec := &Record{Arm: plan.Name + "/graph", Requested: cudaplan.BackendGraph}

	var graph *cudagraph.Graph
	var prev *cudagraph.Node
	// flush instantiates and launches the segment accumulated so far, so
	// that a step which falls back still runs after its predecessors.
	flush := func() error {
		if graph == nil {
			return nil
		}
		exec, err := graph.Instantiate()
		if err != nil {
			return fmt.Errorf("cudaagreement: instantiate graph: %w", err)
		}
		e.execs = append(e.execs, exec)
		graph, prev = nil, nil
		if err := exec.Launch(stream); err != nil {
			return fmt.Errorf("cudaagreement: replay graph: %w", err)
		}
		return nil
	}

	for _, s := range plan.Steps {
		reason := e.declineReason(s)
		if reason == nil {
			node, out, err := e.addStep(&graph, prev, s, bufs)
			switch {
			case err == nil:
				prev = node
				rec.Steps = append(rec.Steps, recordFor(s, cudaplan.BackendGraph, cudaplan.BackendGraph, "", out))
				continue
			case errors.Is(err, errDeclined):
				reason = errors.Unwrap(err)
			default:
				return rec, err
			}
		}
		if err := flush(); err != nil {
			return rec, err
		}
		out, err := launchStep(e.Modules, stream, s, bufs)
		if err != nil {
			return rec, err
		}
		rec.Steps = append(rec.Steps, recordFor(s, cudaplan.BackendGraph, cudaplan.BackendLaunch, reason.Error(), out))
	}
	if err := flush(); err != nil {
		return rec, err
	}
	return rec, nil
}

// errDeclined marks an admission failure that the executor answers with a
// fallback rather than with an error to the caller.
var errDeclined = errors.New("declined")

func (e *GraphExecutor) declineReason(s Step) error {
	if !graphAvailable() {
		return errors.New("driver has no graph entry points")
	}
	if e.Decline != nil {
		return e.Decline(s)
	}
	return nil
}

func (e *GraphExecutor) addStep(graph **cudagraph.Graph, prev *cudagraph.Node, s Step, bufs map[string]Buffer) (*cudagraph.Node, Buffer, error) {
	args, out, err := stepArgs(s, bufs)
	if err != nil {
		return nil, Buffer{}, err
	}
	fn, err := e.Modules.Function(s.Artifact)
	if err != nil {
		return nil, Buffer{}, err
	}
	if *graph == nil {
		g, err := cudagraph.New(e.Device)
		if err != nil {
			return nil, Buffer{}, fmt.Errorf("%w: %w", errDeclined, err)
		}
		e.graphs = append(e.graphs, g)
		*graph = g
	}
	var deps []*cudagraph.Node
	if prev != nil {
		deps = append(deps, prev)
	}
	node, err := (*graph).AddLauncher(cudalaunch.OwnedKernel(fn).Grid(s.Grid).Block(s.Block), args, deps...)
	if err != nil {
		return nil, Buffer{}, fmt.Errorf("%w: %w", errDeclined, err)
	}
	return node, out, nil
}

// Close destroys the graphs and graph executables the executor built. The
// caller must drain the stream first.
func (e *GraphExecutor) Close() error {
	for len(e.execs) > 0 {
		i := len(e.execs) - 1
		if err := e.execs[i].Destroy(); err != nil {
			return fmt.Errorf("cudaagreement: destroy graph exec: %w", err)
		}
		e.execs = e.execs[:i]
	}
	for len(e.graphs) > 0 {
		i := len(e.graphs) - 1
		if err := e.graphs[i].Destroy(); err != nil {
			return fmt.Errorf("cudaagreement: destroy graph: %w", err)
		}
		e.graphs = e.graphs[:i]
	}
	return nil
}
