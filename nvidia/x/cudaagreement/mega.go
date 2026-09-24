package cudaagreement

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaplan"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

// MegaExecutor runs each maximal run of admissible steps as one launch of a
// [cudamega] persistent worker.
//
// A step is admissible when Operations names it, it has exactly one input,
// and its input and output buffers have the same nonzero size in whole uint32
// words. Consecutive admissible steps form one run when each step's input is
// the previous step's output and all of them have the same size; the first
// step of a run may read any bound buffer. Any other step is launched
// ordinarily in its place in the chain, and its record says why.
//
// Unlike the other executors, Execute synchronizes the stream after each
// persistent launch, because what it records is read back from the device: a
// step records [cudaplan.BackendMega] only when the worker's abort word is
// zero and its start-order buffer shows that exactly the CTAs the program
// dealt the task took it. The worker writes into scratch the executor owns;
// each step's slice is then copied device to device into the step's output
// buffer, which is the buffer the record names. Execute does not wait for
// those copies.
//
// The worker is launched as an ordinary grid no wider than the residency the
// driver reports for it; a run whose grid exceeds that residency is declined
// rather than launched, because a persistent grid with a CTA that is not
// resident deadlocks.
//
// The zero value is not usable: Device, Modules, Target, Grid and Compile are
// required.
type MegaExecutor struct {
	Device  *cuda.Device
	Modules *Modules
	Target  cudamega.Target
	Grid    cudamega.Grid

	// Operations maps a step name to the elementwise operation the worker
	// performs for it. The step's own artifact must compute the same thing;
	// agreement with the other arms is what checks that it does.
	Operations map[string]cudamega.Operation

	// Compile turns the worker's CUDA C++ into a loadable image for target.
	Compile func(src string, target cudamega.Target) (cudakernel.Kind, []byte, error)

	// Decline, as in GraphExecutor, forces a step out of the persistent
	// worker. A nil error admits the step.
	Decline func(Step) error

	// Timeout bounds the wait for one persistent launch. Zero means one
	// minute. It bounds the host's wait only; the worker's own spin limit
	// is what ends a worker that cannot make progress.
	Timeout time.Duration

	compiled map[string]*cudakernel.Artifact
	scratch  map[string]cudamem.Slice[uint32]
	order    cudamem.Slice[uint64]
	epoch    uint64
}

// Requested reports [cudaplan.BackendMega].
func (e *MegaExecutor) Requested() cudaplan.TargetBackend { return cudaplan.BackendMega }

// A megaUnit is either one declined step or a run of steps the worker
// performs in one launch.
type megaUnit struct {
	steps    []int // plan indices, in order
	declined string
	words    int // uint32 elements per task, for a run
}

// megaUnits divides plan into runs and declined steps.
func (e *MegaExecutor) megaUnits(plan *Plan, bufs map[string]Buffer) []megaUnit {
	var units []megaUnit
	var run *megaUnit
	for i, s := range plan.Steps {
		words, reason := e.admit(s, bufs)
		if reason != "" {
			units = append(units, megaUnit{steps: []int{i}, declined: reason})
			run = nil
			continue
		}
		if run != nil && run.words == words && plan.Steps[run.steps[len(run.steps)-1]].Output == s.Inputs[0] {
			run.steps = append(run.steps, i)
			continue
		}
		units = append(units, megaUnit{steps: []int{i}, words: words})
		run = &units[len(units)-1]
	}
	return units
}

// admit reports the step's size in uint32 words, or why the worker cannot
// perform it.
func (e *MegaExecutor) admit(s Step, bufs map[string]Buffer) (int, string) {
	if e.Decline != nil {
		if err := e.Decline(s); err != nil {
			return 0, err.Error()
		}
	}
	if _, ok := e.Operations[s.Name]; !ok {
		return 0, "no persistent-worker operation for step"
	}
	if len(s.Inputs) != 1 {
		return 0, fmt.Sprintf("step takes %d inputs; a worker task takes one", len(s.Inputs))
	}
	in, out := bufs[s.Inputs[0]], bufs[s.Output]
	switch {
	case in.Bytes <= 0 || out.Bytes <= 0:
		return 0, "step buffers are not bound"
	case in.Bytes != out.Bytes:
		return 0, fmt.Sprintf("input is %d bytes and output %d; a worker task maps elements one to one", in.Bytes, out.Bytes)
	case in.Bytes%4 != 0:
		return 0, fmt.Sprintf("buffers are %d bytes, not whole uint32 words", in.Bytes)
	}
	return in.Bytes / 4, ""
}

// Execute submits plan on stream, running each admissible run in one
// persistent launch and synchronizing after it.
func (e *MegaExecutor) Execute(stream *cuda.Stream, plan *Plan, bufs map[string]Buffer) (*Record, error) {
	if err := plan.Validate(); err != nil {
		return nil, err
	}
	switch {
	case stream == nil:
		return nil, errors.New("cudaagreement: nil stream")
	case e.Device == nil || e.Modules == nil:
		return nil, errors.New("cudaagreement: mega executor has no device or module cache")
	case e.Compile == nil:
		return nil, errors.New("cudaagreement: mega executor has no compiler")
	}
	rec := &Record{Arm: plan.Name + "/mega", Requested: cudaplan.BackendMega}
	for _, u := range e.megaUnits(plan, bufs) {
		if u.declined == "" {
			steps, reason, err := e.run(stream, plan, u, bufs)
			if err != nil {
				return rec, err
			}
			if reason == "" {
				rec.Steps = append(rec.Steps, steps...)
				continue
			}
			u.declined = reason
		}
		for _, i := range u.steps {
			s := plan.Steps[i]
			out, err := launchStep(e.Modules, stream, s, bufs)
			if err != nil {
				return rec, err
			}
			rec.Steps = append(rec.Steps, recordFor(s, cudaplan.BackendMega, cudaplan.BackendLaunch, u.declined, out))
		}
	}
	return rec, nil
}

// run performs one run in the persistent worker. A nonempty reason declines
// the whole run before anything is launched; an error means a launch was
// attempted and did not complete as the program says it must.
func (e *MegaExecutor) run(stream *cuda.Stream, plan *Plan, u megaUnit, bufs map[string]Buffer) ([]StepRecord, string, error) {
	prog, err := e.program(plan, u)
	if err != nil {
		return nil, err.Error(), nil
	}
	art, err := e.artifact(prog)
	if err != nil {
		return nil, "", err
	}
	fn, err := e.Modules.Function(art)
	if err != nil {
		return nil, "", err
	}
	if reason, err := e.resident(fn); err != nil || reason != "" {
		return nil, reason, err
	}

	// The worker's step order is rank order, which for a chain is plan
	// order; check rather than assume, since the scratch slices are indexed
	// by it.
	steps := prog.Steps()
	for k, s := range steps {
		if s.Task.ID != plan.Steps[u.steps[k]].Name {
			return nil, "", fmt.Errorf("cudaagreement: persistent program orders %s at %d, plan has %s", s.Task.ID, k, plan.Steps[u.steps[k]].Name)
		}
	}
	nt := len(steps)
	n := u.words
	scratch, err := e.buffer("scratch", nt*n)
	if err != nil {
		return nil, "", err
	}
	completed, err := e.buffer("completed", nt)
	if err != nil {
		return nil, "", err
	}
	abort, err := e.buffer("abort", 1)
	if err != nil {
		return nil, "", err
	}
	orderWords, err := cudamega.StartOrderWords(uint64(nt), uint64(e.Grid.CTAs))
	if err != nil {
		return nil, "", fmt.Errorf("cudaagreement: start-order buffer: %w", err)
	}
	if err := e.orderBuffer(int(orderWords)); err != nil {
		return nil, "", err
	}
	order := e.order.Slice(0, int(orderWords))
	for _, s := range []cudamem.Slice[uint32]{completed, abort} {
		if err := s.MemsetAsync(0, stream); err != nil {
			return nil, "", fmt.Errorf("cudaagreement: clear worker state: %w", err)
		}
	}
	if err := order.MemsetAsync(0, stream); err != nil {
		return nil, "", fmt.Errorf("cudaagreement: clear start order: %w", err)
	}

	input := bufs[plan.Steps[u.steps[0]].Inputs[0]]
	e.epoch++
	args := []any{input.Ptr, scratch.Ptr(), completed.Ptr(), order.Ptr(), abort.Ptr(), uint32(n), e.epoch}
	if _, err := art.Schema.Pack(args...); err != nil {
		return nil, "", fmt.Errorf("cudaagreement: persistent worker arguments: %w", err)
	}
	if err := cudalaunch.OwnedKernel(fn).Grid(e.Grid.CTAs).Block(e.Grid.ThreadsPerCTA).On(stream).Launch(args...); err != nil {
		return nil, "", fmt.Errorf("cudaagreement: launch persistent worker: %w", err)
	}
	timeout := e.Timeout
	if timeout == 0 {
		timeout = time.Minute
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	if err := stream.Synchronize(ctx); err != nil {
		return nil, "", fmt.Errorf("cudaagreement: wait for persistent worker: %w", err)
	}
	var code [1]uint32
	if err := abort.CopyToHost(code[:]); err != nil {
		return nil, "", fmt.Errorf("cudaagreement: read abort word: %w", err)
	}
	if code[0] != cudamega.AbortNone {
		return nil, "", fmt.Errorf("cudaagreement: persistent worker aborted with code %d", code[0])
	}
	ord := make([]uint64, orderWords)
	if err := order.CopyToHost(ord); err != nil {
		return nil, "", fmt.Errorf("cudaagreement: read start order: %w", err)
	}
	if err := checkStartOrder(steps, e.Grid.CTAs, ord); err != nil {
		return nil, "", err
	}

	var recs []StepRecord
	for k, i := range u.steps {
		s := plan.Steps[i]
		out := bufs[s.Output]
		if err := copyAsync(e.Device, stream, out.Ptr, scratch.Ptr()+driver.DevicePtr(4*k*n), 4*n); err != nil {
			return nil, "", fmt.Errorf("cudaagreement: copy %s from worker scratch: %w", s.Name, err)
		}
		r := recordFor(s, cudaplan.BackendMega, cudaplan.BackendMega, "", out)
		r.Program = art.Digest
		r.EntryPoint = art.EntryPoint
		r.Grid, r.Block = e.Grid.CTAs, e.Grid.ThreadsPerCTA
		recs = append(recs, r)
	}
	return recs, "", nil
}

// program builds the persistent program for a run: a chain of device-scope
// tasks in plan order.
func (e *MegaExecutor) program(plan *Plan, u megaUnit) (*cudamega.Program, error) {
	tasks := make([]cudataskgraph.Task, len(u.steps))
	ops := make(map[string]cudamega.Operation, len(u.steps))
	for k, i := range u.steps {
		s := plan.Steps[i]
		tasks[k] = cudataskgraph.Task{ID: s.Name, Rank: k, Scope: cudataskgraph.ScopeDevice}
		if k > 0 {
			tasks[k].Deps = []string{plan.Steps[u.steps[k-1]].Name}
		}
		ops[s.Name] = e.Operations[s.Name]
	}
	g, err := cudataskgraph.New(tasks)
	if err != nil {
		return nil, fmt.Errorf("task graph: %v", err)
	}
	prog, err := cudamega.NewProgram(g, ops, e.Grid, e.Target)
	if err != nil {
		return nil, fmt.Errorf("persistent program: %v", err)
	}
	return prog, nil
}

// checkStartOrder reports an error unless every task was started by exactly
// the CTAs that own it and the tickets are the permutation 1..total that a
// single device-scope counter deals. It is the device's own account of which
// worker ran which task, and the record is made from it.
func checkStartOrder(steps []cudamega.Step, ctas uint32, ord []uint64) error {
	nt := len(steps)
	if len(ord) < 1+nt*int(ctas) {
		return fmt.Errorf("cudaagreement: start order has %d words, want %d", len(ord), 1+nt*int(ctas))
	}
	var total uint64
	for _, s := range steps {
		total += uint64(s.Owner.Count())
	}
	if ord[0] != total {
		return fmt.Errorf("cudaagreement: persistent worker dealt %d start tickets, want %d", ord[0], total)
	}
	seen := make([]bool, total+1)
	for c := range ctas {
		for i, s := range steps {
			tk := ord[1+int(c)*nt+i]
			if owns := s.Owner.Contains(c); owns != (tk != 0) {
				return fmt.Errorf("cudaagreement: task %s on CTA %d has ticket %d, owner range is [%d,%d)", s.Task.ID, c, tk, s.Owner.Lo, s.Owner.Hi)
			}
			if tk == 0 {
				continue
			}
			if tk > total || seen[tk] {
				return fmt.Errorf("cudaagreement: task %s on CTA %d has ticket %d, out of range or reused", s.Task.ID, c, tk)
			}
			seen[tk] = true
		}
	}
	return nil
}

// artifact compiles the program's source, once per distinct source.
func (e *MegaExecutor) artifact(p *cudamega.Program) (*cudakernel.Artifact, error) {
	src, err := p.Source()
	if err != nil {
		return nil, fmt.Errorf("cudaagreement: render persistent worker: %w", err)
	}
	if art, ok := e.compiled[src]; ok {
		return art, nil
	}
	kind, image, err := e.Compile(src, e.Target)
	if err != nil {
		return nil, fmt.Errorf("cudaagreement: compile persistent worker: %w", err)
	}
	schema, err := workerSchema()
	if err != nil {
		return nil, err
	}
	art, err := cudakernel.NewArtifact(kind, image, "dag_worker", schema, string(e.Target))
	if err != nil {
		return nil, fmt.Errorf("cudaagreement: persistent worker artifact: %w", err)
	}
	if e.compiled == nil {
		e.compiled = make(map[string]*cudakernel.Artifact)
	}
	e.compiled[src] = art
	return art, nil
}

// workerSchema is the parameter list of the generated dag_worker entry.
func workerSchema() (cudakernel.Schema, error) {
	ptr := func(name string) cudakernel.Param {
		return cudakernel.Param{Name: name, Type: "ptr", IsPointer: true}
	}
	s, err := cudakernel.NewSchema(
		ptr("input"), ptr("scratch"), ptr("completed"), ptr("start_order"), ptr("abort_code"),
		cudakernel.Param{Name: "n", Type: "u32"},
		cudakernel.Param{Name: "epoch", Type: "u64"},
	)
	if err != nil {
		return cudakernel.Schema{}, fmt.Errorf("cudaagreement: persistent worker schema: %w", err)
	}
	return s, nil
}

// resident declines a grid wider than the residency the driver reports for
// the compiled worker.
func (e *MegaExecutor) resident(fn *cuda.Kernel) (string, error) {
	var perSM int32
	err := e.Device.WithContext(func() error {
		var err error
		perSM, err = fn.Raw().MaxActiveBlocksPerMultiprocessor(int32(e.Grid.ThreadsPerCTA), 0)
		return err
	})
	if err != nil {
		return "", fmt.Errorf("cudaagreement: persistent worker occupancy: %w", err)
	}
	sms, err := e.Device.MultiprocessorCount()
	if err != nil {
		return "", fmt.Errorf("cudaagreement: multiprocessor count: %w", err)
	}
	if resident := int64(perSM) * int64(sms); resident < int64(e.Grid.CTAs) {
		return fmt.Sprintf("grid of %d CTAs exceeds residency %d (%d per SM x %d SMs)", e.Grid.CTAs, resident, perSM, sms), nil
	}
	return "", nil
}

// buffer returns executor-owned scratch of at least words elements.
func (e *MegaExecutor) buffer(name string, words int) (cudamem.Slice[uint32], error) {
	if s, ok := e.scratch[name]; ok && s.Len() >= words {
		return s.Slice(0, words), nil
	}
	if s, ok := e.scratch[name]; ok {
		if err := s.Free(); err != nil {
			return cudamem.Slice[uint32]{}, fmt.Errorf("cudaagreement: free %s: %w", name, err)
		}
		delete(e.scratch, name)
	}
	s, err := cudamem.Alloc[uint32](e.Device, words)
	if err != nil {
		return cudamem.Slice[uint32]{}, fmt.Errorf("cudaagreement: allocate %s: %w", name, err)
	}
	if e.scratch == nil {
		e.scratch = make(map[string]cudamem.Slice[uint32])
	}
	e.scratch[name] = s
	return s, nil
}

func (e *MegaExecutor) orderBuffer(words int) error {
	if e.order.Len() >= words {
		return nil
	}
	if e.order.Len() > 0 {
		if err := e.order.Free(); err != nil {
			return fmt.Errorf("cudaagreement: free start order: %w", err)
		}
		e.order = cudamem.Slice[uint64]{}
	}
	s, err := cudamem.Alloc[uint64](e.Device, words)
	if err != nil {
		return fmt.Errorf("cudaagreement: allocate start order: %w", err)
	}
	e.order = s
	return nil
}

func copyAsync(dev *cuda.Device, stream *cuda.Stream, dst, src driver.DevicePtr, bytes int) error {
	return dev.WithContext(func() error {
		return stream.WithHandle(func(h driver.Stream) error {
			return dst.MemcpyAsync(src, uint(bytes), h)
		})
	})
}

// Close frees the scratch the executor allocated. The caller must drain the
// stream first.
func (e *MegaExecutor) Close() error {
	for name, s := range e.scratch {
		if err := s.Free(); err != nil {
			return fmt.Errorf("cudaagreement: free %s: %w", name, err)
		}
		delete(e.scratch, name)
	}
	if e.order.Len() > 0 {
		if err := e.order.Free(); err != nil {
			return fmt.Errorf("cudaagreement: free start order: %w", err)
		}
		e.order = cudamem.Slice[uint64]{}
	}
	return nil
}
