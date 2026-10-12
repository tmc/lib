package cudafuse

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
	"sync"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// ExecutionBinding binds a named input to contiguous FP32 storage. Shape and
// DType describe the supplied view, not a conversion to be performed.
type ExecutionBinding struct {
	Buffer cudamem.Slice[float32]
	Shape  []int
	DType  cudair.DataType
}

// ExecutionStats counts successful native allocation, free and compute launch
// calls made by this execution. Input allocations belong to the caller.
type ExecutionStats struct{ Allocations, Frees, Launches int }

// KernelReceipt identifies the actual source compiled and PTX loaded.
type KernelReceipt struct{ Name, SourceDigest, PTXDigest, Target, CompilerKey string }

// ErrOutputGuard means execution overwrote an output allocation's guard.
var ErrOutputGuard = errors.New("cudafuse: output guard overwritten")

// ErrOutputUnwritten means a logical result retains the reserved fill value.
var ErrOutputUnwritten = errors.New("cudafuse: output was not written")

// Execution owns output, temporaries and loaded modules. The caller must Close
// it even when Execute returns both an execution and an error. In particular,
// cancellation does not free in-flight storage. Close waits for actual stream
// completion; a failed wait leaves ownership intact for retry.
// The zero value is not usable for Wait or Readback; Close is harmless.
type Execution struct {
	mu                          sync.Mutex
	ops                         executionOps
	resources                   []*executionBuffer
	kernels                     []*executionKernel
	pending                     []io.Closer
	releases                    []func()
	output                      *executionBuffer
	stats                       ExecutionStats
	receipts                    []KernelReceipt
	attempted, complete, closed bool
	n                           int
	shape                       []int
	failure                     error
}

type executionBuffer struct {
	ptr   driver.DevicePtr
	check func() error
	read  func([]float32) error
	free  func() error
	freed func() bool
}
type executionKernel struct {
	receipt KernelReceipt
	launch  func([]driver.DevicePtr, int) error
	close   func() error
}
type executionOps struct {
	alloc   func(int) (*executionBuffer, error)
	compile func(FusedKernel) (*executionKernel, error)
	sync    func(context.Context) error
}

// Execute compiles and runs row-major FP32 pointwise regions and rank-one
// axis-zero sums and inclusive rank-one sum scans on stream. Sum accumulates FP32 values sequentially in FP64
// and converts once to FP32. Empty sums write +0 to output shape [1]. Scan
// seeds with the first FP32 value, preserving its signed zero, and visits later
// positions in order using FP32 nearest-even addition. Empty scans return [0]
// without device work. The reserved unwritten NaN payload remains unsupported.
// It waits
// through Stream.Synchronize before exposing output. Unsupported graphs or
// invalid bindings are refused before compilation or allocation.
// The caller must not change the plan, graph, or binding metadata during this
// call. Execution snapshots graph metadata and owns each materialized result;
// it borrows caller input allocations until Close successfully drains cleanup.
// The caller must keep stream and its device live until the execution is
// closed, and must not mutate borrowed input storage while work is pending.
// Runtime-owned outputs include two guard elements in their allocation;
// Readback refuses overwritten guards or the reserved unwritten poison value.
func (p *MaterializedPlan) Execute(ctx context.Context, stream *cuda.Stream, bindings map[string]ExecutionBinding) (*Execution, error) {
	return p.execute(ctx, stream, bindings, false)
}

// ExecuteEager executes one kernel and allocates one result for each nonempty primitive
// in the same supported graph. It provides a concrete baseline for Execute.
// It has the same stream, device and input-lifetime requirements as Execute.
func (p *MaterializedPlan) ExecuteEager(ctx context.Context, stream *cuda.Stream, bindings map[string]ExecutionBinding) (*Execution, error) {
	return p.execute(ctx, stream, bindings, true)
}

func (p *MaterializedPlan) execute(ctx context.Context, stream *cuda.Stream, bindings map[string]ExecutionBinding, eager bool) (*Execution, error) {
	plan, err := snapshotExecution(p, eager)
	if err != nil {
		return nil, err
	}
	nodes := plan.nodes
	n := int(mustExecutionElements(plan.root))
	if n != 0 && (stream == nil || stream.Device() == nil) {
		return nil, errors.New("cudafuse: nil execution stream or device")
	}
	var dev *cuda.Device
	if stream != nil {
		dev = stream.Device()
	}
	descriptions := map[string]executionBindingDescription{}
	for name, b := range bindings {
		descriptions[name] = executionBindingDescription{shape: b.Shape, dtype: b.DType, length: b.Buffer.Len(), ptr: uint64(b.Buffer.Ptr()), bytes: uint64(b.Buffer.SizeBytes()), device: b.Buffer.Device()}
	}
	if err := validateExecutionBindings(nodes, n, dev, descriptions); err != nil {
		return nil, err
	}
	inputs := make(map[*LazyNode]driver.DevicePtr)
	var releases []func()
	rollback := func() {
		for _, release := range releases {
			release()
		}
	}
	var ranges [][2]uint64
	count := 0
	for _, node := range nodes {
		if node.Op != "input" {
			continue
		}
		count++
		b, ok := bindings[node.Name()]
		if !ok || b.DType != cudair.Float32 || !slices.Equal(b.Shape, node.Shape) || b.Buffer.Len() != int(mustExecutionElements(node)) {
			rollback()
			return nil, fmt.Errorf("cudafuse: invalid binding %s", node.Name())
		}
		if mustExecutionElements(node) == 0 {
			continue
		}
		release, e := b.Buffer.Borrow(dev)
		if e != nil {
			rollback()
			return nil, fmt.Errorf("cudafuse: binding %s: %w", node.Name(), e)
		}
		releases = append(releases, release)
		ptr := uint64(b.Buffer.Ptr())
		size := uint64(b.Buffer.SizeBytes())
		if ptr == 0 || size > math.MaxUint64-ptr {
			rollback()
			return nil, errors.New("cudafuse: invalid input extent")
		}
		for _, r := range ranges {
			if ptr < r[1] && r[0] < ptr+size {
				rollback()
				return nil, errors.New("cudafuse: aliased input bindings")
			}
		}
		ranges = append(ranges, [2]uint64{ptr, ptr + size})
		inputs[node] = driver.DevicePtr(ptr)
	}
	if len(bindings) != count {
		rollback()
		return nil, errors.New("cudafuse: unexpected binding")
	}
	if ctx != nil && ctx.Err() != nil {
		rollback()
		return nil, ctx.Err()
	}
	if n == 0 {
		return &Execution{n: 0, shape: slices.Clone(plan.root.Shape), complete: true}, nil
	}
	e, err := runExecutionPlan(ctx, plan, inputs, nativeExecutionOps(dev, stream), releases)
	return e, err
}

type executionBindingDescription struct {
	shape      []int
	dtype      cudair.DataType
	length     int
	ptr, bytes uint64
	device     *cuda.Device
}

func validateExecutionBindings(nodes []*LazyNode, _ int, device *cuda.Device, bindings map[string]executionBindingDescription) error {
	count := 0
	var ranges [][2]uint64
	for _, node := range nodes {
		if node.Op != "input" {
			continue
		}
		count++
		elements, err := indexedElements(node)
		if err != nil {
			return err
		}
		n := int(elements)
		b, ok := bindings[node.Name()]
		if !ok || b.dtype != cudair.Float32 || !slices.Equal(b.shape, node.Shape) || b.length != n {
			return fmt.Errorf("cudafuse: invalid binding %s", node.Name())
		}
		if n == 0 {
			if b.ptr != 0 && b.device != device {
				return errors.New("cudafuse: empty binding device differs")
			}
			continue
		}
		if b.device != device || b.ptr == 0 || b.bytes != uint64(n)*4 || b.bytes > math.MaxUint64-b.ptr {
			return errors.New("cudafuse: binding ownership or extent differs")
		}
		for _, r := range ranges {
			if b.ptr < r[1] && r[0] < b.ptr+b.bytes {
				return errors.New("cudafuse: aliased input bindings")
			}
		}
		ranges = append(ranges, [2]uint64{b.ptr, b.ptr + b.bytes})
	}
	if len(bindings) != count {
		return errors.New("cudafuse: unexpected binding")
	}
	return nil
}

func executionNodes(p *MaterializedPlan) ([]*LazyNode, int, error) {
	if _, err := snapshotExecution(p, false); err != nil {
		return nil, 0, err
	}
	nodes, err := checkedTopologicalOrder(p.RootNode)
	if err != nil {
		return nil, 0, err
	}
	count, _ := indexedElements(p.RootNode)
	return nodes, int(count), nil
}

func mustExecutionElements(n *LazyNode) uint64 {
	count, _ := indexedElements(n) // The execution snapshot already validated it.
	return count
}

func executionSource(nodes []*LazyNode, root *LazyNode, name string, eager bool) (FusedKernel, error) {
	indexed := false
	for _, node := range nodes {
		indexed = indexed || node.Op == "transpose" || node.Op == "reshape"
	}
	if indexed {
		var inputs []*LazyNode
		if eager {
			for _, input := range root.Inputs {
				if !slices.Contains(inputs, input) {
					inputs = append(inputs, input)
				}
			}
		} else {
			for _, node := range nodes {
				if node.Op == "input" {
					inputs = append(inputs, node)
				}
			}
		}
		return compileIndexedLazyRegion(inputs, []*LazyNode{root}, name)
	}
	g := cudair.NewGraph()
	values := map[*LazyNode]*cudair.Node{}
	addInput := func(node *LazyNode) *cudair.Node {
		if v := values[node]; v != nil {
			return v
		}
		count, _ := indexedElements(node) // executionNodes already checked the domain.
		v := g.AddInput(node.Name(), []int{int(count)}, node.DType)
		values[node] = v
		return v
	}
	for _, node := range nodes {
		if node.Op == "input" {
			if !eager {
				addInput(node)
			}
			continue
		}
		if eager && node != root {
			continue
		}
		in := make([]*cudair.Node, len(node.Inputs))
		for i, n := range node.Inputs {
			if eager {
				in[i] = addInput(n)
			} else {
				in[i] = values[n]
			}
			if in[i] == nil {
				return FusedKernel{}, errors.New("cudafuse: missing emitted operand")
			}
		}
		if len(in) == 1 {
			values[node] = g.AddUnary(node.Op, in[0])
		} else {
			values[node] = g.AddBinary(node.Op, in[0], in[1])
		}
	}
	return compileRegionKernel(g, name, values[root])
}

func runExecution(ctx context.Context, nodes []*LazyNode, n int, inputs map[*LazyNode]driver.DevicePtr, eager bool, ops executionOps, releases []func()) (*Execution, error) {
	if len(nodes) == 0 {
		return nil, errors.New("cudafuse: empty execution graph")
	}
	regions, err := partitionIntoRegions(nodes)
	if err != nil {
		return nil, err
	}
	plan, err := snapshotExecution(&MaterializedPlan{Regions: regions, RootNode: nodes[len(nodes)-1]}, eager)
	if err != nil {
		return nil, err
	}
	if int(mustExecutionElements(plan.root)) != n {
		return nil, errors.New("cudafuse: execution root extent differs")
	}
	bound := make(map[*LazyNode]driver.DevicePtr)
	for original, ptr := range inputs {
		bound[plan.originals[original]] = ptr
	}
	return runExecutionPlan(ctx, plan, bound, ops, releases)
}

func runExecutionPlan(ctx context.Context, plan *executionPlan, inputs map[*LazyNode]driver.DevicePtr, ops executionOps, releases []func()) (*Execution, error) {
	e := &Execution{n: int(mustExecutionElements(plan.root)), shape: slices.Clone(plan.root.Shape), ops: ops, releases: releases}
	values := make(map[*LazyNode]driver.DevicePtr)
	for node, ptr := range inputs {
		values[node] = ptr
	}
	fail := func(err error) (*Execution, error) {
		e.failure = err
		if !e.attempted {
			return e, errors.Join(err, e.Close())
		}
		return e, err
	}
	for i, step := range plan.steps {
		if step.count == 0 {
			continue
		}
		if ctx != nil && ctx.Err() != nil {
			return fail(ctx.Err())
		}
		source, err := step.source(fmt.Sprintf("lazy_execution_%d", i))
		if err != nil {
			return fail(err)
		}
		kernel, err := ops.compile(source)
		if kernel != nil {
			e.kernels = append(e.kernels, kernel)
		}
		if err != nil {
			var pending io.Closer
			if errors.As(err, &pending) {
				e.pending = append(e.pending, pending)
			}
			return fail(fmt.Errorf("cudafuse: compile/load: %w", err))
		}
		if kernel == nil || kernel.launch == nil {
			return fail(errors.New("cudafuse: missing execution kernel"))
		}
		e.receipts = append(e.receipts, kernel.receipt)
		pointers := make([]driver.DevicePtr, 0, len(step.inputs)+len(step.outputs))
		byName := make(map[string]*LazyNode, len(step.inputs))
		for _, in := range step.inputs {
			byName[in.Name()] = in
		}
		for _, name := range source.Inputs {
			in := byName[name]
			if in == nil {
				return fail(errors.New("cudafuse: unknown source operand"))
			}
			ptr := values[in]
			if ptr == 0 && mustExecutionElements(in) != 0 {
				return fail(errors.New("cudafuse: missing native operand"))
			}
			pointers = append(pointers, ptr)
		}
		for _, out := range step.outputs {
			if ctx != nil && ctx.Err() != nil {
				return fail(ctx.Err())
			}
			buffer, err := ops.alloc(int(mustExecutionElements(out)))
			if buffer != nil {
				e.resources = append(e.resources, buffer)
				e.stats.Allocations++
			}
			if err != nil {
				return fail(fmt.Errorf("cudafuse: allocate: %w", err))
			}
			if buffer == nil || buffer.ptr == 0 {
				return fail(errors.New("cudafuse: missing execution buffer"))
			}
			pointers = append(pointers, buffer.ptr)
			values[out] = buffer.ptr
			if out == plan.root {
				e.output = buffer
			}
		}
		if ctx != nil && ctx.Err() != nil {
			return fail(ctx.Err())
		}
		// Mark every attempt before the native call: failed submission may still
		// have touched the stream. Later stages cannot inherit a completed epoch.
		e.attempted, e.complete = true, false
		if err := kernel.launch(pointers, step.count); err != nil {
			return fail(fmt.Errorf("cudafuse: launch: %w", err))
		}
		e.stats.Launches++
	}
	if !e.attempted {
		if e.n != 0 {
			return fail(errors.New("cudafuse: no compute operation"))
		}
		e.complete = true
		return e, nil
	}
	if err := e.Wait(ctx); err != nil {
		return e, err
	}
	return e, nil
}

// Shape returns a copy of the execution's validated output shape. Sum has
// shape [1], including an empty-input sum; rank-zero pointwise results have [].
func (e *Execution) Shape() []int {
	if e == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return slices.Clone(e.shape)
}

// Wait establishes actual device completion. A canceled or failed wait leaves
// buffers and modules owned by e for a subsequent Wait or Close.
func (e *Execution) Wait(ctx context.Context) error {
	if e == nil {
		return errors.New("cudafuse: nil execution")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return errors.New("cudafuse: closed execution")
	}
	if e.complete {
		return nil
	}
	if e.ops.sync == nil {
		return errors.New("cudafuse: uninitialized execution")
	}
	if err := e.ops.sync(ctx); err != nil {
		return fmt.Errorf("cudafuse: synchronize: %w", err)
	}
	e.complete = true
	// A failed submission has no admissible output. Preserve its original
	// error instead of replacing it with a poison check on unwritten storage.
	if e.failure != nil {
		return nil
	}
	for _, b := range e.resources {
		if b != nil && b.check != nil {
			if err := b.check(); err != nil {
				e.failure = fmt.Errorf("cudafuse: check completed output: %w", err)
				return e.failure
			}
		}
	}
	return nil
}

// Readback waits for completion and copies the owned output to host storage.
// Empty executions return an empty slice without any device calls.
func (e *Execution) Readback(ctx context.Context) ([]float32, error) {
	if err := e.Wait(ctx); err != nil {
		return nil, err
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, errors.New("cudafuse: closed execution")
	}
	if e.failure != nil {
		return nil, e.failure
	}
	out := make([]float32, e.n)
	if e.n == 0 {
		return out, nil
	}
	if err := e.output.read(out); err != nil {
		return nil, fmt.Errorf("cudafuse: read output: %w", err)
	}
	return out, nil
}

// Stats returns counts of successful execution-owned native calls.
func (e *Execution) Stats() ExecutionStats {
	if e == nil {
		return ExecutionStats{}
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.stats
}

// Receipts returns a copy of the actual source/load identities.
func (e *Execution) Receipts() []KernelReceipt {
	if e == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return slices.Clone(e.receipts)
}

// Close waits without cancellation before releasing resources. Failed cleanup
// retains the failed owner for retry; inputs remain caller-owned throughout.
func (e *Execution) Close() error {
	if e == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	if e.attempted && !e.complete {
		if err := e.ops.sync(context.Background()); err != nil {
			return fmt.Errorf("cudafuse: drain before close: %w", err)
		}
		e.complete = true
	}
	var errs []error
	for i, b := range e.resources {
		if b != nil {
			err := b.free()
			// cudamem clears its owner only after the native free succeeds, even
			// when restoring the previous context subsequently reports an error.
			if err == nil || b.freed != nil && b.freed() {
				e.resources[i] = nil
				e.stats.Frees++
			}
			if err != nil {
				errs = append(errs, err)
			}
		}
	}
	for i, k := range e.kernels {
		if k != nil {
			if err := k.close(); err != nil {
				errs = append(errs, err)
			} else {
				e.kernels[i] = nil
			}
		}
	}
	for i, owner := range e.pending {
		if owner != nil {
			if err := owner.Close(); err != nil {
				errs = append(errs, err)
			} else {
				e.pending[i] = nil
			}
		}
	}
	if len(errs) != 0 {
		return errors.Join(errs...)
	}
	for _, release := range e.releases {
		release()
	}
	e.releases = nil
	e.closed = true
	return nil
}

func nativeExecutionOps(dev *cuda.Device, stream *cuda.Stream) executionOps {
	return executionOps{
		sync: stream.Synchronize,
		alloc: func(n int) (*executionBuffer, error) {
			storage, err := cudamem.Alloc[float32](dev, n+2)
			if storage.Ptr() == 0 {
				return nil, err
			}
			view := storage.Slice(1, n+1)
			b := &executionBuffer{ptr: view.Ptr(), free: storage.Free, freed: func() bool { return storage.Ptr() == 0 }}
			b.check = func() error {
				all := make([]float32, n+2)
				if err := storage.CopyToHost(all); err != nil {
					return err
				}
				return checkExecutionOutput(all, n)
			}
			b.read = func(out []float32) error {
				all := make([]float32, n+2)
				if err := storage.CopyToHost(all); err != nil {
					return err
				}
				if err := checkExecutionOutput(all, n); err != nil {
					return err
				}
				copy(out, all[1:n+1])
				return nil
			}
			if err != nil {
				return b, err
			}
			poison := make([]float32, n+2)
			for i := range poison {
				poison[i] = math.Float32frombits(0x7fc01234)
			}
			return b, storage.CopyFromHost(poison)
		},
		compile: func(f FusedKernel) (*executionKernel, error) {
			major, minor, err := dev.ComputeCapability()
			if err != nil {
				return nil, err
			}
			cm, err := nvrtcx.Compile(f.Source, nvrtcx.WithArch(major, minor), nvrtcx.WithFlag("--fmad=false"))
			if err != nil {
				return nil, err
			}
			m, _, err := cm.LoadOwnedOutput(dev, nvrtcx.OutputPTX)
			if m == nil {
				return nil, err
			}
			k := &executionKernel{close: m.Close, receipt: KernelReceipt{Name: f.Name, SourceDigest: fmt.Sprintf("%x", sha256.Sum256([]byte(f.Source))), PTXDigest: fmt.Sprintf("%x", sha256.Sum256([]byte(cm.PTX()))), Target: fmt.Sprintf("compute_%d%d", major, minor), CompilerKey: cm.Key()}}
			if err != nil {
				return k, err
			}
			fn, err := m.Function(f.Name)
			if err != nil {
				return k, err
			}
			k.launch = func(ptrs []driver.DevicePtr, n int) error {
				args := make([]unsafe.Pointer, len(ptrs)+1)
				for i := range ptrs {
					args[i] = unsafe.Pointer(&ptrs[i])
				}
				length := int32(n)
				args[len(ptrs)] = unsafe.Pointer(&length)
				return stream.WithHandle(func(s driver.Stream) error {
					return fn.Launch([3]uint32{uint32((n + 255) / 256), 1, 1}, [3]uint32{256, 1, 1}, 0, s, args)
				})
			}
			return k, nil
		},
	}
}

func checkExecutionOutput(all []float32, n int) error {
	if len(all) != n+2 || math.Float32bits(all[0]) != 0x7fc01234 || math.Float32bits(all[n+1]) != 0x7fc01234 {
		return ErrOutputGuard
	}
	for _, v := range all[1 : n+1] {
		if math.Float32bits(v) == 0x7fc01234 {
			return ErrOutputUnwritten
		}
	}
	return nil
}
