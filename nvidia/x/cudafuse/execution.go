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
	failure                     error
}

type executionBuffer struct {
	ptr   driver.DevicePtr
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

// Execute compiles and runs the supported row-major FP32 region on stream. It waits
// through Stream.Synchronize before exposing output. Unsupported graphs or
// invalid bindings are refused before compilation or allocation.
// The caller must keep stream and its device live until the execution is
// closed, and must not mutate borrowed input storage while work is pending.
// Runtime-owned outputs include two guard elements in their allocation;
// Readback refuses overwritten guards or the reserved unwritten poison value.
func (p *MaterializedPlan) Execute(ctx context.Context, stream *cuda.Stream, bindings map[string]ExecutionBinding) (*Execution, error) {
	return p.execute(ctx, stream, bindings, false)
}

// ExecuteEager executes one kernel and allocates one result for each primitive
// in the same supported graph. It provides a concrete baseline for Execute.
// It has the same stream, device and input-lifetime requirements as Execute.
func (p *MaterializedPlan) ExecuteEager(ctx context.Context, stream *cuda.Stream, bindings map[string]ExecutionBinding) (*Execution, error) {
	return p.execute(ctx, stream, bindings, true)
}

func (p *MaterializedPlan) execute(ctx context.Context, stream *cuda.Stream, bindings map[string]ExecutionBinding, eager bool) (*Execution, error) {
	nodes, n, err := executionNodes(p)
	if err != nil {
		return nil, err
	}
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
		if !ok || b.DType != cudair.Float32 || !slices.Equal(b.Shape, node.Shape) || b.Buffer.Len() != n {
			rollback()
			return nil, fmt.Errorf("cudafuse: invalid binding %s", node.Name())
		}
		if n == 0 {
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
		return &Execution{n: 0, complete: true}, nil
	}
	e, err := runExecution(ctx, nodes, n, inputs, eager, nativeExecutionOps(dev, stream), releases)
	return e, err
}

type executionBindingDescription struct {
	shape      []int
	dtype      cudair.DataType
	length     int
	ptr, bytes uint64
	device     *cuda.Device
}

func validateExecutionBindings(nodes []*LazyNode, n int, device *cuda.Device, bindings map[string]executionBindingDescription) error {
	count := 0
	var ranges [][2]uint64
	for _, node := range nodes {
		if node.Op != "input" {
			continue
		}
		count++
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
	if p == nil || p.RootNode == nil || len(p.Regions) != 1 {
		return nil, 0, errors.New("cudafuse: execution requires one elementwise region")
	}
	r := p.Regions[0]
	if r == nil || r.Compaction || IsMaterializingBoundary(r.Boundary) || len(r.Outputs) != 1 || r.Outputs[0] != p.RootNode {
		return nil, 0, errors.New("cudafuse: unsupported execution boundary")
	}
	if p.RootNode.Op == "input" {
		return nil, 0, errors.New("cudafuse: execution requires a compute output")
	}
	count, err := indexedElements(p.RootNode)
	if err != nil {
		return nil, 0, err
	}
	n := int(count)
	var nodes []*LazyNode
	state := map[*LazyNode]int{}
	names := map[string]bool{}
	var visit func(*LazyNode) error
	visit = func(node *LazyNode) error {
		if node == nil {
			return errors.New("cudafuse: nil input node")
		}
		if state[node] == 1 {
			return errors.New("cudafuse: cyclic execution graph")
		}
		if state[node] == 2 {
			return nil
		}
		if node.Err != nil {
			return node.Err
		}
		elements, err := indexedElements(node)
		if err != nil || node.DType != cudair.Float32 || elements != count {
			return errors.New("cudafuse: unsupported execution dtype or shape")
		}
		if node.Op == "input" {
			if len(node.Inputs) != 0 || len(node.Attrs) != 0 {
				return errors.New("cudafuse: unsupported execution input attributes or arity")
			}
		} else if err := validateIndexedOperator(node); err != nil {
			return err
		}
		if names[node.Name()] {
			return errors.New("cudafuse: duplicate node name")
		}
		names[node.Name()] = true
		state[node] = 1
		for _, in := range node.Inputs {
			if err := visit(in); err != nil {
				return err
			}
		}
		state[node] = 2
		nodes = append(nodes, node)
		return nil
	}
	if err := visit(p.RootNode); err != nil {
		return nil, 0, err
	}
	// The region must actually describe this graph, not an unrelated cached kernel.

	if len(r.Nodes) != len(nodes) {
		return nil, 0, errors.New("cudafuse: incomplete execution region")
	}
	seen := map[*LazyNode]bool{}
	for _, node := range r.Nodes {
		if state[node] != 2 || seen[node] {
			return nil, 0, errors.New("cudafuse: stale execution region")
		}
		seen[node] = true
	}
	seen = map[*LazyNode]bool{}
	for _, in := range r.Inputs {
		if in == nil || state[in] != 2 || in.Op != "input" || seen[in] {
			return nil, 0, errors.New("cudafuse: stale execution region inputs")
		}
		seen[in] = true
	}
	for _, node := range nodes {
		if node.Op == "input" && !seen[node] {
			return nil, 0, errors.New("cudafuse: incomplete execution region inputs")
		}
	}
	return nodes, n, nil
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
	e := &Execution{n: n, ops: ops, releases: releases}
	values := map[string]driver.DevicePtr{}
	for node, ptr := range inputs {
		values[node.Name()] = ptr
	}
	fail := func(err error) (*Execution, error) {
		e.failure = err
		if !e.attempted {
			return e, errors.Join(err, e.Close())
		}
		return e, err
	}
	root := nodes[len(nodes)-1]
	for _, node := range nodes {
		if node.Op == "input" || !eager && node != root {
			continue
		}
		if ctx != nil && ctx.Err() != nil {
			return fail(ctx.Err())
		}
		source, err := executionSource(nodes, node, fmt.Sprintf("lazy_execution_%d", node.ID), eager)
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
		e.receipts = append(e.receipts, kernel.receipt)
		if ctx != nil && ctx.Err() != nil {
			return fail(ctx.Err())
		}
		buffer, err := ops.alloc(n)
		if buffer != nil {
			e.resources = append(e.resources, buffer)
			e.stats.Allocations++
		}
		if err != nil {
			return fail(fmt.Errorf("cudafuse: allocate: %w", err))
		}
		if ctx != nil && ctx.Err() != nil {
			return fail(ctx.Err())
		}
		pointers := make([]driver.DevicePtr, 0, len(source.Inputs)+1)
		for _, name := range source.Inputs {
			ptr := values[name]
			if ptr == 0 {
				return fail(errors.New("cudafuse: missing native operand"))
			}
			pointers = append(pointers, ptr)
		}
		pointers = append(pointers, buffer.ptr)
		e.attempted = true
		if err := kernel.launch(pointers, n); err != nil {
			return fail(fmt.Errorf("cudafuse: launch: %w", err))
		}
		e.stats.Launches++
		values[node.Name()] = buffer.ptr
		e.output = buffer
	}
	if !e.attempted {
		return fail(errors.New("cudafuse: no compute operation"))
	}
	if err := e.Wait(ctx); err != nil {
		return e, err
	}
	return e, nil
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
