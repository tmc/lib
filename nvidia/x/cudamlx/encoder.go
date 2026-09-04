package cudamlx

import (
	"fmt"
	"os"
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
)

// Dim3 is a kernel grid or block shape.
type Dim3 struct {
	X, Y, Z uint32
}

func (d Dim3) orOne() Dim3 {
	if d.X == 0 {
		d.X = 1
	}
	if d.Y == 0 {
		d.Y = 1
	}
	if d.Z == 0 {
		d.Z = 1
	}
	return d
}

// KernelArgs collects kernel parameters as the pointers-to-values array
// that cuGraphAddKernelNode and cuLaunchKernelEx require. Port of
// KernelArgs in jit_module.h. The zero value is ready to use.
type KernelArgs struct {
	ptrs []unsafe.Pointer
}

// Args returns the parameter pointer array.
func (a *KernelArgs) Args() []unsafe.Pointer {
	return a.ptrs
}

// AppendBuffer appends the buffer's device pointer.
func (a *KernelArgs) AppendBuffer(b *Buffer) {
	a.AppendDevicePtr(driver.DevicePtr(b.Data))
}

// AppendDevicePtr appends a raw device pointer.
func (a *KernelArgs) AppendDevicePtr(p driver.DevicePtr) {
	appendValue(a, p)
}

// AppendInt32 appends an int32 parameter.
func (a *KernelArgs) AppendInt32(v int32) { appendValue(a, v) }

// AppendUint32 appends a uint32 parameter.
func (a *KernelArgs) AppendUint32(v uint32) { appendValue(a, v) }

// AppendInt64 appends an int64 parameter.
func (a *KernelArgs) AppendInt64(v int64) { appendValue(a, v) }

// AppendFloat32 appends a float32 parameter.
func (a *KernelArgs) AppendFloat32(v float32) { appendValue(a, v) }

// AppendBool appends a bool parameter.
func (a *KernelArgs) AppendBool(v bool) { appendValue(a, v) }

// AppendShape appends a shape padded to MaxNDim, matching kernels that
// take a Shape (cuda::std::array<int32_t, MAX_NDIM>) by value. Port of
// append_ndim.
func (a *KernelArgs) AppendShape(shape []int32) error {
	if len(shape) > MaxNDim {
		return fmt.Errorf("ndim can not be larger than %d", MaxNDim)
	}
	var s Shape
	copy(s[:], shape)
	appendValue(a, s)
	return nil
}

// AppendStrides appends strides padded to MaxNDim, matching kernels that
// take a Strides (cuda::std::array<int64_t, MAX_NDIM>) by value.
func (a *KernelArgs) AppendStrides(strides []int64) error {
	if len(strides) > MaxNDim {
		return fmt.Errorf("ndim can not be larger than %d", MaxNDim)
	}
	var s Strides
	copy(s[:], strides)
	appendValue(a, s)
	return nil
}

func appendValue[T any](a *KernelArgs, v T) {
	p := new(T)
	*p = v
	a.ptrs = append(a.ptrs, unsafe.Pointer(p))
}

// CommandEncoder batches kernel launches into a CUDA graph, tracking
// buffer dependencies so independent kernels run concurrently. Port of
// cu::CommandEncoder in device.cpp, without the graph-exec cache: each
// Commit instantiates and launches a fresh executable graph.
type CommandEncoder struct {
	dev    *Device
	stream driver.Stream
	worker *Worker

	useGraphs bool
	graph     driver.Graph

	nodeCount       int
	bytesInGraph    uint64
	maxOpsPerGraph  int
	maxMBPerGraph   int
	inConcurrent    bool
	fromNodes       []driver.GraphNode
	toNodes         []driver.GraphNode
	concurrentNodes []driver.GraphNode
	activeDeps      []uintptr
	activeOutputs   []uintptr
	nodeMap         map[uintptr]driver.GraphNode
	retained        []any
}

// NewCommandEncoder creates an encoder with its own non-blocking stream
// and worker. Set CUDAMLX_USE_CUDA_GRAPHS=0 to launch kernels directly
// on the stream instead of batching them into graphs.
func NewCommandEncoder(dev *Device) (*CommandEncoder, error) {
	e := &CommandEncoder{
		dev:       dev,
		useGraphs: os.Getenv("CUDAMLX_USE_CUDA_GRAPHS") != "0",
		nodeMap:   make(map[uintptr]driver.GraphNode),
	}
	// Port of get_graph_limits, tunable per compute capability in the
	// C++; keep the conservative default.
	e.maxOpsPerGraph = 20
	e.maxMBPerGraph = 100
	err := dev.WithCurrent(func() error {
		s, err := driver.NewStream(uint32(driver.StreamNonBlocking))
		if err != nil {
			return err
		}
		e.stream = s
		if e.useGraphs {
			e.graph, err = driver.NewGraph(0)
			if err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	e.worker = NewWorker(dev)
	return e, nil
}

// Device returns the encoder's device.
func (e *CommandEncoder) Device() *Device { return e.dev }

// Stream returns the encoder's stream.
func (e *CommandEncoder) Stream() driver.Stream { return e.stream }

// SetInputBuffer records buf as an input of the next kernel node.
func (e *CommandEncoder) SetInputBuffer(buf *Buffer) {
	if !e.useGraphs {
		return
	}
	e.bytesInGraph += buf.Size
	e.activeDeps = append(e.activeDeps, uintptr(unsafe.Pointer(buf)))
}

// SetOutputBuffer records buf as an output of the next kernel node.
func (e *CommandEncoder) SetOutputBuffer(buf *Buffer) {
	if !e.useGraphs {
		return
	}
	id := uintptr(unsafe.Pointer(buf))
	e.activeDeps = append(e.activeDeps, id)
	e.activeOutputs = append(e.activeOutputs, id)
}

// Retain keeps v alive until the current batch's completion handlers
// run. Port of add_temporary.
func (e *CommandEncoder) Retain(v any) {
	e.retained = append(e.retained, v)
}

// AddCompletedHandler queues task to run after the current batch's GPU
// work completes.
func (e *CommandEncoder) AddCompletedHandler(task func()) {
	e.worker.AddTask(task)
}

// Concurrent runs fn with dependency tracking relaxed: kernel nodes
// added inside fn depend on the nodes before the region, run
// concurrently with each other, and are joined by an empty node. Port of
// ConcurrentContext.
func (e *CommandEncoder) Concurrent(fn func() error) error {
	e.inConcurrent = true
	err := fn()
	e.inConcurrent = false
	if err != nil || !e.useGraphs {
		return err
	}

	empty, gerr := e.graph.AddEmptyNode(nil, 0)
	if gerr != nil {
		return fmt.Errorf("empty node: %w", gerr)
	}
	e.nodeCount++
	for _, from := range e.concurrentNodes {
		e.fromNodes = append(e.fromNodes, from)
		e.toNodes = append(e.toNodes, empty)
	}
	outputs := e.activeOutputs
	e.activeOutputs = nil
	e.insertGraphDependencies(e.concurrentNodes)
	e.concurrentNodes = nil
	for _, o := range outputs {
		e.nodeMap[o] = empty
	}
	return nil
}

// AddKernelNode adds one kernel launch with the parameters collected in
// args. Without graphs it launches directly on the stream.
func (e *CommandEncoder) AddKernelNode(fn driver.Function, grid, block Dim3, smemBytes uint32, args *KernelArgs) error {
	grid, block = grid.orOne(), block.orOne()
	params := args.Args()
	defer runtime.KeepAlive(params)

	if !e.useGraphs {
		e.nodeCount++
		return e.dev.WithCurrent(func() error {
			cfg := driver.LaunchConfig{
				GridDimX: grid.X, GridDimY: grid.Y, GridDimZ: grid.Z,
				BlockDimX: block.X, BlockDimY: block.Y, BlockDimZ: block.Z,
				SharedMemBytes: smemBytes,
				HStream:        e.stream,
			}
			return fn.LaunchKernelEx(&cfg, params, nil)
		})
	}

	kp := driver.KernelNodeParams{
		Func:     fn,
		GridDimX: grid.X, GridDimY: grid.Y, GridDimZ: grid.Z,
		BlockDimX: block.X, BlockDimY: block.Y, BlockDimZ: block.Z,
		SharedMemBytes: smemBytes,
	}
	if len(params) > 0 {
		kp.KernelParams = uintptr(unsafe.Pointer(&params[0]))
	}
	node, err := e.graph.AddKernelNode(nil, 0, &kp)
	if err != nil {
		return fmt.Errorf("add kernel node: %w", err)
	}
	e.nodeCount++
	if e.inConcurrent {
		e.concurrentNodes = append(e.concurrentNodes, node)
	} else {
		e.insertGraphDependencies([]driver.GraphNode{node})
	}
	return nil
}

func (e *CommandEncoder) insertGraphDependencies(nodes []driver.GraphNode) {
	var deps []driver.GraphNode
	seen := make(map[driver.GraphNode]bool)
	for _, d := range e.activeDeps {
		if n, ok := e.nodeMap[d]; ok && !seen[n] {
			seen[n] = true
			deps = append(deps, n)
		}
	}
	e.activeDeps = nil
	for _, o := range e.activeOutputs {
		for _, n := range nodes {
			e.nodeMap[o] = n
		}
	}
	e.activeOutputs = nil
	for _, from := range deps {
		for _, to := range nodes {
			e.fromNodes = append(e.fromNodes, from)
			e.toNodes = append(e.toNodes, to)
		}
	}
}

// NeedsCommit reports whether the graph has grown past the per-graph
// limits.
func (e *CommandEncoder) NeedsCommit() bool {
	return e.nodeCount > e.maxOpsPerGraph || e.bytesInGraph>>20 > uint64(e.maxMBPerGraph)
}

// Commit launches the accumulated graph on the stream and hands
// completion handlers to the worker.
func (e *CommandEncoder) Commit() error {
	if len(e.retained) > 0 {
		retained := e.retained
		e.retained = nil
		e.worker.AddTask(func() { _ = retained })
	}
	if e.useGraphs && e.nodeCount > 0 {
		err := e.dev.WithCurrent(func() error {
			if len(e.fromNodes) > 0 {
				if err := e.graph.AddDependencies(&e.fromNodes[0], &e.toNodes[0], nil, uint(len(e.fromNodes))); err != nil {
					return fmt.Errorf("add dependencies: %w", err)
				}
			}
			exec, err := e.graph.Instantiate(0)
			if err != nil {
				return fmt.Errorf("instantiate graph: %w", err)
			}
			defer exec.Destroy()
			if err := exec.GraphLaunch(e.stream); err != nil {
				return fmt.Errorf("launch graph: %w", err)
			}
			if err := e.graph.Destroy(); err != nil {
				return err
			}
			e.graph, err = driver.NewGraph(0)
			return err
		})
		e.clearGraphState()
		if err != nil {
			return err
		}
	}
	if err := e.worker.Commit(e.stream); err != nil {
		return err
	}
	e.nodeCount = 0
	e.bytesInGraph = 0
	return nil
}

// Synchronize commits pending work and blocks until kernels and
// completion handlers have finished.
func (e *CommandEncoder) Synchronize() error {
	if err := e.dev.WithCurrent(func() error { return e.stream.Synchronize() }); err != nil {
		return err
	}
	done := make(chan struct{})
	e.AddCompletedHandler(func() { close(done) })
	if err := e.Commit(); err != nil {
		return err
	}
	<-done
	return nil
}

// Close synchronizes and stops the worker.
func (e *CommandEncoder) Close() error {
	err := e.Synchronize()
	e.worker.Stop()
	return err
}

func (e *CommandEncoder) clearGraphState() {
	e.fromNodes = nil
	e.toNodes = nil
	e.concurrentNodes = nil
	e.activeDeps = nil
	e.activeOutputs = nil
	e.nodeMap = make(map[uintptr]driver.GraphNode)
}
