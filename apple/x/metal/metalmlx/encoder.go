package metalmlx

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/tmc/apple/metal"
	"github.com/tmc/apple/objc"
)

// Size is a Metal grid or threadgroup size. A zero field is treated as one.
type Size struct{ Width, Height, Depth uint }

func (s Size) mtl() metal.MTLSize {
	return metal.MTLSize{Width: orOne(s.Width), Height: orOne(s.Height), Depth: orOne(s.Depth)}
}

func orOne(v uint) uint {
	if v == 0 {
		return 1
	}
	return v
}

// CommandEncoder encodes work for one stream. It owns a command queue and
// the command buffer being built, tracks which buffers each dispatch reads
// and writes, and inserts the barriers and fences that order them. Port of
// metal::CommandEncoder in mlx/backend/metal/device.cpp.
//
// A CommandEncoder is not safe for concurrent use, which matches MLX: one
// encoder belongs to one stream.
type CommandEncoder struct {
	dev    *Device
	queue  metal.MTLCommandQueue
	buffer metal.MTLCommandBuffer

	bufferOps   int
	bufferSizes uint64

	// The residency sets and how many of them this queue has attached.
	setsAttached uint64

	// Events hooked to the current command buffer.
	waitEvents   []*Event
	signalEvents []signalEvent

	// Error from a previously committed command buffer.
	errMu sync.Mutex
	err   error

	// Set within a single compute encoder and reset by EndEncoding.
	encoder      metal.MTLComputeCommandEncoder
	fence        metal.MTLFence
	needsBarrier bool
	concurrent   bool

	prevInputs        map[objc.ID]metal.MTLResource
	prevOutputs       map[objc.ID]metal.MTLResource
	nextInputs        map[objc.ID]metal.MTLResource
	nextOutputs       map[objc.ID]metal.MTLResource
	concurrentOutputs map[objc.ID]metal.MTLResource
	allInputs         map[objc.ID]struct{}
	allOutputs        map[objc.ID]struct{}

	// Outputs of prior compute encoders and the fence that signals them.
	outputsMu     sync.Mutex
	prevCEOutputs map[objc.ID]metal.MTLFence
}

type signalEvent struct {
	event *Event
	value uint64
}

// NewCommandEncoder makes an encoder with its own command queue.
func NewCommandEncoder(d *Device) (*CommandEncoder, error) {
	if d == nil {
		return nil, fmt.Errorf("metalmlx: nil device")
	}
	d.mu.Lock()
	if d.closed {
		d.mu.Unlock()
		return nil, fmt.Errorf("metalmlx: device is closed")
	}
	d.encoders++
	d.mu.Unlock()
	e := &CommandEncoder{
		dev:               d,
		prevInputs:        make(map[objc.ID]metal.MTLResource),
		prevOutputs:       make(map[objc.ID]metal.MTLResource),
		nextInputs:        make(map[objc.ID]metal.MTLResource),
		nextOutputs:       make(map[objc.ID]metal.MTLResource),
		concurrentOutputs: make(map[objc.ID]metal.MTLResource),
		allInputs:         make(map[objc.ID]struct{}),
		allOutputs:        make(map[objc.ID]struct{}),
		prevCEOutputs:     make(map[objc.ID]metal.MTLFence),
	}
	var err error
	WithPool(func() {
		e.queue = d.raw.NewCommandQueue()
		if !valid(e.queue) {
			err = fmt.Errorf("metalmlx: failed to make new command queue")
			return
		}
		retain(e.queue)
		// Sets made later are attached in Commit.
		d.residency.attachNewSets(e.queue, &e.setsAttached)
		e.buffer = newCommandBuffer(e.queue)
	})
	if err != nil {
		d.mu.Lock()
		d.encoders--
		d.mu.Unlock()
		return nil, err
	}
	return e, nil
}

// newCommandBuffer makes a command buffer that does not retain the resources
// encoded into it, and retains the buffer itself so it outlives the pool.
func newCommandBuffer(q metal.MTLCommandQueue) metal.MTLCommandBuffer {
	buf := q.CommandBufferWithUnretainedReferences()
	retain(buf)
	return buf
}

// Device returns the encoder's device.
func (e *CommandEncoder) Device() *Device { return e.dev }

// CommandBuffer returns the command buffer being encoded into.
func (e *CommandEncoder) CommandBuffer() metal.MTLCommandBuffer { return e.buffer }

// SetComputePipelineState sets the kernel for the following dispatches.
func (e *CommandEncoder) SetComputePipelineState(kernel metal.MTLComputePipelineState) {
	e.computeEncoder().SetComputePipelineState(kernel)
}

// SetBuffer binds buf at idx and records it as both an input and an output,
// which orders it against everything else in flight.
func (e *CommandEncoder) SetBuffer(buf metal.MTLBuffer, idx, offset int) {
	e.allInputs[buf.GetID()] = struct{}{}
	e.allOutputs[buf.GetID()] = struct{}{}
	e.computeEncoder().SetBufferWithOffsetAtIndex(buf, uint(offset), uint(idx))
}

// SetInputBuffer binds buf at idx as an input of the next dispatch.
func (e *CommandEncoder) SetInputBuffer(buf metal.MTLBuffer, idx, offset int) {
	id := buf.GetID()
	if _, ok := e.allInputs[id]; !ok {
		e.allInputs[id] = struct{}{}
		e.bufferSizes += uint64(buf.Length())
	}
	e.nextInputs[id] = buf
	if _, ok := e.prevOutputs[id]; ok {
		e.needsBarrier = true
	}
	e.computeEncoder().SetBufferWithOffsetAtIndex(buf, uint(offset), uint(idx))
}

// SetOutputBuffer binds buf at idx as an output of the next dispatch.
func (e *CommandEncoder) SetOutputBuffer(buf metal.MTLBuffer, idx, offset int) {
	// Barriers are decided before the buffer joins the output set.
	e.SetInputBuffer(buf, idx, offset)
	e.RegisterOutput(buf)
}

// RegisterOutput records buf as written by the next dispatch without binding
// it, so later dispatches wait for it.
func (e *CommandEncoder) RegisterOutput(buf metal.MTLBuffer) {
	id := buf.GetID()
	e.allOutputs[id] = struct{}{}
	if e.concurrent {
		e.concurrentOutputs[id] = buf
		return
	}
	e.nextOutputs[id] = buf
	if _, ok := e.prevInputs[id]; ok {
		e.needsBarrier = true
	}
}

// SetBytes binds a small block of data at idx, copied into the command
// buffer. Use [ValueBytes] or [SliceBytes] to make the block.
func (e *CommandEncoder) SetBytes(data []byte, idx int) {
	e.computeEncoder().SetBytesLengthAtIndex(data, uint(idx))
}

// SetThreadgroupMemoryLength reserves threadgroup memory at idx.
func (e *CommandEncoder) SetThreadgroupMemoryLength(length, idx int) {
	e.computeEncoder().SetThreadgroupMemoryLengthAtIndex(uint(length), uint(idx))
}

// ValueBytes returns the bytes of *v, for [CommandEncoder.SetBytes].
//
// It returns an error for int, uint and uintptr: those are 64 bits wide on Apple silicon
// while MSL's int and uint are 32, so binding one writes twice the bytes the
// kernel reads and shifts every argument after it. Convert to a fixed-width
// type at the call site instead.
func ValueBytes[T any](v *T) ([]byte, error) {
	if v == nil {
		return nil, fmt.Errorf("metalmlx: nil kernel argument")
	}
	if err := rejectWordSized(v); err != nil {
		return nil, err
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(v)), unsafe.Sizeof(*v)), nil
}

// SliceBytes returns the bytes of s, for [CommandEncoder.SetBytes]. Like
// [ValueBytes] it returns an error for an int, uint, or uintptr element.
func SliceBytes[T any](s []T) ([]byte, error) {
	if len(s) == 0 {
		return nil, nil
	}
	if err := rejectWordSized(&s[0]); err != nil {
		return nil, err
	}
	var t T
	return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), uintptr(len(s))*unsafe.Sizeof(t)), nil
}

// rejectWordSized rejects a Go word-sized integer, whose width
// does not match the MSL type of the same name. The type switch on a concrete
// pointer type costs nothing, which matters: these run once per kernel
// argument per dispatch.
func rejectWordSized(p any) error {
	switch p.(type) {
	case *int, *uint, *uintptr:
		return fmt.Errorf("metalmlx: cannot bind %T as a kernel argument: Go's int and uint are 64 bits and MSL's are 32; use a fixed-width type", p)
	}
	return nil
}

func mustValueBytes[T any](v *T) []byte {
	b, err := ValueBytes(v)
	if err != nil {
		panic(err)
	}
	return b
}

// Concurrent runs fn with dispatches allowed to overlap: outputs registered
// inside fn are not ordered against each other, only against work encoded
// after fn returns. Port of CommandEncoder::ConcurrentContext.
func (e *CommandEncoder) Concurrent(fn func()) {
	e.concurrent = true
	fn()
	e.concurrent = false
	for id, buf := range e.concurrentOutputs {
		e.prevOutputs[id] = buf
	}
	clear(e.concurrentOutputs)
}

// maybeInsertBarrier inserts a barrier when the next dispatch reads or writes
// a buffer the previous ones touched, and rolls the input and output sets.
func (e *CommandEncoder) maybeInsertBarrier() {
	if e.needsBarrier {
		e.computeEncoder().MemoryBarrierWithScope(metal.MTLBarrierScopeBuffers)
		e.needsBarrier = false
		e.prevInputs, e.nextInputs = e.nextInputs, e.prevInputs
		e.prevOutputs, e.nextOutputs = e.nextOutputs, e.prevOutputs
	} else {
		for id, buf := range e.nextInputs {
			e.prevInputs[id] = buf
		}
		for id, buf := range e.nextOutputs {
			e.prevOutputs[id] = buf
		}
	}
	clear(e.nextInputs)
	clear(e.nextOutputs)
}

// DispatchThreadgroups dispatches grid threadgroups of group threads.
func (e *CommandEncoder) DispatchThreadgroups(grid, group Size) {
	e.maybeInsertBarrier()
	e.bufferOps++
	e.computeEncoder().DispatchThreadgroupsThreadsPerThreadgroup(grid.mtl(), group.mtl())
}

// DispatchThreads dispatches a grid of threads, in threadgroups of group.
func (e *CommandEncoder) DispatchThreads(grid, group Size) {
	e.maybeInsertBarrier()
	e.bufferOps++
	e.computeEncoder().DispatchThreadsThreadsPerThreadgroup(grid.mtl(), group.mtl())
}

// Barrier orders every buffer access before it against those after it.
func (e *CommandEncoder) Barrier() {
	e.computeEncoder().MemoryBarrierWithScope(metal.MTLBarrierScopeBuffers)
}

// EndEncoding closes the current compute encoder.
//
// Each compute encoder has its own fence, and the encoder keeps a map of the
// outputs of earlier compute encoders to the fence that signals them. An
// encoder waits on the fences of its inputs, records its outputs in the map,
// and always signals its own fence. A completion handler then drops its
// outputs from the map, to bound the map and to avoid needless waits.
func (e *CommandEncoder) EndEncoding() {
	if e.encoder == nil {
		return
	}
	fence := e.fence
	waitingOn := make(map[objc.ID]metal.MTLFence)
	e.outputsMu.Lock()
	for in := range e.allInputs {
		f, ok := e.prevCEOutputs[in]
		if !ok {
			continue
		}
		// One wait per fence is enough.
		if _, waited := waitingOn[f.GetID()]; waited {
			continue
		}
		e.encoder.WaitForFence(f)
		retain(f) // held until this encoder's work completes
		waitingOn[f.GetID()] = f
	}
	for out := range e.allOutputs {
		e.prevCEOutputs[out] = fence
	}
	e.outputsMu.Unlock()

	e.encoder.UpdateFence(fence)
	allOutputs := e.allOutputs
	e.buffer.AddCompletedHandler(func(metal.MTLCommandBuffer) {
		e.outputsMu.Lock()
		for o := range allOutputs {
			if f, ok := e.prevCEOutputs[o]; ok && f.GetID() == fence.GetID() {
				delete(e.prevCEOutputs, o)
			}
		}
		e.outputsMu.Unlock()
		release(fence)
		for _, f := range waitingOn {
			release(f)
		}
	})

	e.encoder.EndEncoding()
	release(e.encoder)
	e.encoder = nil
	e.fence = nil
	e.needsBarrier = false
	e.concurrent = false
	clear(e.prevInputs)
	clear(e.nextInputs)
	clear(e.prevOutputs)
	clear(e.nextOutputs)
	clear(e.concurrentOutputs)
	e.allInputs = make(map[objc.ID]struct{})
	e.allOutputs = make(map[objc.ID]struct{})
}

// SignalEvent signals event with value once the work encoded so far is done.
func (e *CommandEncoder) SignalEvent(event *Event, value uint64) error {
	ev, err := event.MTLEvent()
	if err != nil {
		return err
	}
	e.EndEncoding()
	e.buffer.EncodeSignalEventValue(ev, value)
	e.signalEvents = append(e.signalEvents, signalEvent{event, value})
	return nil
}

// WaitEvent makes the following work wait for event to reach value.
func (e *CommandEncoder) WaitEvent(event *Event, value uint64) error {
	ev, err := event.MTLEvent()
	if err != nil {
		return err
	}
	e.EndEncoding()
	e.buffer.EncodeWaitForEventValue(ev, value)
	e.waitEvents = append(e.waitEvents, event)
	return nil
}

// NeedsCommit reports whether the command buffer has grown past the device's
// dispatch count or size threshold.
func (e *CommandEncoder) NeedsCommit() bool {
	maxOps, maxMB := e.dev.MaxOpsMBPerBuffer()
	return e.bufferOps > maxOps || int(e.bufferSizes>>20) > maxMB
}

// Commit submits the command buffer and starts a new one. completion, when
// not nil, runs on a Metal thread once the GPU is done.
func (e *CommandEncoder) Commit(completion func()) {
	// Metal locks a command buffer's residency at commit time, so attach any
	// sets made since the last commit first.
	e.dev.residency.attachNewSets(e.queue, &e.setsAttached)

	signals := e.signalEvents
	e.signalEvents = nil
	e.waitEvents = nil
	e.buffer.AddCompletedHandler(func(cbuf metal.MTLCommandBuffer) {
		if completion != nil {
			completion()
		}
		if cbuf.Status() != metal.MTLCommandBufferStatusError {
			return
		}
		e.setError(fmt.Errorf("metalmlx: command buffer execution failed: %s", commandBufferError(cbuf)))
		// Metal does not signal the events on error, so signal them here to
		// avoid waiting forever.
		for _, s := range signals {
			_ = s.event.Signal(s.value)
		}
	})

	old := e.buffer
	WithPool(func() {
		e.buffer.Commit()
		e.buffer = newCommandBuffer(e.queue)
	})
	release(old)
	e.bufferOps = 0
	e.bufferSizes = 0
}

// Synchronize commits the work encoded so far and waits for the GPU. It
// returns the error of any command buffer that has failed.
func (e *CommandEncoder) Synchronize() error {
	cbuf := e.buffer
	retain(cbuf) // Commit replaces and releases e.buffer
	e.EndEncoding()
	e.Commit(nil)
	cbuf.WaitUntilCompleted()
	release(cbuf)
	return e.Error()
}

// Error returns the error of a failed command buffer, if any.
func (e *CommandEncoder) Error() error {
	e.errMu.Lock()
	defer e.errMu.Unlock()
	return e.err
}

func (e *CommandEncoder) setError(err error) {
	e.errMu.Lock()
	defer e.errMu.Unlock()
	if e.err == nil { // keep the earliest error
		e.err = err
	}
}

// Close waits for outstanding work and releases the queue and its buffer. It
// is idempotent and returns any command-buffer error observed while closing.
func (e *CommandEncoder) Close() error {
	if e == nil || e.queue == nil {
		return nil
	}
	err := e.Synchronize()
	release(e.buffer)
	release(e.queue)
	e.buffer = nil
	e.queue = nil
	e.dev.mu.Lock()
	e.dev.encoders--
	e.dev.mu.Unlock()
	return err
}

// computeEncoder returns the open compute encoder, opening one, with a fresh
// fence, when there is none.
func (e *CommandEncoder) computeEncoder() metal.MTLComputeCommandEncoder {
	if e.encoder == nil {
		WithPool(func() {
			e.encoder = e.buffer.ComputeCommandEncoderWithDispatchType(metal.MTLDispatchTypeConcurrent)
			retain(e.encoder)
			e.fence = e.dev.raw.NewFence()
		})
	}
	return e.encoder
}

func commandBufferError(cbuf metal.MTLCommandBuffer) string {
	if err := cbuf.Error(); err.GetID() != 0 {
		return err.LocalizedDescription()
	}
	return "unknown error"
}
