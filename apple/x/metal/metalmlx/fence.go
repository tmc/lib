package metalmlx

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/tmc/apple/metal"
)

// fenceSource is mlx/backend/metal/kernels/fence.metal. MLX ships these
// kernels in its metallib; the port compiles them on demand instead.
const fenceSource = `
#pragma METAL internals : enable

#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
#endif
namespace metal {
constexpr constant metal::thread_scope thread_scope_system =
    static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__);
}

#include <metal_atomic>

[[kernel]] void input_coherent(
    volatile coherent(system) device uint* input [[buffer(0)]],
    const constant uint& size [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  if (index < size) {
    input[index] = input[index];
  }
  metal::atomic_thread_fence(
      metal::mem_flags::mem_device,
      metal::memory_order_seq_cst,
      metal::thread_scope_system);
}

[[kernel]] void fence_update(
    volatile coherent(system) device uint* timestamp [[buffer(0)]],
    constant uint& value [[buffer(1)]]) {
  timestamp[0] = value;
  metal::atomic_thread_fence(
      metal::mem_flags::mem_device,
      metal::memory_order_seq_cst,
      metal::thread_scope_system);
}

[[kernel]] void fence_wait(
    volatile coherent(system) device uint* timestamp [[buffer(0)]],
    constant uint& value [[buffer(1)]]) {
  while (1) {
    metal::atomic_thread_fence(
        metal::mem_flags::mem_device,
        metal::memory_order_seq_cst,
        metal::thread_scope_system);
    if (timestamp[0] >= value) {
      break;
    }
  }
}
`

// Fence orders work between streams, and between the CPU and the GPU, by a
// counter that only goes up. Port of FenceImpl in
// mlx/backend/metal/fence.cpp.
//
// The fast path, selected by MLX_METAL_FAST_SYNCH, keeps the counter in
// shared memory and moves it with single-thread kernels, so a waiter does not
// have to end its command buffer. Otherwise the counter is an [Event].
type Fence struct {
	mu    sync.Mutex
	dev   *Device
	fast  bool
	count uint32

	// Fast path.
	buf metal.MTLBuffer
	val *uint32
	lib metal.MTLLibrary

	// Event path.
	event *Event
}

// NewFence makes a fence with count zero.
func NewFence(d *Device) (*Fence, error) {
	f := &Fence{dev: d}
	f.fast = d.raw.SupportsFamily(metal.MTLGPUFamilyMetal3) && envInt("MLX_METAL_FAST_SYNCH", 0) != 0
	if !f.fast {
		event, err := NewEvent(d)
		if err != nil {
			return nil, err
		}
		f.event = event
		return f, nil
	}
	buf, err := d.alloc.Malloc(4)
	if err != nil {
		return nil, err
	}
	lib, err := d.LibraryFromSource("mlx_fence", CompileOptions{},
		func() (string, error) { return fenceSource, nil })
	if err != nil {
		d.alloc.Free(buf)
		return nil, err
	}
	f.buf, f.lib = buf, lib
	f.val = (*uint32)(buf.Contents())
	atomic.StoreUint32(f.val, 0)
	return f, nil
}

// Count returns the fence's current value.
func (f *Fence) Count() (uint32, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.closedLocked() {
		return 0, fmt.Errorf("metalmlx: fence is closed")
	}
	return f.count, nil
}

// Wait makes the work encoded after it wait for the fence, and registers out
// as written so nothing that depends on it starts early.
func (f *Fence) Wait(e *CommandEncoder, out metal.MTLBuffer) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.closedLocked() {
		return fmt.Errorf("metalmlx: fence is closed")
	}
	if !f.fast {
		return e.WaitEvent(f.event, uint64(f.count))
	}
	kernel, err := f.dev.Kernel(f.lib, "fence_wait")
	if err != nil {
		return err
	}
	e.RegisterOutput(out)
	e.SetComputePipelineState(kernel)
	e.SetBuffer(f.buf, 0, 0)
	count := f.count
	e.SetBytes(mustValueBytes(&count), 1)
	e.DispatchThreads(Size{1, 1, 1}, Size{1, 1, 1})
	return nil
}

// Update advances the fence once the work encoded so far is done. With
// crossDevice, in is first made visible to other devices.
func (f *Fence) Update(e *CommandEncoder, in metal.MTLBuffer, crossDevice bool) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.closedLocked() {
		return fmt.Errorf("metalmlx: fence is closed")
	}
	f.count++
	if !f.fast {
		return e.SignalEvent(f.event, uint64(f.count))
	}
	if crossDevice {
		kernel, err := f.dev.Kernel(f.lib, "input_coherent")
		if err != nil {
			return err
		}
		nthreads := uint32((in.Length() + 3) / 4)
		e.SetComputePipelineState(kernel)
		e.SetInputBuffer(in, 0, 0)
		e.SetBytes(mustValueBytes(&nthreads), 1)
		e.DispatchThreadgroups(Size{Width: uint((nthreads + 1023) / 1024)}, Size{Width: 1024})
	}
	// Order the update after the kernels it fences.
	e.Barrier()

	kernel, err := f.dev.Kernel(f.lib, "fence_update")
	if err != nil {
		return err
	}
	count := f.count
	e.SetComputePipelineState(kernel)
	e.SetBuffer(f.buf, 0, 0)
	e.SetBytes(mustValueBytes(&count), 1)
	e.DispatchThreads(Size{1, 1, 1}, Size{1, 1, 1})
	return nil
}

// UpdateCPU advances the fence from the CPU.
func (f *Fence) UpdateCPU() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.closedLocked() {
		return fmt.Errorf("metalmlx: fence is closed")
	}
	f.count++
	if !f.fast {
		return f.event.Signal(uint64(f.count))
	}
	atomic.StoreUint32(f.val, f.count)
	return nil
}

// WaitCPU blocks the calling goroutine until the fence reaches its count.
func (f *Fence) WaitCPU() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.closedLocked() {
		return fmt.Errorf("metalmlx: fence is closed")
	}
	if !f.fast {
		return f.event.Wait(uint64(f.count))
	}
	for atomic.LoadUint32(f.val) < f.count {
		runtime.Gosched()
	}
	return nil
}

// Destroy releases the fence's resources. It is idempotent and excludes
// concurrent method calls. The caller must synchronize every command buffer
// that refers to the fence before destroying it.
func (f *Fence) Destroy() {
	if f == nil {
		return
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.closedLocked() {
		return
	}
	if f.fast {
		f.dev.alloc.Free(f.buf)
		f.buf = nil
		f.val = nil
		return
	}
	f.event.Destroy()
	f.event = nil
}

func (f *Fence) closedLocked() bool {
	if f.fast {
		return !valid(f.buf)
	}
	return f.event == nil
}
