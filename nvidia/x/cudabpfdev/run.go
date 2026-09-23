package cudabpfdev

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"runtime"
	"slices"
	"sync"
	"time"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvjitlinkx"
)

// Snapshot is a device invocation's host view after successful kernel completion
// and readback. Inactive Values are zero. The fixed-capacity Map is private to
// this invocation; Misses counts out-of-range keys. This is snapshot-at-boundary
// consistency, not a coherent live view of device memory. No timing is implied.
type Snapshot struct {
	Values [32]uint32
	Map    []uint32
	Misses uint64
}

// Execution owns one invocation's module, stream and buffers. It must not be
// copied. Close every non-nil Execution returned by Execute, including on error,
// before closing its device. Failed cleanup retains resources for retry.
type Execution struct {
	mu        sync.Mutex
	ops       runOps
	link      *nvjitlinkx.Result
	module    driver.Module
	stream    driver.Stream
	ptrs      [3]driver.DevicePtr
	pending   bool
	ready     bool
	snapshot  Snapshot
	boundary  boundary
	limits    resourceLimits
	resources compiledResources
}

// Execute assembles a translation with nvJitLink and runs its one-warp oracle
// kernel on dev. It allocates and zeroes private map, result and miss buffers,
// launches on a private nonblocking stream, waits for that stream, then copies
// the result. It does not inject into another kernel. The driver checks native
// target compatibility. This runner limits maps to 1048576 slots.
// Before allocating invocation buffers, it queries compiled function resources
// and rejects register/shared-memory budget overruns, any local memory (including
// spills), or a thread limit below 32. Declaration counts are not substituted
// for these loaded-function attributes.
//
// Cancellation is checked between native calls; it cannot interrupt compilation
// or a running kernel. Cancellation never permits freeing in-flight storage.
// A snapshot is unavailable after any execution error, including readback or
// context restoration failure. A non-nil result must always be closed.
func Execute(ctx context.Context, dev *cuda.Device, t Translation, active uint32) (*Execution, error) {
	if dev == nil {
		return nil, fmt.Errorf("cudabpfdev: nil device")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if t.slots > 1<<20 {
		return nil, fmt.Errorf("cudabpfdev: device runner map exceeds 1048576 slots")
	}
	// Assemble validates the translation seal before any native link operation.
	linked, err := Assemble(t)
	e := &Execution{ops: nativeRunOps(dev), link: linked, limits: t.limits}
	if err != nil {
		return e, err
	}
	if linked == nil || len(linked.CUBIN) == 0 || len(linked.PTX) != 0 {
		return e, fmt.Errorf("cudabpfdev: linker did not return CUBIN")
	}
	a, err := cudakernel.NewArtifact(cudakernel.KindCUBIN, linked.CUBIN, t.Artifact.EntryPoint, t.Artifact.Schema, t.Artifact.Target)
	if err != nil {
		return e, fmt.Errorf("cudabpfdev: linked artifact: %w", err)
	}
	s, err := NewArtifactSource(a, Geometry{Grid: [3]uint32{1, 1, 1}, Block: [3]uint32{32, 1, 1}})
	if err != nil {
		return e, err
	}
	err = e.execute(ctx, s, t.slots, active)
	return e, err
}

// Snapshot returns a detached host view and whether execution completed
// successfully. It remains available after Close. An error or zero Execution
// has no snapshot; retrying cleanup does not turn failed execution into success.
func (e *Execution) Snapshot() (Snapshot, bool) {
	if e == nil {
		return Snapshot{}, false
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	s := e.snapshot
	s.Map = slices.Clone(s.Map)
	return s, e.ready
}

func (e *Execution) execute(ctx context.Context, source *ArtifactSource, slots, active uint32) error {
	var snapshot Snapshot
	err := e.ops.run(func() error {
		if err := ctx.Err(); err != nil {
			return err
		}
		var err error
		e.module, err = source.load(func(f func() error) error { return f() }, e.ops.load)
		if err != nil {
			return err
		}
		fn, err := e.ops.function(e.module, source.launch.Artifact.EntryPoint)
		if err != nil {
			return err
		}
		e.resources, err = e.ops.resources(fn)
		if err != nil {
			return err
		}
		if err := e.resources.check(e.limits); err != nil {
			return err
		}
		e.stream, err = e.ops.stream()
		if err != nil {
			return err
		}
		// A zero-slot map still gets a valid private pointer, never dereferenced.
		sizes := [3]uint{uint(max(slots, 1)) * 4, 128, 8}
		for i, size := range sizes {
			if err := ctx.Err(); err != nil {
				return err
			}
			e.ptrs[i], err = e.ops.alloc(size)
			if err != nil {
				return err
			}
			e.pending = true
			if err := e.ops.zero(e.ptrs[i], size, e.stream); err != nil {
				return err
			}
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		// Mark before launch: even a launch error must not assume no submission.
		e.pending = true
		if err := e.ops.launch(fn, e.stream, e.ptrs, active); err != nil {
			return err
		}
		if err := e.ops.synchronize(e.stream); err != nil {
			return err
		}
		e.pending = false
		if err := ctx.Err(); err != nil {
			return err
		}
		var data [3][]byte
		for i, size := range sizes {
			data[i] = make([]byte, size)
			if err := e.ops.read(e.ptrs[i], data[i]); err != nil {
				return err
			}
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		snapshot.Map = make([]uint32, slots)
		for i := range snapshot.Map {
			snapshot.Map[i] = binary.LittleEndian.Uint32(data[0][i*4:])
		}
		for i := range snapshot.Values {
			snapshot.Values[i] = binary.LittleEndian.Uint32(data[1][i*4:])
		}
		snapshot.Misses = binary.LittleEndian.Uint64(data[2])
		return nil
	})
	if err != nil {
		return fmt.Errorf("cudabpfdev: execute: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	e.snapshot, e.ready = snapshot, true
	e.boundary = boundary{timestamp: time.Now().UnixNano(), digest: source.launch.Artifact.Digest,
		entry: source.launch.Artifact.EntryPoint, target: source.launch.Artifact.Target, active: active}
	return nil
}

// Close waits for any possibly submitted work before releasing device memory.
// It is repeatable and safe concurrently with Snapshot and other Close calls.
// A failed wait releases nothing; successful frees are recorded immediately,
// even if restoring the previous context subsequently fails.
func (e *Execution) Close() error {
	if e == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	var linkErr error
	if e.link != nil {
		linkErr = e.link.Close()
		if linkErr == nil {
			e.link = nil
		}
	}
	if e.module == 0 && e.stream == 0 && e.ptrs == [3]driver.DevicePtr{} {
		return linkErr
	}
	err := e.ops.run(func() error {
		if e.pending {
			if err := e.ops.synchronize(e.stream); err != nil {
				return err
			}
			e.pending = false
		}
		for i, ptr := range e.ptrs {
			if ptr != 0 {
				if err := e.ops.free(ptr); err != nil {
					return err
				}
				e.ptrs[i] = 0
			}
		}
		if e.stream != 0 {
			if err := e.ops.destroy(e.stream); err != nil {
				return err
			}
			e.stream = 0
		}
		if e.module != 0 {
			if err := e.ops.unload(e.module); err != nil {
				return err
			}
			e.module = 0
		}
		return nil
	})
	if err = errors.Join(linkErr, err); err != nil {
		return fmt.Errorf("cudabpfdev: close execution: %w", err)
	}
	return nil
}

type runOps struct {
	run         func(func() error) error
	load        func([]byte, cudakernel.JITOptions) (driver.Module, error)
	function    func(driver.Module, string) (driver.Function, error)
	resources   func(driver.Function) (compiledResources, error)
	stream      func() (driver.Stream, error)
	alloc       func(uint) (driver.DevicePtr, error)
	zero        func(driver.DevicePtr, uint, driver.Stream) error
	launch      func(driver.Function, driver.Stream, [3]driver.DevicePtr, uint32) error
	synchronize func(driver.Stream) error
	read        func(driver.DevicePtr, []byte) error
	free        func(driver.DevicePtr) error
	destroy     func(driver.Stream) error
	unload      func(driver.Module) error
}

func nativeRunOps(dev *cuda.Device) runOps {
	return runOps{
		run: dev.WithContext, load: loadImage,
		function:  driver.Module.GetFunction,
		resources: func(fn driver.Function) (compiledResources, error) { return queryResources(fn.GetAttribute) },
		stream:    func() (driver.Stream, error) { return driver.NewStream(uint32(driver.StreamNonBlocking)) },
		alloc:     driver.MemAlloc,
		zero: func(ptr driver.DevicePtr, size uint, stream driver.Stream) error {
			return ptr.MemsetD8Async(0, size, stream)
		},
		launch: func(fn driver.Function, stream driver.Stream, ptrs [3]driver.DevicePtr, active uint32) error {
			args := []unsafe.Pointer{unsafe.Pointer(&ptrs[0]), unsafe.Pointer(&ptrs[1]), unsafe.Pointer(&ptrs[2]), unsafe.Pointer(&active)}
			var pins runtime.Pinner
			for _, arg := range args {
				pins.Pin(arg)
			}
			defer pins.Unpin()
			return fn.LaunchKernel(1, 1, 1, 32, 1, 1, 0, stream, args)
		},
		synchronize: driver.Stream.Synchronize,
		read: func(ptr driver.DevicePtr, dst []byte) error {
			var pins runtime.Pinner
			pins.Pin(&dst[0])
			defer pins.Unpin()
			return ptr.MemcpyDtoH(unsafe.Pointer(&dst[0]), uint(len(dst)))
		},
		free: driver.DevicePtr.MemFree, destroy: driver.Stream.Destroy, unload: driver.Module.Unload,
	}
}
