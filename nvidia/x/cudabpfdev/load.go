package cudabpfdev

import (
	"errors"
	"fmt"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvjitlinkx"
)

// Selection identifies the image selected after loading and resource checks.
// Candidate reports selection of instrumentation, not execution or equivalence.
// Reason explains why the original was selected; it is empty for a candidate.
type Selection struct {
	Kernel    KernelIdentity
	Candidate bool
	Reason    string
}

// LoadedInjection owns private original/candidate modules and linker resources.
// It must not be copied. Close every non-nil result of Injection.Load, including
// error results, before closing its device. Each Load creates separate modules;
// no instrumentation map is shared with another LoadedInjection.
type LoadedInjection struct {
	mu                              sync.Mutex
	ops                             injectionLoadOps
	modules                         [2]driver.Module
	link                            *nvjitlinkx.Result
	function                        driver.Function
	globals                         [2]driver.DevicePtr
	resources                       [2]compiledResources
	geometry                        Geometry
	schema                          cudakernel.Schema
	slots                           uint32
	ready                           bool
	selection                       Selection
	originalKernel                  KernelIdentity
	stream                          driver.Stream
	pending, used, closed, complete bool
	snapshot                        MapSnapshot
	timestamp                       int64
}

// Load first loads the original, then links the candidate through nvJitLink and
// checks its loaded resources and map symbols. A candidate failure selects the
// original with a reason. An original-load failure or context-restoration failure
// returns an error and no usable selection. No kernel is launched by Load.
//
// Compiled total registers/shared bytes must fit the preparation budget; the
// added register cost must fit the declared instrumentation allowance. Candidate
// local memory is rejected. The supplied block size must fit the loaded function.
// Map sizes and alignment are checked before candidate selection; initialization
// and completion-boundary readback belong to the invocation path.
func (i *Injection) Load(dev *cuda.Device) (*LoadedInjection, error) {
	if dev == nil {
		return nil, fmt.Errorf("cudabpfdev: nil device")
	}
	return i.load(nativeInjectionLoadOps(dev))
}

func (i *Injection) load(ops injectionLoadOps) (*LoadedInjection, error) {
	if i == nil || i.original == nil {
		return nil, fmt.Errorf("cudabpfdev: missing original source")
	}
	l := &LoadedInjection{ops: ops, geometry: i.original.launch.Geometry, schema: cloneArtifact(i.original.launch.Artifact).Schema, slots: i.slots}
	a := i.original.launch.Artifact
	l.selection = Selection{Kernel: KernelIdentity{a.Digest, a.EntryPoint, a.Target}, Reason: i.reason}
	l.originalKernel = l.selection.Kernel
	err := ops.run(func() error {
		var err error
		l.modules[0], err = i.original.load(func(f func() error) error { return f() }, ops.load)
		if err != nil {
			return err
		}
		l.function, err = ops.function(l.modules[0], a.EntryPoint)
		if err != nil {
			return fmt.Errorf("cudabpfdev: original function: %w", err)
		}
		if i.candidate == nil {
			return nil
		}
		fallback := func(err error) error { l.selection.Reason = err.Error(); return nil }
		if i.slots > 1<<20 {
			return fallback(fmt.Errorf("cudabpfdev: instrumentation map exceeds runner capacity"))
		}
		l.resources[0], err = ops.resources(l.function)
		if err != nil {
			return fallback(err)
		}
		base := l.resources[0]
		if base.registers < 0 || base.sharedBytes < 0 || base.localBytes < 0 || base.maxThreads < 0 {
			return fallback(fmt.Errorf("cudabpfdev: invalid original resource attributes"))
		}
		var candidate *cudakernel.Artifact
		candidate, l.link, err = ops.compile(i.candidate.launch.Artifact)
		if err != nil {
			return fallback(fmt.Errorf("cudabpfdev: candidate link: %w", err))
		}
		if candidate == nil {
			return fallback(fmt.Errorf("cudabpfdev: missing linked candidate"))
		}
		source, err := NewArtifactSource(candidate, l.geometry)
		if err != nil {
			return fallback(err)
		}
		l.modules[1], err = source.load(func(f func() error) error { return f() }, ops.load)
		if err != nil {
			return fallback(fmt.Errorf("cudabpfdev: candidate load: %w", err))
		}
		fn, err := ops.function(l.modules[1], a.EntryPoint)
		if err != nil {
			return fallback(err)
		}
		l.resources[1], err = ops.resources(fn)
		if err != nil {
			return fallback(err)
		}
		res := l.resources[1]
		if err := res.check(i.limits); err != nil {
			return fallback(err)
		}
		if int64(res.registers)-int64(base.registers) > int64(i.addedRegisters) {
			return fallback(fmt.Errorf("cudabpfdev: compiled register increase exceeds instrumentation allowance"))
		}
		threads := uint64(1)
		for _, d := range l.geometry.Block {
			if uint64(d) > uint64(res.maxThreads)/threads {
				return fallback(fmt.Errorf("cudabpfdev: block size exceeds candidate thread limit"))
			}
			threads *= uint64(d)
		}
		if uint64(res.sharedBytes)+uint64(l.geometry.SharedBytes) > uint64(i.limits.sharedBytes) {
			return fallback(fmt.Errorf("cudabpfdev: total shared memory exceeds injection budget"))
		}
		for n, global := range []struct {
			name  string
			size  uint
			align driver.DevicePtr
		}{
			{"__cudabpfdev_map", uint(max(i.slots, 1)) * 4, 4},
			{"__cudabpfdev_misses", 8, 8},
		} {
			ptr, size, err := ops.global(l.modules[1], global.name)
			if err != nil {
				return fallback(err)
			}
			if ptr == 0 || ptr%global.align != 0 || size != global.size {
				return fallback(fmt.Errorf("cudabpfdev: invalid instrumentation global %s", global.name))
			}
			l.globals[n] = ptr
		}
		mapStart, missStart := uint64(l.globals[0]), uint64(l.globals[1])
		mapEnd, missEnd := mapStart+uint64(max(i.slots, 1))*4, missStart+8
		if mapEnd < mapStart || missEnd < missStart || mapStart < missEnd && missStart < mapEnd {
			return fallback(fmt.Errorf("cudabpfdev: overlapping or overflowing instrumentation globals"))
		}
		// Function selection changes only after all candidate checks succeed.
		l.function = fn
		l.selection = Selection{Kernel: KernelIdentity{candidate.Digest, candidate.EntryPoint, candidate.Target}, Candidate: true}
		return nil
	})
	if err != nil {
		return l, fmt.Errorf("cudabpfdev: load injection: %w", err)
	}
	l.ready = true
	return l, nil
}

// Selection returns the image decision after a successful Load. It remains
// available after Close as a receipt; it does not authorize further use then.
func (l *LoadedInjection) Selection() (Selection, error) {
	if l == nil {
		return Selection{}, fmt.Errorf("cudabpfdev: no loaded selection")
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if !l.ready {
		return Selection{}, fmt.Errorf("cudabpfdev: no loaded selection")
	}
	return l.selection, nil
}

// Close waits for possibly submitted work, then destroys the stream and unloads
// retained modules. It retries failed cleanup. Successes are recorded before
// context restoration so retry cannot release the same handle twice. A failed
// wait releases nothing. After any Close call, further launches are forbidden.
func (l *LoadedInjection) Close() error {
	if l == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.closed = true
	var linkErr error
	if l.link != nil {
		linkErr = l.link.Close()
		if linkErr == nil {
			l.link = nil
		}
	}
	if l.modules == [2]driver.Module{} && l.stream == 0 {
		return linkErr
	}
	err := l.ops.run(func() error {
		if l.pending {
			if err := l.ops.synchronize(l.stream); err != nil {
				return err
			}
			l.pending = false
		}
		if l.stream != 0 {
			if err := l.ops.destroy(l.stream); err != nil {
				return err
			}
			l.stream = 0
		}
		for n := len(l.modules) - 1; n >= 0; n-- {
			if l.modules[n] != 0 {
				if err := l.ops.unload(l.modules[n]); err != nil {
					return err
				}
				l.modules[n] = 0
			}
		}
		return nil
	})
	if err = errors.Join(linkErr, err); err != nil {
		return fmt.Errorf("cudabpfdev: close injection: %w", err)
	}
	return nil
}

type injectionLoadOps struct {
	run         func(func() error) error
	load        func([]byte, cudakernel.JITOptions) (driver.Module, error)
	function    func(driver.Module, string) (driver.Function, error)
	resources   func(driver.Function) (compiledResources, error)
	compile     func(*cudakernel.Artifact) (*cudakernel.Artifact, *nvjitlinkx.Result, error)
	global      func(driver.Module, string) (driver.DevicePtr, uint, error)
	unload      func(driver.Module) error
	stream      func() (driver.Stream, error)
	zero        func(driver.DevicePtr, uint, driver.Stream) error
	launch      func(driver.Function, Geometry, driver.Stream, cudakernel.Schema, []byte) error
	synchronize func(driver.Stream) error
	read        func(driver.DevicePtr, []byte) error
	destroy     func(driver.Stream) error
}

func nativeInjectionLoadOps(dev *cuda.Device) injectionLoadOps {
	native := nativeRunOps(dev)
	return injectionLoadOps{
		run: dev.WithContext, load: loadImage, function: driver.Module.GetFunction,
		resources: func(fn driver.Function) (compiledResources, error) { return queryResources(fn.GetAttribute) },
		compile: func(a *cudakernel.Artifact) (*cudakernel.Artifact, *nvjitlinkx.Result, error) {
			return compileInjection(a, nvjitlinkx.Link)
		},
		global: driver.Module.GetGlobal, unload: driver.Module.Unload,
		stream: native.stream, zero: native.zero, launch: launchInjection,
		synchronize: native.synchronize, read: native.read, destroy: native.destroy,
	}
}

func compileInjection(a *cudakernel.Artifact, link func([]nvjitlinkx.Input, ...string) (*nvjitlinkx.Result, error)) (*cudakernel.Artifact, *nvjitlinkx.Result, error) {
	in, err := nvjitlinkx.InputFromKernelArtifact("injected.ptx", a)
	if err != nil {
		return nil, nil, err
	}
	r, err := link([]nvjitlinkx.Input{in}, "-arch="+a.Target)
	if err != nil {
		return nil, r, err
	}
	if r == nil || len(r.CUBIN) == 0 || len(r.PTX) != 0 {
		return nil, r, fmt.Errorf("cudabpfdev: linker did not return CUBIN")
	}
	artifact, err := cudakernel.NewArtifact(cudakernel.KindCUBIN, r.CUBIN, a.EntryPoint, a.Schema, a.Target)
	return artifact, r, err
}
