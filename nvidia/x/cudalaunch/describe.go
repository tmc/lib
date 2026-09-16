package cudalaunch

import (
	"errors"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

// Description is a supported, read-only view of a configured launch. It carries
// what another package needs to re-express the launch — as a graph node, for
// example — without reading a Launcher's unexported fields.
//
// A Description holds the owning Kernel, which keeps the module from being
// collected for as long as the Description is reachable. That is not a lease:
// [cuda.Module.Close] unloads the module whatever else still refers to it, and
// the cached driver.Function then names a function that is no longer loaded.
// A caller must not Close a module while a Description of one of its kernels is
// still in use, including by a graph or an executable graph built from it.
//
// The zero Description is not usable; obtain one from Launcher.Describe.
type Description struct {
	// Function is the kernel entry point. It is never zero in a Description
	// returned by Describe.
	Function driver.Function

	// Grid and Block are the execution configuration in blocks and threads.
	Grid  Dim3
	Block Dim3

	// SharedMem is dynamic shared memory per block, in bytes.
	SharedMem uint32

	// Attrs are the launch attributes configured on the launcher, in order.
	// A consumer that cannot express an attribute must report an error rather
	// than drop it.
	Attrs []driver.LaunchAttribute

	// kernel retains module ownership. It is nil when the launch targets a
	// bare driver function whose module the caller owns.
	kernel *cuda.Kernel
}

// Kernel returns the owned kernel the description came from, or nil if the
// launch targets a bare driver function. Retaining the Description keeps the
// kernel's module reachable, which is not the same as keeping it loaded: see
// the lifetime note on Description.
func (d Description) Kernel() *cuda.Kernel { return d.kernel }

// Describe returns the launch l is configured to perform. It reports an error
// if l has no kernel.
//
// Describe does not detect a module that has already been closed. A Kernel
// keeps its driver.Function after its module is unloaded and cuda exposes no
// liveness predicate, so a Description taken from a closed module carries a
// handle that will fail, or worse, only when something launches it.
func (l *Launcher) Describe() (Description, error) {
	if l == nil {
		return Description{}, errors.New("cudalaunch: nil launcher")
	}
	fn := l.fn
	if fn == 0 && l.owned != nil {
		fn = l.owned.Raw()
	}
	if fn == 0 {
		return Description{}, errors.New("cudalaunch: nil kernel")
	}
	return Description{
		Function:  fn,
		Grid:      l.grid,
		Block:     l.block,
		SharedMem: l.shmem,
		Attrs:     l.Attrs(),
		kernel:    l.owned,
	}, nil
}
