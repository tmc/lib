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
// A Description holds the owning Kernel, so the module stays loaded for as long
// as the Description is reachable. The zero Description is not usable; obtain
// one from Launcher.Describe.
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
// launch targets a bare driver function. Retaining the Description retains the
// kernel's module.
func (d Description) Kernel() *cuda.Kernel { return d.kernel }

// Describe returns the launch l is configured to perform. It reports an error
// if l has no kernel.
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
