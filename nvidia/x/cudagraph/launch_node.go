package cudagraph

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

// launchArgs owns the storage a kernel node's parameters point into.
//
// driver.KernelNodeParams.KernelParams is a uintptr, which the garbage
// collector does not trace. A graph node built from it therefore has no
// reference to the packed array, nor to the module the kernel came from.
// Graph.AddLaunch keeps a launchArgs on the graph, and Instantiate hands the
// set to the resulting Exec, so the storage outlives both the caller's locals
// and the graph itself.
type launchArgs struct {
	params []unsafe.Pointer
	desc   cudalaunch.Description
}

// AddLauncher adds a kernel node running what l is configured to launch. It is
// the common case of AddLaunch: a caller holding a Launcher needs no
// intermediate value.
func (g *Graph) AddLauncher(l *cudalaunch.Launcher, args []any, dependencies ...*Node) (*Node, error) {
	desc, err := l.Describe()
	if err != nil {
		return nil, err
	}
	return g.AddLaunch(desc, args, dependencies...)
}

// AddLaunch adds a kernel node executing desc with args, and returns it.
//
// The graph owns the packed arguments and desc's module for as long as the
// graph or an Exec instantiated from it is reachable, so the caller need not
// retain anything. AddLaunch reports an error for any launch attribute a
// kernel node cannot express, rather than dropping it: attributes on a graph
// launch are carried by edges and by instantiation options, not by the node.
// See AddProgrammaticDependency for programmatic dependent launch and
// AddDeviceUpdatableKernelNode for device-updatable nodes.
func (g *Graph) AddLaunch(desc cudalaunch.Description, args []any, dependencies ...*Node) (*Node, error) {
	if g == nil || g.raw == 0 {
		return nil, fmt.Errorf("cudagraph: graph is closed")
	}
	if desc.Function == 0 {
		return nil, fmt.Errorf("cudagraph: launch description has no kernel function")
	}
	if err := checkDim("grid", desc.Grid); err != nil {
		return nil, err
	}
	if err := checkDim("block", desc.Block); err != nil {
		return nil, err
	}
	for _, attr := range desc.Attrs {
		if attr.Id == driver.LaunchAttributeIgnore {
			continue
		}
		return nil, fmt.Errorf("cudagraph: launch attribute %v cannot be expressed as a kernel node", attr.Id)
	}

	params, err := cudalaunch.PackArgs(args...)
	if err != nil {
		return nil, fmt.Errorf("cudagraph: pack args: %w", err)
	}
	owner := &launchArgs{params: params, desc: desc}

	node := driver.KernelNodeParams{
		Func:           desc.Function,
		GridDimX:       desc.Grid.X,
		GridDimY:       desc.Grid.Y,
		GridDimZ:       desc.Grid.Z,
		BlockDimX:      desc.Block.X,
		BlockDimY:      desc.Block.Y,
		BlockDimZ:      desc.Block.Z,
		SharedMemBytes: desc.SharedMem,
	}
	if len(params) > 0 {
		node.KernelParams = uintptr(unsafe.Pointer(&params[0]))
	}

	// Retain before the driver reads through KernelParams, so the storage is
	// reachable from the graph for the whole call and everything after it.
	g.retain(owner)

	added, err := g.AddKernelNode(&node, dependencies...)
	runtime.KeepAlive(owner)
	if err != nil {
		g.release(owner)
		return nil, err
	}
	added.retained = owner
	return added, nil
}

// checkDim rejects a zero extent. A Launcher starts at one block of one
// thread, so a zero here is an explicit configuration, not an unset field, and
// substituting one for it would run a launch nobody asked for.
func checkDim(what string, d cudalaunch.Dim3) error {
	if d.X == 0 || d.Y == 0 || d.Z == 0 {
		return fmt.Errorf("cudagraph: %s dimension %dx%dx%d has a zero extent", what, d.X, d.Y, d.Z)
	}
	return nil
}

func (g *Graph) retain(a *launchArgs) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.retained = append(g.retained, a)
}

func (g *Graph) release(a *launchArgs) {
	g.mu.Lock()
	defer g.mu.Unlock()
	for i, held := range g.retained {
		if held == a {
			g.retained = append(g.retained[:i], g.retained[i+1:]...)
			return
		}
	}
}

// launchStorage returns a snapshot of the argument storage the graph owns.
func (g *Graph) launchStorage() []*launchArgs {
	g.mu.Lock()
	defer g.mu.Unlock()
	if len(g.retained) == 0 {
		return nil
	}
	out := make([]*launchArgs, len(g.retained))
	copy(out, g.retained)
	return out
}
