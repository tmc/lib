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
		GridDimX:       atLeastOne(desc.Grid.X),
		GridDimY:       atLeastOne(desc.Grid.Y),
		GridDimZ:       atLeastOne(desc.Grid.Z),
		BlockDimX:      atLeastOne(desc.Block.X),
		BlockDimY:      atLeastOne(desc.Block.Y),
		BlockDimZ:      atLeastOne(desc.Block.Z),
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

func atLeastOne(v uint32) uint32 {
	if v == 0 {
		return 1
	}
	return v
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
