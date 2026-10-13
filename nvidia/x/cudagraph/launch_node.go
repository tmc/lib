package cudagraph

import (
	"fmt"
	"reflect"
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalayout"
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
	params  []unsafe.Pointer
	desc    cudalaunch.Description
	sizes   []uintptr
	buffers [][]byte
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
// The graph owns the packed arguments and keeps desc's module reachable for as
// long as the graph or an Exec instantiated from it is reachable, so the caller
// need not retain either against the collector. It is not a lease on the
// module: [cuda.Module.Close] unloads it regardless, and the node's function
// handle then names nothing loaded. The caller must not close the module until
// every graph and Exec built from desc is done with it.
//
// AddLaunch reports an error for any launch attribute a
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
	owner := &launchArgs{params: params, desc: desc, sizes: argumentSizes(args, len(params))}

	node := driver.KernelNodeParams{
		Func:           desc.Function,
		GridDim:        driver.Dim3(desc.Grid),
		BlockDim:       driver.Dim3(desc.Block),
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
	g.recordKernelNode(added.raw, node, owner)
	return added, nil
}

// checkDim rejects a zero extent. A Launcher starts at one block of one
// thread, so a zero here is an explicit configuration, not an unset field, and
// substituting one for it would run a launch nobody asked for.
func checkDim(what string, d driver.Dim3) error {
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
	for _, state := range g.kernelNodes {
		if state.args == a {
			return
		}
	}
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

// argumentSizes describes PackArgs storage without guessing expanded or raw
// layouts. Native parameter offsets and sizes are checked before byte updates.
func argumentSizes(args []any, count int) []uintptr {
	if len(args) != count || count > 128 {
		return nil
	}
	sizes := make([]uintptr, count)
	var total uintptr
	for i, arg := range args {
		switch arg.(type) {
		case cudalayout.Partition, cudalayout.PartitionMut:
			return nil
		}
		if _, ok := arg.(cudalaunch.DevicePtrProvider); ok {
			sizes[i] = 8
		} else {
			t := reflect.TypeOf(arg)
			if t == nil {
				return nil
			}
			if t.Kind() == reflect.Pointer {
				t = t.Elem()
			}
			switch t.Kind() {
			case reflect.Int, reflect.Uint:
				sizes[i] = 8
			case reflect.Bool:
				sizes[i] = 1
			default:
				sizes[i] = t.Size()
			}
		}
		if sizes[i] == 0 || sizes[i] > 65536-total {
			return nil
		}
		total += sizes[i]
	}
	return sizes
}

// kernelNodeState is an executable snapshot, not the mutable original hNode.
type kernelNodeState struct {
	params driver.KernelNodeParams
	args   *launchArgs
}

func (g *Graph) recordKernelNode(node driver.GraphNode, params driver.KernelNodeParams, args *launchArgs) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.kernelNodes == nil {
		g.kernelNodes = make(map[driver.GraphNode]kernelNodeState)
	}
	if args == nil && params.KernelParams != 0 && params.Extra == 0 {
		for _, held := range g.retained {
			if len(held.params) != 0 && uintptr(unsafe.Pointer(&held.params[0])) == params.KernelParams {
				args = held
				break
			}
		}
	}
	g.kernelNodes[node] = kernelNodeState{params, args}
}

func (g *Graph) kernelStorage() map[driver.GraphNode]kernelNodeState {
	g.mu.Lock()
	defer g.mu.Unlock()
	out := make(map[driver.GraphNode]kernelNodeState, len(g.kernelNodes))
	for node, state := range g.kernelNodes {
		out[node] = state
	}
	return out
}
