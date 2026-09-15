package cudagraph

import (
	"errors"
	"fmt"
	"runtime"

	"github.com/tmc/lib/nvidia/driver"
)

// KernelOrder returns the executable handle and a copy of its retained kernel
// symbols in dependency order. RetainKernelOrder must have been selected when
// instantiating. Update and Destroy invalidate this evidence. Source graph
// destruction does not invalidate the snapshot.
//
// The caller must not mutate the executable through raw driver bindings or
// concurrently with profiling. This evidence describes a native graph topology;
// associating it with profiler ordinals still requires a qualified collector.
func (e *Exec) KernelOrder() (driver.GraphExec, []string, error) {
	if e == nil || e.raw == 0 {
		return 0, nil, errors.New("cudagraph: executable graph is closed")
	}
	if e.kernelOrder == nil {
		return 0, nil, errors.New("cudagraph: executable graph has no retained kernel order")
	}
	return e.raw, append([]string(nil), e.kernelOrder...), nil
}

func snapshotKernelOrder(g driver.Graph) ([]string, error) {
	const limit = 65536
	nodes, err := g.NodesBounded(limit)
	if err != nil {
		return nil, err
	}
	var count uint
	if err := g.GetEdges(nil, nil, nil, &count); err != nil {
		return nil, err
	}
	if count > limit {
		return nil, errors.New("cudagraph: graph edge capacity exceeded")
	}
	from, to := make([]driver.GraphNode, count), make([]driver.GraphNode, count)
	if count > 0 {
		var pin runtime.Pinner
		pin.Pin(&from[0])
		pin.Pin(&to[0])
		defer pin.Unpin()
		// A nil edge-data output rejects nondefault dependency types with CUDA's
		// LOSSY_QUERY error. Only full completion dependencies prove this order.
		if err := g.GetEdges(&from[0], &to[0], nil, &count); err != nil {
			return nil, err
		}
		if count != uint(len(from)) {
			return nil, errors.New("cudagraph: graph edge count changed")
		}
	}
	symbols := make(map[driver.GraphNode]string, len(nodes))
	for _, node := range nodes {
		kind, err := node.GetType()
		if err != nil {
			return nil, err
		}
		switch kind {
		case driver.GraphNodeTypeEmpty, driver.GraphNodeTypeMemcpy, driver.GraphNodeTypeMemset,
			driver.GraphNodeTypeMemAlloc, driver.GraphNodeTypeMemFree:
			// Memory nodes contribute completion dependencies, not kernel symbols.
			// CUDA Programming Guide 4.2.5 defines their GPU-ordered lifetimes:
			// https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#graph-memory-nodes
			// A profiler must still reconcile its native range count; this topology
			// does not prove how the driver implements allocation or other actions.
		case driver.GraphNodeTypeKernel:
			attr, err := node.GraphKernelNodeGetAttribute(int32(driver.LaunchAttributeDeviceUpdatableKernelNode))
			if err != nil {
				return nil, err
			}
			if attr[0]&0xffffffff != 0 {
				return nil, errors.New("cudagraph: kernel order does not support device-updatable nodes")
			}
			params, err := node.GraphKernelNodeGetParams()
			if err != nil {
				return nil, err
			}
			var name string
			if params.Func != 0 {
				name, err = params.Func.GetName()
			} else if params.Kern != 0 {
				name, err = params.Kern.GetName()
			} else {
				return nil, errors.New("cudagraph: kernel node has no function")
			}
			if err != nil {
				return nil, err
			}
			if name == "" {
				return nil, errors.New("cudagraph: kernel node has no symbol")
			}
			symbols[node] = name
		default:
			return nil, fmt.Errorf("cudagraph: kernel order does not support node type %s", kind)
		}
	}
	return orderKernels(nodes, from, to, symbols)
}

func orderKernels(nodes, from, to []driver.GraphNode, symbols map[driver.GraphNode]string) ([]string, error) {
	if len(nodes) == 0 {
		return nil, errors.New("cudagraph: graph has no kernels")
	}
	if len(from) != len(to) {
		return nil, errors.New("cudagraph: invalid graph edges")
	}
	degree := make(map[driver.GraphNode]int, len(nodes))
	next := make(map[driver.GraphNode][]driver.GraphNode, len(nodes))
	for _, n := range nodes {
		if n == 0 {
			return nil, errors.New("cudagraph: null graph node")
		}
		if _, ok := degree[n]; ok {
			return nil, errors.New("cudagraph: duplicate graph node")
		}
		degree[n] = 0
	}
	edges := make(map[[2]driver.GraphNode]bool, len(from))
	for i, a := range from {
		b := to[i]
		if _, ok := degree[a]; !ok {
			return nil, errors.New("cudagraph: unknown graph edge source")
		}
		if _, ok := degree[b]; !ok {
			return nil, errors.New("cudagraph: unknown graph edge target")
		}
		edge := [2]driver.GraphNode{a, b}
		if edges[edge] {
			return nil, errors.New("cudagraph: duplicate graph edge")
		}
		edges[edge] = true
		degree[b]++
		next[a] = append(next[a], b)
	}
	var ready, kernels []driver.GraphNode
	enqueue := func(n driver.GraphNode) {
		if symbols[n] == "" {
			ready = append(ready, n)
		} else {
			kernels = append(kernels, n)
		}
	}
	for _, n := range nodes {
		if degree[n] == 0 {
			enqueue(n)
		}
	}
	var names []string
	seen := 0
	visit := func(n driver.GraphNode) {
		seen++
		for _, m := range next[n] {
			degree[m]--
			if degree[m] == 0 {
				enqueue(m)
			}
		}
	}
	for len(ready) > 0 || len(kernels) > 0 {
		// Independent nonkernel nodes need no relative order. Drain the entire
		// frontier: any kernel they can release competes with kernels ready now.
		for len(ready) > 0 {
			n := ready[len(ready)-1]
			ready = ready[:len(ready)-1]
			visit(n)
		}
		if len(kernels) > 1 {
			return nil, errors.New("cudagraph: graph kernel order is ambiguous")
		}
		if len(kernels) == 1 {
			n := kernels[0]
			kernels = kernels[:0]
			names = append(names, symbols[n])
			visit(n)
		}
	}
	if seen != len(nodes) {
		return nil, errors.New("cudagraph: graph contains a dependency cycle")
	}
	if len(names) == 0 {
		return nil, errors.New("cudagraph: graph has no kernels")
	}
	return names, nil
}
