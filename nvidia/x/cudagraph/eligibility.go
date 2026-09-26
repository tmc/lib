package cudagraph

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
)

// Eligibility describes the structural validation, node composition, and
// replay suitability of a CUDA execution graph.
type Eligibility struct {
	Captured bool     // True if the graph has been successfully captured and can be instantiated
	Reasons  []string // Diagnostic reasons explaining any ineligibility or replay caveats
	Nodes    int      // Total number of nodes in the graph
	Kernels  int      // Number of kernel execution nodes
	Updates  []string // Descriptions of nodes requiring or supporting dynamic parameter updates
}

// Eligibility inspects the graph DAG, counts node types, discovers dynamic parameter
// update points, and verifies instantiation readiness.
func (g *Graph) Eligibility() (Eligibility, error) {
	if g == nil || g.raw == 0 {
		return Eligibility{
			Captured: false,
			Reasons:  []string{"graph is closed or uninitialized"},
		}, errors.New("cudagraph: graph is closed")
	}

	const maxNodes = 65536
	var nodes []driver.GraphNode
	err := g.withContext(func() error {
		var err error
		nodes, err = g.raw.NodesBounded(maxNodes)
		return err
	})
	if err != nil {
		return Eligibility{
			Captured: false,
			Reasons:  []string{fmt.Sprintf("failed to query nodes: %v", err)},
		}, fmt.Errorf("cudagraph: query nodes: %w", err)
	}

	el := Eligibility{
		Nodes: len(nodes),
	}

	if len(nodes) == 0 {
		el.Captured = false
		el.Reasons = append(el.Reasons, "graph has no nodes")
		return el, nil
	}

	err = g.withContext(func() error {
		for i, node := range nodes {
			kind, err := node.Type()
			if err != nil {
				el.Reasons = append(el.Reasons, fmt.Sprintf("node %d: get type: %v", i, err))
				continue
			}

			switch kind {
			case driver.GraphNodeTypeKernel:
				el.Kernels++
				attr, err := driver.KernelNode{GraphNode: node}.Attribute(int32(driver.LaunchAttributeDeviceUpdatableKernelNode))
				if err == nil && attr[0]&0xffffffff != 0 {
					el.Updates = append(el.Updates, fmt.Sprintf("node %d: device-updatable kernel node", i))
				}
				params, err := driver.KernelNode{GraphNode: node}.Params()
				if err != nil {
					el.Reasons = append(el.Reasons, fmt.Sprintf("node %d: get params: %v", i, err))
				} else if params.Func == 0 && params.Kern == 0 {
					el.Reasons = append(el.Reasons, fmt.Sprintf("node %d: kernel node has no function handle", i))
				}
			case driver.GraphNodeTypeMemcpy, driver.GraphNodeTypeMemset,
				driver.GraphNodeTypeEmpty, driver.GraphNodeTypeMemAlloc, driver.GraphNodeTypeMemFree:
				// Supported standard execution and memory nodes
			default:
				el.Reasons = append(el.Reasons, fmt.Sprintf("node %d: unsupported node type %s", i, kind))
			}
		}

		// Attempt instantiation to verify driver acceptance
		rawExec, err := g.raw.Instantiate(0)
		if err != nil {
			el.Reasons = append(el.Reasons, fmt.Sprintf("driver instantiation check failed: %v", err))
		} else {
			_ = rawExec.Close()
		}
		return nil
	})
	if err != nil {
		return el, err
	}

	el.Captured = len(el.Reasons) == 0
	return el, nil
}
