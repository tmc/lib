package cudagraph

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
)

// Node represents an individual execution node within a CUDA graph.
type Node struct {
	raw   driver.GraphNode
	graph *Graph
}

// Raw returns the underlying driver.GraphNode handle.
func (n *Node) Raw() driver.GraphNode {
	if n == nil {
		return 0
	}
	return n.raw
}

// Graph returns the graph that contains this node.
func (n *Node) Graph() *Graph {
	if n == nil {
		return nil
	}
	return n.graph
}

// Destroy destroys the node within its containing graph.
func (n *Node) Destroy() error {
	if n == nil || n.raw == 0 {
		return nil
	}
	if n.graph == nil {
		return errors.New("cudagraph: node has no owning graph")
	}
	if err := n.graph.withContext(n.raw.GraphDestroyNode); err != nil {
		return fmt.Errorf("cudagraph: destroy node: %w", err)
	}
	n.raw = 0
	return nil
}

// AddEmptyNode creates an empty node in g that performs no operation and can be used
// for barrier and dependency synchronization.
func (g *Graph) AddEmptyNode(dependencies ...*Node) (*Node, error) {
	if g == nil || g.raw == 0 {
		return nil, errors.New("cudagraph: graph is closed")
	}
	var depRaw []driver.GraphNode
	var depPtr *driver.GraphNode
	if len(dependencies) > 0 {
		depRaw = make([]driver.GraphNode, len(dependencies))
		for i, d := range dependencies {
			if d == nil || d.raw == 0 {
				return nil, errors.New("cudagraph: nil dependency node")
			}
			depRaw[i] = d.raw
		}
		depPtr = &depRaw[0]
	}
	var rawNode driver.GraphNode
	err := g.withContext(func() error {
		var err error
		rawNode, err = g.raw.AddEmptyNode(depPtr, uint(len(dependencies)))
		return err
	})
	if err != nil {
		return nil, fmt.Errorf("cudagraph: add empty node: %w", err)
	}
	return &Node{raw: rawNode, graph: g}, nil
}

// AddKernelNode creates a kernel execution node in g with the given parameters and dependencies.
func (g *Graph) AddKernelNode(params *driver.KernelNodeParams, dependencies ...*Node) (*Node, error) {
	if g == nil || g.raw == 0 {
		return nil, errors.New("cudagraph: graph is closed")
	}
	if params == nil {
		return nil, errors.New("cudagraph: nil kernel node params")
	}
	var depRaw []driver.GraphNode
	var depPtr *driver.GraphNode
	if len(dependencies) > 0 {
		depRaw = make([]driver.GraphNode, len(dependencies))
		for i, d := range dependencies {
			if d == nil || d.raw == 0 {
				return nil, errors.New("cudagraph: nil dependency node")
			}
			depRaw[i] = d.raw
		}
		depPtr = &depRaw[0]
	}
	var rawNode driver.GraphNode
	err := g.withContext(func() error {
		var err error
		rawNode, err = g.raw.AddKernelNode(depPtr, uint(len(dependencies)), params)
		return err
	})
	if err != nil {
		return nil, fmt.Errorf("cudagraph: add kernel node: %w", err)
	}
	return &Node{raw: rawNode, graph: g}, nil
}

// ProgrammaticEdgeData returns a driver.GraphEdgeData configured for a
// Programmatic Dependent Launch (PDL) dependency edge.
func ProgrammaticEdgeData() driver.GraphEdgeData {
	return driver.GraphEdgeData{
		FromPort: uint8(driver.GraphKernelNodePortProgrammatic),
		ToPort:   0,
		Type:     uint8(driver.GraphDependencyTypeProgrammatic),
	}
}

// AddDependencies adds dependency edges from nodes in from to nodes in to.
// from and to must have equal non-zero lengths.
// Optional edgeData specifies annotations (such as Programmatic edges). If omitted,
// standard (default) dependency edges are created.
func (g *Graph) AddDependencies(from, to []*Node, edgeData ...driver.GraphEdgeData) error {
	if g == nil || g.raw == 0 {
		return errors.New("cudagraph: graph is closed")
	}
	if len(from) != len(to) {
		return errors.New("cudagraph: mismatched from and to node slice lengths")
	}
	if len(from) == 0 {
		return nil
	}
	if len(edgeData) > 0 && len(edgeData) != len(from) {
		return errors.New("cudagraph: mismatched edgeData slice length")
	}

	fromRaw := make([]driver.GraphNode, len(from))
	toRaw := make([]driver.GraphNode, len(to))
	for i := range from {
		if from[i] == nil || from[i].raw == 0 || to[i] == nil || to[i].raw == 0 {
			return errors.New("cudagraph: nil node in dependency edge")
		}
		fromRaw[i] = from[i].raw
		toRaw[i] = to[i].raw
	}

	var edgePtr *driver.GraphEdgeData
	if len(edgeData) > 0 {
		edgePtr = &edgeData[0]
	}

	err := g.withContext(func() error {
		return g.raw.AddDependencies(&fromRaw[0], &toRaw[0], edgePtr, uint(len(from)))
	})
	if err != nil {
		return fmt.Errorf("cudagraph: add dependencies: %w", err)
	}
	return nil
}

// AddDependency adds a single standard dependency edge from from to to.
func (g *Graph) AddDependency(from, to *Node) error {
	return g.AddDependencies([]*Node{from}, []*Node{to})
}

// AddProgrammaticDependency adds a typed Programmatic dependent launch edge from from to to.
// Both from and to must be kernel execution nodes (created via [Graph.AddKernelNode]).
// Empty nodes (created via [Graph.AddEmptyNode]) do not possess programmatic completion ports
// and will be rejected by the CUDA driver with CUDA_ERROR_INVALID_VALUE.
func (g *Graph) AddProgrammaticDependency(from, to *Node) error {
	return g.AddDependencies([]*Node{from}, []*Node{to}, ProgrammaticEdgeData())
}
