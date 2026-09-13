package cudagraph_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

func TestProgrammaticEdgeData(t *testing.T) {
	edge := cudagraph.ProgrammaticEdgeData()
	if edge.Type != uint8(driver.GraphDependencyTypeProgrammatic) {
		t.Errorf("edge.Type = %d, want %d", edge.Type, driver.GraphDependencyTypeProgrammatic)
	}
	if edge.FromPort != uint8(driver.GraphKernelNodePortProgrammatic) {
		t.Errorf("edge.FromPort = %d, want %d", edge.FromPort, driver.GraphKernelNodePortProgrammatic)
	}
}

func TestNodeClosedGraphRejection(t *testing.T) {
	var closedGraph *cudagraph.Graph
	if _, err := closedGraph.AddEmptyNode(); err == nil {
		t.Error("AddEmptyNode on nil graph expected error, got nil")
	}
	if err := closedGraph.AddDependencies(nil, nil); err == nil {
		t.Error("AddDependencies on nil graph expected error, got nil")
	}

	var closedNode *cudagraph.Node
	if err := closedNode.Destroy(); err != nil {
		t.Errorf("Destroy on nil node expected nil error, got %v", err)
	}
	if closedNode.Raw() != 0 {
		t.Errorf("Raw on nil node = %v, want 0", closedNode.Raw())
	}
	if closedNode.Graph() != nil {
		t.Errorf("Graph on nil node = %v, want nil", closedNode.Graph())
	}
}

func TestNodeDependencyValidation(t *testing.T) {
	dev := cudatest.Require(t)
	_ = dev

	g, err := cudagraph.New()
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()

	// Empty slices is a no-op
	if err := g.AddDependencies(nil, nil); err != nil {
		t.Errorf("empty AddDependencies failed: %v", err)
	}

	// Mismatched lengths
	node1 := &cudagraph.Node{}
	node2 := &cudagraph.Node{}
	if err := g.AddDependencies([]*cudagraph.Node{node1}, []*cudagraph.Node{node1, node2}); err == nil {
		t.Error("AddDependencies with mismatched slice lengths expected error, got nil")
	}

	// Mismatched edgeData lengths
	edgeData := cudagraph.ProgrammaticEdgeData()
	if err := g.AddDependencies([]*cudagraph.Node{node1}, []*cudagraph.Node{node2}, edgeData, edgeData); err == nil {
		t.Error("AddDependencies with mismatched edgeData expected error, got nil")
	}

	// Nil node in dependency edge
	if err := g.AddDependencies([]*cudagraph.Node{nil}, []*cudagraph.Node{node2}); err == nil {
		t.Error("AddDependencies with nil node expected error, got nil")
	}
}

func TestNodeGraphConstruction(t *testing.T) {
	dev := cudatest.Require(t)
	_ = dev

	g, err := cudagraph.New()
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()

	n1, err := g.AddEmptyNode()
	if err != nil {
		t.Fatalf("AddEmptyNode n1 failed: %v", err)
	}
	if n1.Raw() == 0 {
		t.Error("expected non-zero raw node handle")
	}
	if n1.Graph() != g {
		t.Error("node graph pointer mismatch")
	}

	n2, err := g.AddEmptyNode(n1)
	if err != nil {
		t.Fatalf("AddEmptyNode n2 with dependency failed: %v", err)
	}

	n3, err := g.AddEmptyNode()
	if err != nil {
		t.Fatalf("AddEmptyNode n3 failed: %v", err)
	}

	// Add manual dependency edge
	if err := g.AddDependency(n2, n3); err != nil {
		t.Fatalf("AddDependency failed: %v", err)
	}

	// Add programmatic dependency edge
	n4, err := g.AddEmptyNode()
	if err != nil {
		t.Fatalf("AddEmptyNode n4 failed: %v", err)
	}
	if err := g.AddProgrammaticDependency(n3, n4); err != nil {
		t.Fatalf("AddProgrammaticDependency failed: %v", err)
	}

	// Instantiate graph to verify topology validity
	exec, err := g.Instantiate()
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}
	defer exec.Destroy()

	// Destroy node
	if err := n4.Destroy(); err != nil {
		t.Errorf("Destroy node failed: %v", err)
	}
}
