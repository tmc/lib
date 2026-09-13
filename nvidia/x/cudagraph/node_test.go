package cudagraph_test

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
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
	if _, err := closedGraph.AddKernelNode(nil); err == nil {
		t.Error("AddKernelNode on nil graph expected error, got nil")
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

	compiled, err := nvrtcx.Compile(incrementKernel, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatal(err)
	}
	mod, err := compiled.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(mod.Unload)
	fn, err := mod.GetFunction("increment")
	if err != nil {
		t.Fatal(err)
	}

	buf, err := cudamem.Alloc[uint32](dev, 128)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(buf.Free)

	dptr := buf.Ptr()
	count := int32(128)
	args := []unsafe.Pointer{
		unsafe.Pointer(&dptr),
		unsafe.Pointer(&count),
	}

	err = dev.WithContext(func() error {
		g, err := cudagraph.New()
		if err != nil {
			return err
		}
		defer g.Destroy()

		n1, err := g.AddEmptyNode()
		if err != nil {
			return fmt.Errorf("AddEmptyNode n1: %w", err)
		}
		if n1.Raw() == 0 {
			t.Error("expected non-zero raw node handle")
		}
		if n1.Graph() != g {
			t.Error("node graph pointer mismatch")
		}

		n2, err := g.AddEmptyNode(n1)
		if err != nil {
			return fmt.Errorf("AddEmptyNode n2: %w", err)
		}

		n3, err := g.AddEmptyNode()
		if err != nil {
			return fmt.Errorf("AddEmptyNode n3: %w", err)
		}

		// Add manual standard dependency edge between empty nodes
		if err := g.AddDependency(n2, n3); err != nil {
			return fmt.Errorf("AddDependency failed: %w", err)
		}

		// Add kernel nodes for programmatic dependency edge (PDL requires kernel nodes)
		kp1 := driver.KernelNodeParams{
			Func:         fn,
			GridDimX:     1,
			GridDimY:     1,
			GridDimZ:     1,
			BlockDimX:    128,
			BlockDimY:    1,
			BlockDimZ:    1,
			KernelParams: uintptr(unsafe.Pointer(&args[0])),
		}
		k1, err := g.AddKernelNode(&kp1, n3)
		if err != nil {
			return fmt.Errorf("AddKernelNode k1: %w", err)
		}

		kp2 := driver.KernelNodeParams{
			Func:         fn,
			GridDimX:     1,
			GridDimY:     1,
			GridDimZ:     1,
			BlockDimX:    128,
			BlockDimY:    1,
			BlockDimZ:    1,
			KernelParams: uintptr(unsafe.Pointer(&args[0])),
		}
		k2, err := g.AddKernelNode(&kp2)
		if err != nil {
			return fmt.Errorf("AddKernelNode k2: %w", err)
		}

		// Add programmatic dependency edge between kernel nodes
		if err := g.AddProgrammaticDependency(k1, k2); err != nil {
			return fmt.Errorf("AddProgrammaticDependency failed: %w", err)
		}

		// Instantiate graph to verify topology validity
		exec, err := g.Instantiate()
		if err != nil {
			return fmt.Errorf("Instantiate failed: %w", err)
		}
		defer exec.Destroy()

		// Destroy node
		if err := k2.Destroy(); err != nil {
			t.Errorf("Destroy node failed: %v", err)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
