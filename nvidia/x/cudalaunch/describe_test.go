package cudalaunch_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

func TestDescribe(t *testing.T) {
	const fn = driver.Function(0x1000)
	l := cudalaunch.Kernel(fn).Grid(4, 2, 3).Block(128, 2).SharedMem(2048).WithCooperative(true)

	desc, err := l.Describe()
	if err != nil {
		t.Fatalf("Describe: %v", err)
	}
	if desc.Function != fn {
		t.Errorf("Function = %#x, want %#x", desc.Function, fn)
	}
	if want := (cudalaunch.Dim3{X: 4, Y: 2, Z: 3}); desc.Grid != want {
		t.Errorf("Grid = %+v, want %+v", desc.Grid, want)
	}
	if want := (cudalaunch.Dim3{X: 128, Y: 2, Z: 1}); desc.Block != want {
		t.Errorf("Block = %+v, want %+v", desc.Block, want)
	}
	if desc.SharedMem != 2048 {
		t.Errorf("SharedMem = %d, want 2048", desc.SharedMem)
	}
	if len(desc.Attrs) != 1 || desc.Attrs[0].Id != driver.LaunchAttributeCooperative {
		t.Errorf("Attrs = %+v, want one cooperative attribute", desc.Attrs)
	}
	if desc.Kernel() != nil {
		t.Errorf("Kernel() = %v, want nil for a bare driver function", desc.Kernel())
	}
}

// TestDescribeAttrsCopied checks that a Description does not alias the
// launcher, so a later reconfiguration cannot rewrite a description already
// handed to a graph.
func TestDescribeAttrsCopied(t *testing.T) {
	l := cudalaunch.Kernel(0x1000).WithCooperative(true)
	desc, err := l.Describe()
	if err != nil {
		t.Fatalf("Describe: %v", err)
	}
	l.WithAttrs(driver.LaunchAttribute{Id: driver.LaunchAttributePriority})
	l.Grid(99)
	if len(desc.Attrs) != 1 {
		t.Errorf("Attrs grew to %d after reconfiguring the launcher", len(desc.Attrs))
	}
	if desc.Grid.X != 1 {
		t.Errorf("Grid.X = %d, want the value at Describe time", desc.Grid.X)
	}
}

func TestDescribeNoKernel(t *testing.T) {
	if _, err := cudalaunch.Kernel(0).Describe(); err == nil {
		t.Error("Describe with no kernel succeeded, want error")
	}
	var nilLauncher *cudalaunch.Launcher
	if _, err := nilLauncher.Describe(); err == nil {
		t.Error("Describe on a nil launcher succeeded, want error")
	}
}
