package cudagraph

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

func describeOrFatal(t *testing.T, l *cudalaunch.Launcher) cudalaunch.Description {
	t.Helper()
	desc, err := l.Describe()
	if err != nil {
		t.Fatalf("Describe: %v", err)
	}
	return desc
}

func TestAddLaunchClosedGraph(t *testing.T) {
	desc := describeOrFatal(t, cudalaunch.Kernel(0x1000).Grid(1).Block(1))
	var g *Graph
	if _, err := g.AddLaunch(desc, nil); err == nil {
		t.Error("AddLaunch on a nil graph succeeded, want error")
	}
	if _, err := (&Graph{}).AddLaunch(desc, nil); err == nil {
		t.Error("AddLaunch on a closed graph succeeded, want error")
	}
}

func TestAddLaunchNoFunction(t *testing.T) {
	g := &Graph{raw: 1}
	if _, err := g.AddLaunch(cudalaunch.Description{}, nil); err == nil {
		t.Error("AddLaunch with no kernel function succeeded, want error")
	}
	if n := len(g.launchStorage()); n != 0 {
		t.Errorf("graph retained %d argument sets after a rejected launch, want 0", n)
	}
}

// TestAddLaunchUnsupportedAttribute checks the conversion rule the reflection
// helper this API replaced got wrong: an attribute a kernel node cannot carry
// must fail the conversion, not vanish from it.
func TestAddLaunchUnsupportedAttribute(t *testing.T) {
	for _, tt := range []struct {
		name     string
		launcher *cudalaunch.Launcher
	}{
		{"cooperative", cudalaunch.Kernel(0x1000).WithCooperative(true)},
		{"pdl", cudalaunch.Kernel(0x1000).WithProgrammaticStreamSerialization(true)},
		{"priority", cudalaunch.Kernel(0x1000).WithAttrs(driver.LaunchAttribute{Id: driver.LaunchAttributePriority})},
	} {
		t.Run(tt.name, func(t *testing.T) {
			g := &Graph{raw: 1}
			_, err := g.AddLaunch(describeOrFatal(t, tt.launcher), nil)
			if err == nil {
				t.Fatal("AddLaunch accepted an attribute a kernel node cannot express")
			}
			if !strings.Contains(err.Error(), "cannot be expressed") {
				t.Errorf("error = %v, want it to name the unexpressible attribute", err)
			}
			if n := len(g.launchStorage()); n != 0 {
				t.Errorf("graph retained %d argument sets after a rejected launch, want 0", n)
			}
		})
	}
}

// TestAddLaunchIgnoredAttribute checks that the composition filler attribute,
// which carries no launch configuration, is not mistaken for a real one.
func TestAddLaunchIgnoredAttribute(t *testing.T) {
	desc := describeOrFatal(t, cudalaunch.Kernel(0x1000).
		WithAttrs(driver.LaunchAttribute{Id: driver.LaunchAttributeIgnore}))
	g := &Graph{raw: 1}
	_, err := g.AddLaunch(desc, nil)
	if err != nil && strings.Contains(err.Error(), "cannot be expressed") {
		t.Errorf("AddLaunch rejected the ignore attribute: %v", err)
	}
}

func TestAddLaunchBadArgument(t *testing.T) {
	desc := describeOrFatal(t, cudalaunch.Kernel(0x1000).Grid(1).Block(1))
	g := &Graph{raw: 1}
	if _, err := g.AddLaunch(desc, []any{nil}); err == nil {
		t.Fatal("AddLaunch packed a nil argument, want error")
	}
	if n := len(g.launchStorage()); n != 0 {
		t.Errorf("graph retained %d argument sets after a pack failure, want 0", n)
	}
}

func TestGraphLaunchStorageSnapshot(t *testing.T) {
	g := &Graph{raw: 1}
	a, b := &launchArgs{}, &launchArgs{}
	g.retain(a)
	g.retain(b)
	held := g.launchStorage()
	if len(held) != 2 {
		t.Fatalf("launchStorage = %d entries, want 2", len(held))
	}
	g.release(a)
	if len(held) != 2 {
		t.Error("launchStorage returned a live view of the graph's slice, want a snapshot")
	}
	if got := g.launchStorage(); len(got) != 1 || got[0] != b {
		t.Errorf("after release, launchStorage = %v, want only the remaining entry", got)
	}
}
