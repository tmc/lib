package cudagraph

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

// storageOwner builds a graph holding one distinguishable piece of argument
// storage. The kernel function stands in for the module owner: two graphs
// built from different modules retain different Descriptions, which is the
// difference ownership has to follow. Argument values are not the test,
// because the native graph may copy them.
func storageOwner(fn uintptr) *Graph {
	g := &Graph{raw: 1}
	g.retain(&launchArgs{desc: cudalaunch.Description{Function: driver.Function(fn)}})
	return g
}

func ownedFunctions(t *testing.T, held []*launchArgs) []uintptr {
	t.Helper()
	var out []uintptr
	for _, a := range held {
		out = append(out, uintptr(a.desc.Function))
	}
	return out
}

// TestExecAdoptsUpdatedStorage covers the ownership transfer an update makes.
// The driver reads kernel parameters through an untraced uintptr, so an Exec
// still holding only the previous graph's storage would let the storage it now
// executes be collected.
func TestExecAdoptsUpdatedStorage(t *testing.T) {
	old := storageOwner(0x1111)
	next := storageOwner(0x2222)
	e := &Exec{raw: 1, retained: old.launchStorage()}

	e.adoptStorage(next)

	got := ownedFunctions(t, e.retained)
	if len(got) != 1 || got[0] != 0x2222 {
		t.Fatalf("after adopting, Exec owns %#x, want the updated graph's [0x2222]", got)
	}
}

// TestExecKeepsStorageWhenUpdateFails is the other half: a rejected update
// leaves the Exec executing what it executed before, so it must keep owning
// that. A closed graph is refused before the driver is reached.
func TestExecKeepsStorageWhenUpdateFails(t *testing.T) {
	old := storageOwner(0x1111)
	e := &Exec{raw: 1, retained: old.launchStorage()}

	err := e.Update(&Graph{raw: 0})
	if err == nil {
		t.Fatal("Update accepted a closed graph")
	}
	if !strings.Contains(err.Error(), "graph is closed") {
		t.Fatalf("Update error = %v, want it to name the closed graph", err)
	}
	got := ownedFunctions(t, e.retained)
	if len(got) != 1 || got[0] != 0x1111 {
		t.Fatalf("after a failed update, Exec owns %#x, want the original [0x1111]", got)
	}
}

// TestExecAdoptsEmptyStorage checks that adopting a graph with no retained
// launches releases the previous storage rather than keeping it alive. A graph
// built by stream capture or from raw kernel nodes retains nothing.
func TestExecAdoptsEmptyStorage(t *testing.T) {
	e := &Exec{raw: 1, retained: storageOwner(0x1111).launchStorage()}
	e.adoptStorage(&Graph{raw: 1})
	if len(e.retained) != 0 {
		t.Fatalf("Exec still owns %#x after adopting a graph that retains nothing", ownedFunctions(t, e.retained))
	}
}
