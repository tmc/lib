package nccltest

import "testing"

func TestGraphValidationBeforeCapture(t *testing.T) {
	for _, states := range [][]rankState{nil, {{}}, {{stream: 1}, {}}} {
		called := false
		graphs, err := captureGraphs(states, func() error { called = true; return nil })
		if err == nil || called || len(graphs) != 0 {
			t.Fatalf("invalid capture: graphs=%v err=%v called=%v", graphs, err, called)
		}
	}
	if _, err := captureGraphs([]rankState{{stream: 1}}, nil); err == nil {
		t.Fatal("nil submission accepted")
	}
	if err := launchGraphs(nil, nil); err == nil {
		t.Fatal("empty launch accepted")
	}
	if err := launchGraphs([]rankState{{}}, nil); err == nil {
		t.Fatal("mismatched launch accepted")
	}
	if err := closeGraphs(nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := closeGraphs([]rankState{{}}, nil); err == nil {
		t.Fatal("mismatched cleanup accepted")
	}
}

// A cleanup error must preserve every rank, since captured NCCL work can
// reference peer buffers. A rank-count error exercises this without CUDA.
func TestGraphCleanupRetainsPeerResources(t *testing.T) {
	for _, graph := range []rankGraph{
		{capturing: true}, {graph: 1}, {executable: 1},
	} {
		states := make([]rankState, 2)
		if err := closeGraphs(states, []rankGraph{graph}); err == nil {
			t.Fatal("mismatched cleanup accepted")
		}
		for rank, state := range states {
			if !state.retainResources {
				t.Fatalf("rank %d resources would be freed with a live graph", rank)
			}
		}
	}
	states := make([]rankState, 2)
	if err := closeGraphs(states, make([]rankGraph, 2)); err != nil {
		t.Fatal(err)
	}
	for rank, state := range states {
		if state.retainResources {
			t.Fatalf("rank %d retained without a live graph", rank)
		}
	}
}
