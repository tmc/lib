package cudaplan

import (
	"errors"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudagraph"
)

func TestGraphBuildFailureOwnership(t *testing.T) {
	failure := errors.New("injected graph failure")
	stages := []string{"destroy exec", "destroy graph", "create", "first", "second", "instantiate", "success"}
	for index, stage := range stages {
		t.Run(stage, func(t *testing.T) {
			oldGraph, newGraph := &cudagraph.Graph{}, &cudagraph.Graph{}
			oldExec, newExec := &cudagraph.Exec{}, &cudagraph.Exec{}
			first, second := &cudagraph.Node{}, &cudagraph.Node{}
			p := &ExecutablePlan{
				graphObj: oldGraph, graphExec: oldExec, rebuildGraph: true,
				graphPointers: map[string]driver.DevicePtr{"x": 8},
				steps:         []StepPlan{{NodeName: "first"}, {NodeName: "second"}},
			}
			var calls []string
			visit := func(name string) error {
				calls = append(calls, name)
				if name == stage {
					return failure
				}
				return nil
			}
			ops := graphBuildOps{
				destroyExec: func(e *cudagraph.Exec) error {
					if e != oldExec {
						t.Fatal("wrong old executable")
					}
					return visit("destroy exec")
				},
				destroyGraph: func(g *cudagraph.Graph) error {
					if g != oldGraph {
						t.Fatal("wrong old graph")
					}
					return visit("destroy graph")
				},
				create: func() (*cudagraph.Graph, error) { return newGraph, visit("create") },
				add: func(g *cudagraph.Graph, step StepPlan, pointers map[string]driver.DevicePtr, previous *cudagraph.Node) (*cudagraph.Node, error) {
					if g != newGraph || pointers["x"] != 16 {
						t.Fatal("wrong graph or bindings")
					}
					if step.NodeName == "first" {
						if previous != nil {
							t.Fatal("first node depends on stale graph")
						}
						return first, visit("first")
					}
					if previous != first {
						t.Fatal("lost sequential workspace dependency")
					}
					return second, visit("second")
				},
				instantiate: func(g *cudagraph.Graph) (*cudagraph.Exec, error) {
					if g != newGraph {
						t.Fatal("instantiated wrong graph")
					}
					return newExec, visit("instantiate")
				},
			}
			pointers := map[string]driver.DevicePtr{"x": 16}
			err := p.prepareGraphWith(pointers, ops)
			if stage == "success" {
				if err != nil || p.closed || p.rebuildGraph || p.graphObj != newGraph || p.graphExec != newExec {
					t.Fatalf("bad completed build: %v", err)
				}
				pointers["x"] = 32
				if p.graphPointers["x"] != 16 {
					t.Fatal("caller mutated captured identity")
				}
				if err := p.prepareGraphWith(map[string]driver.DevicePtr{"x": 16}, graphBuildOps{}); err != nil {
					t.Fatal(err)
				}
			} else {
				if !errors.Is(err, failure) || !p.closed || !p.rebuildGraph {
					t.Fatalf("failed build remained executable: %v", err)
				}
				if p.graphPointers["x"] != 8 {
					t.Fatal("failed build published new bindings")
				}
				if _, err := p.Execute(nil, nil); !errors.Is(err, ErrClosed) {
					t.Fatalf("partial graph executable: %v", err)
				}
			}
			limit := index + 1
			if stage == "success" {
				limit = len(stages) - 1
			}
			if !reflect.DeepEqual(calls, stages[:limit]) {
				t.Fatalf("calls = %v, want %v", calls, stages[:limit])
			}
			wantGraph, wantExec := newGraph, newExec
			if index < 2 {
				wantGraph = oldGraph
			}
			if index == 0 {
				wantExec = oldExec
			} else if index < 5 {
				wantExec = nil
			}
			if p.graphObj != wantGraph || p.graphExec != wantExec {
				t.Fatal("lost failed-build resource owner")
			}
			cleanupCalls := 0
			cleanup := planCloseOps{
				graphExec: func(e *cudagraph.Exec) error {
					if e != wantExec {
						t.Fatal("wrong cleanup exec")
					}
					cleanupCalls++
					return nil
				},
				graph: func(g *cudagraph.Graph) error {
					if g != wantGraph {
						t.Fatal("wrong cleanup graph")
					}
					cleanupCalls++
					return nil
				},
			}
			for range 2 {
				if err := p.closeWith(cleanup); err != nil {
					t.Fatal(err)
				}
			}
			wantCalls := 1
			if wantExec != nil {
				wantCalls++
			}
			if cleanupCalls != wantCalls {
				t.Fatalf("cleanup calls = %d, want %d", cleanupCalls, wantCalls)
			}
		})
	}
}

func TestReplayBindingIdentity(t *testing.T) {
	p := &ExecutablePlan{
		graphExec:     &cudagraph.Exec{},
		graphPointers: map[string]driver.DevicePtr{"input": 0x1000, "output": 0x2000},
	}
	if err := p.prepareGraph(map[string]driver.DevicePtr{"output": 0x2000, "input": 0x1000}); err != nil {
		t.Fatal(err)
	}
	for _, pointers := range []map[string]driver.DevicePtr{
		{"input": 0x3000, "output": 0x2000},
		{"input": 0x1000, "output": 0x3000},
		{"input": 0x1000},
		{"input": 0x1000, "output": 0x2000, "workspace": 0x4000},
	} {
		if err := p.prepareGraph(pointers); !errors.Is(err, ErrStaleBinding) {
			t.Fatalf("changed bindings accepted: %v", err)
		}
		if p.closed {
			t.Fatal("stale binding made plan uncloseable for rebinding")
		}
	}
	old := p.graphExec
	p.inFlight = true
	p.InvalidateGraphBindings()
	if !p.staleGraph {
		t.Fatal("invalidation lost")
	}
	p.RebindGraph()
	if p.staleGraph || !p.rebuildGraph || p.graphExec != old {
		t.Fatal("rebinding must request replacement without destroying an active graph")
	}
	if _, err := p.Execute(nil, nil); !errors.Is(err, ErrInvocationInFlight) {
		t.Fatalf("rebind bypassed invocation ownership: %v", err)
	}
}

func TestCapturedGraphOwnership(t *testing.T) {
	enqueueFailure := errors.New("enqueue failure")
	endFailure := errors.New("end capture failure")
	for _, fail := range []bool{false, true} {
		p := &ExecutablePlan{steps: []StepPlan{{Backend: BackendCuBLASLt}, {Backend: BackendGraph}}}
		graph, exec := &cudagraph.Graph{}, &cudagraph.Exec{}
		calls := 0
		ops := graphBuildOps{
			captured: true,
			create: func() (*cudagraph.Graph, error) {
				return capturePlan(func(fn func()) (*cudagraph.Graph, error) {
					fn()
					if fail {
						return graph, endFailure
					}
					return graph, nil
				}, func() error {
					calls++
					if fail {
						return enqueueFailure
					}
					return nil
				})
			},
			instantiate: func(g *cudagraph.Graph) (*cudagraph.Exec, error) {
				if fail || g != graph {
					t.Fatal("invalid graph instantiated")
				}
				return exec, nil
			},
			add: func(*cudagraph.Graph, StepPlan, map[string]driver.DevicePtr, *cudagraph.Node) (*cudagraph.Node, error) {
				t.Fatal("captured operations added a second time")
				return nil, nil
			},
		}
		err := p.prepareGraphWith(map[string]driver.DevicePtr{"x": 8}, ops)
		if calls != 1 || p.graphObj != graph {
			t.Fatal("lost capture owner or duplicated enqueue")
		}
		if fail {
			if !errors.Is(err, enqueueFailure) || !errors.Is(err, endFailure) || !p.closed || p.graphExec != nil {
				t.Fatalf("capture failure state: %v", err)
			}
		} else if err != nil || p.closed || p.graphExec != exec {
			t.Fatalf("capture success state: %v", err)
		}
	}
}
