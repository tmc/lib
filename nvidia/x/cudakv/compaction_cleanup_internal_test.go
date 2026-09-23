package cudakv

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudagraph"
)

func TestCompactionCaptureOwnership(t *testing.T) {
	captureErr := errors.New("capture failed")
	copyErr := errors.New("copy failed")
	instantiateErr := errors.New("instantiate failed")
	for _, stage := range []string{"capture", "copy", "both", "instantiate", "success"} {
		t.Run(stage, func(t *testing.T) {
			graph, exec := &cudagraph.Graph{}, &cudagraph.Exec{}
			instantiations := 0
			cg, err := captureCompaction(CompactionPlan{}, func(fn func()) (*cudagraph.Graph, error) {
				fn()
				if stage == "capture" || stage == "both" {
					return graph, captureErr
				}
				return graph, nil
			}, func(g *cudagraph.Graph) (*cudagraph.Exec, error) {
				instantiations++
				if g != graph {
					t.Fatal("wrong graph")
				}
				if stage == "instantiate" {
					return exec, instantiateErr
				}
				return exec, nil
			}, func() error {
				if stage == "copy" || stage == "both" {
					return copyErr
				}
				return nil
			})
			if cg == nil || cg.graph != graph || cg.closed != (stage != "success") {
				t.Fatal("lost owner or wrong execution state")
			}
			if errors.Is(err, captureErr) != (stage == "capture" || stage == "both") ||
				errors.Is(err, copyErr) != (stage == "copy" || stage == "both") ||
				errors.Is(err, instantiateErr) != (stage == "instantiate") ||
				(err == nil) != (stage == "success") {
				t.Fatalf("unexpected error: %v", err)
			}
			wantExec := stage == "instantiate" || stage == "success"
			if (cg.exec == exec) != wantExec || (instantiations == 1) != wantExec {
				t.Fatal("wrong instantiation ownership or order")
			}
			if err != nil && cg.Launch(nil) == nil {
				t.Fatal("failed construction accepted launch")
			}
			freedExec, freedGraph := 0, 0
			if err := cg.destroyWith(func(*cudagraph.Exec) error { freedExec++; return nil }, func(*cudagraph.Graph) error { freedGraph++; return nil }); err != nil {
				t.Fatal(err)
			}
			if freedGraph != 1 || (freedExec == 1) != wantExec {
				t.Fatal("cleanup lost resource")
			}
		})
	}
}

func TestCompactionDestroyRetry(t *testing.T) {
	failure := errors.New("destroy failed")
	for _, stage := range []string{"exec", "graph"} {
		t.Run(stage, func(t *testing.T) {
			graph, exec := &cudagraph.Graph{}, &cudagraph.Exec{}
			cg := &CompactionGraph{graph: graph, exec: exec}
			execCalls, graphCalls := 0, 0
			retry := false
			destroyExec := func(e *cudagraph.Exec) error {
				execCalls++
				if e != exec {
					t.Fatal("wrong executable")
				}
				if !retry && stage == "exec" {
					return failure
				}
				return nil
			}
			destroyGraph := func(g *cudagraph.Graph) error {
				graphCalls++
				if g != graph {
					t.Fatal("wrong graph")
				}
				if !retry && stage == "graph" {
					return failure
				}
				return nil
			}
			if err := cg.destroyWith(destroyExec, destroyGraph); !errors.Is(err, failure) {
				t.Fatal(err)
			}
			if !cg.closed || cg.graph != graph || (cg.exec != nil) != (stage == "exec") {
				t.Fatal("lost failed cleanup owner")
			}
			if stage == "exec" && graphCalls != 0 {
				t.Fatal("destroyed dependency after executable failure")
			}
			if err := cg.Launch(nil); err == nil {
				t.Fatal("launch accepted during failed teardown")
			}
			retry = true
			for range 2 {
				if err := cg.destroyWith(destroyExec, destroyGraph); err != nil {
					t.Fatal(err)
				}
			}
			wantExec, wantGraph := 2, 1
			if stage == "graph" {
				wantExec, wantGraph = 1, 2
			}
			if execCalls != wantExec || graphCalls != wantGraph || cg.exec != nil || cg.graph != nil {
				t.Fatalf("destroy counts=%d/%d", execCalls, graphCalls)
			}
		})
	}
}

func TestCompactionRejectsClosedPool(t *testing.T) {
	pc, err := NewTestPagedCache(1, 1, 1, 1, 4)
	if err != nil {
		t.Fatal(err)
	}
	tree := NewRadixTree(16, 1)
	if _, _, err := tree.Insert([]int32{1}, []int{2}); err != nil {
		t.Fatal(err)
	}
	if err := pc.Close(); err != nil {
		t.Fatal(err)
	}
	cg := &CompactionGraph{plan: CompactionPlan{Remappings: map[int]int{2: 0}}}
	if err := cg.ApplyToCache(pc, nil, tree); err == nil {
		t.Fatal("closed pool accepted remapping")
	}
	_, blocks, _ := tree.MatchPrefix([]int32{1})
	if len(blocks) != 1 || blocks[0] != 2 {
		t.Fatalf("rejected remapping changed radix tree: %v", blocks)
	}
}
