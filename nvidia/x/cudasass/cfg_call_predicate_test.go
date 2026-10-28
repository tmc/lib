package cudasass_test

import (
	"fmt"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

func callPredicateCFG(t *testing.T, predicate, opcode, target string) (*cudasass.Function, *cudasass.CFG) {
	t.Helper()
	listing := fmt.Sprintf(".text.calls:\n /*0000*/ %s %s %s ;\n /*0010*/ EXIT ;\n /*0020*/ RET ;\n", predicate, opcode, target)
	d, err := cudasass.Parse(strings.NewReader(listing))
	if err != nil {
		t.Fatal(err)
	}
	fn := d.Function("calls")
	if fn == nil {
		t.Fatal("parsed function missing")
	}
	g, err := cudasass.NewCFG(fn)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(g.Blocks, cudasass.BasicBlocks(fn)) {
		t.Fatal("CFG changed the BasicBlocks partition")
	}
	var starts []uint64
	for _, b := range g.Blocks {
		starts = append(starts, b.Start)
	}
	if !slices.Equal(starts, []uint64{0, 0x10, 0x20}) {
		t.Fatalf("block starts = %#x, want [0x0 0x10 0x20]", starts)
	}
	return fn, g
}

func TestCFGCallPredicates(t *testing.T) {
	for _, opcode := range []string{"CALL", "CALL.REL.NOINC"} {
		for _, tt := range []struct {
			predicate string
			call      bool
		}{
			{"@!PT", false},
			{"@!UPT", false},
			{"", true},
			{"@PT", true},
			{"@UPT", true},
			{"@P0", true},
			{"@!P0", true},
		} {
			t.Run(opcode+"/"+tt.predicate, func(t *testing.T) {
				_, g := callPredicateCFG(t, tt.predicate, opcode, "0x20")
				wantEdges := []cudasass.FlowEdge{{From: 0, To: 1, Kind: cudasass.FallThrough}}
				wantSuccessors := []int{1}
				if tt.call {
					wantEdges = append(wantEdges, cudasass.FlowEdge{From: 0, To: 2, Kind: cudasass.Call})
					wantSuccessors = append(wantSuccessors, 2)
				}
				if !slices.Equal(g.Edges, wantEdges) || !slices.Equal(g.Successors(0), wantSuccessors) {
					t.Fatalf("edges = %v, successors = %v; want %v, %v", g.Edges, g.Successors(0), wantEdges, wantSuccessors)
				}
				dot := g.DOT()
				if got := strings.Contains(dot, `"BB0" -> "BB2" [label="call"];`); got != tt.call {
					t.Fatalf("DOT call edge = %v, want %v:\n%s", got, tt.call, dot)
				}
				if !strings.Contains(dot, `"BB0" -> "BB1" [label="fallthrough"];`) || !strings.Contains(dot, `"BB2" [label="0020: RET ;\l"];`) {
					t.Fatalf("DOT lost fallthrough or retained target block:\n%s", dot)
				}
				if !slices.Equal(cudasass.ListDOTGraphs(dot), []string{"calls"}) {
					t.Fatalf("DOT consumer graph names = %v", cudasass.ListDOTGraphs(dot))
				}
				if extracted, err := cudasass.ExtractDOTGraph(dot, "calls"); err != nil || strings.TrimSpace(extracted) != strings.TrimSpace(dot) {
					t.Fatalf("DOT extraction = %q, %v", extracted, err)
				}
			})
		}
	}
}

func TestCFGUnresolvedCallPredicates(t *testing.T) {
	for _, predicate := range []string{"", "@PT", "@P0", "@!PT", "@!UPT"} {
		t.Run(predicate, func(t *testing.T) {
			_, g := callPredicateCFG(t, predicate, "CALL", "external_function")
			want := []cudasass.FlowEdge{{From: 0, To: 1, Kind: cudasass.FallThrough}}
			if !slices.Equal(g.Edges, want) || !slices.Equal(g.Successors(0), []int{1}) {
				t.Fatalf("external call edges = %v, successors = %v", g.Edges, g.Successors(0))
			}
		})
	}
}

func TestCFGBranchPredicateControl(t *testing.T) {
	for _, tt := range []struct {
		predicate string
		want      []cudasass.FlowEdge
	}{
		{"@!PT", []cudasass.FlowEdge{{From: 0, To: 1, Kind: cudasass.FallThrough}}},
		{"@PT", []cudasass.FlowEdge{{From: 0, To: 2, Kind: cudasass.Taken}}},
		{"@P0", []cudasass.FlowEdge{{From: 0, To: 1, Kind: cudasass.FallThrough}, {From: 0, To: 2, Kind: cudasass.Taken}}},
	} {
		t.Run(tt.predicate, func(t *testing.T) {
			_, g := callPredicateCFG(t, tt.predicate, "BRA", "0x20")
			if !slices.Equal(g.Edges, tt.want) {
				t.Fatalf("branch edges = %v, want %v", g.Edges, tt.want)
			}
		})
	}
}
