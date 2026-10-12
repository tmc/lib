package cudamega

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

func invocationProgram(t *testing.T, extent uint64, maximum uint32) *Program {
	t.Helper()
	g, err := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "a", Scope: cudataskgraph.ScopeDevice, Extent: extent},
		{ID: "b", Rank: 1, Deps: []string{"a"}, Scope: cudataskgraph.ScopeDevice, Extent: extent},
	})
	if err != nil {
		t.Fatal(err)
	}
	p, err := NewProgram(g, map[string]Operation{"a": {Kind: Add, Scalar: 3}, "b": {Kind: Multiply, Scalar: 2}}, Grid{CTAs: 2, ThreadsPerCTA: 32, MaxElements: maximum}, SM80)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestProgramInvocation(t *testing.T) {
	for _, tc := range []struct {
		name               string
		declared, elements uint64
		maximum            uint32
		want               string
	}{
		{"runtime", 0, 3, 0, ""},
		{"maximum", 0, 3, 3, ""},
		{"above maximum", 0, 4, 3, "exceeds maximum"},
		{"exact", 3, 3, 0, ""},
		{"short", 3, 2, 0, "declares extent"},
		{"long", 3, 4, 0, "declares extent"},
		{"zero", 0, 0, 0, "invalid finite program extent"},
		{"uint32", 0, uint64(math.MaxUint32) + 1, 0, "invalid finite program extent"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := invocationProgram(t, tc.declared, tc.maximum)
			got, err := p.Invocation(tc.elements)
			if tc.want != "" {
				if err == nil || !strings.Contains(err.Error(), tc.want) || got != (Invocation{}) {
					t.Fatalf("Invocation = %+v, %v, want zero value and %q", got, err, tc.want)
				}
				return
			}
			want := Invocation{Elements: uint32(tc.elements), ScratchWords: 2 * int(tc.elements), StartOrderWords: 5}
			if err != nil || got != want {
				t.Fatalf("Invocation = %+v, %v, want %+v", got, err, want)
			}
		})
	}
	var p *Program
	if got, err := p.Invocation(3); err == nil || got != (Invocation{}) {
		t.Fatalf("nil program = %+v, %v", got, err)
	}
}

func TestInvocationSizes(t *testing.T) {
	for _, tc := range []struct {
		name                       string
		tasks, elements, ctas, max uint64
		refuse                     bool
	}{
		{"valid", 2, 3, 2, math.MaxInt64, false},
		{"scratch byte boundary", 2, 3, 2, 40, false},
		{"scratch bytes overflow", 2, 6, 1, 40, true},
		{"order bytes overflow", 2, 3, 2, 39, true},
		{"scratch int32 overflow", 1024, math.MaxUint32, 1, math.MaxInt32, true},
		{"order int32 overflow", 1024, 1, math.MaxUint32, math.MaxInt32, true},
		{"zero tasks", 0, 1, 1, math.MaxInt64, true},
		{"task limit", 1025, 1, 1, math.MaxInt64, true},
		{"zero ctas", 1, 1, 0, math.MaxInt64, true},
		{"cta range", 1, 1, uint64(math.MaxUint32) + 1, math.MaxInt64, true},
		{"element range", 1024, math.MaxUint64, 1, math.MaxInt64, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := invocationSizes(tc.tasks, tc.elements, tc.ctas, tc.max)
			if (err != nil) != tc.refuse || tc.refuse && got != (Invocation{}) {
				t.Fatalf("sizes = %+v, %v, want refusal %v", got, err, tc.refuse)
			}
		})
	}
}

func ExampleProgram_Invocation() {
	g, _ := cudataskgraph.New([]cudataskgraph.Task{{ID: "add", Scope: cudataskgraph.ScopeDevice, Extent: 3}})
	p, _ := NewProgram(g, map[string]Operation{"add": {Kind: Add, Scalar: 1}}, Grid{CTAs: 1, ThreadsPerCTA: 32, MaxElements: 3}, SM80)
	inv, err := p.Invocation(3)
	fmt.Println(inv.Elements, inv.ScratchWords, inv.StartOrderWords, err)
	// Output: 3 3 2 <nil>
}

func ExampleInvocation() {
	g, _ := cudataskgraph.New([]cudataskgraph.Task{{ID: "add", Scope: cudataskgraph.ScopeDevice}})
	p, _ := NewProgram(g, map[string]Operation{"add": {Kind: Add}}, Grid{CTAs: 1, ThreadsPerCTA: 32}, SM80)
	inv, _ := p.Invocation(2)
	fmt.Println(inv.ScratchWords*4, inv.StartOrderWords*8)
	// Output: 8 16
}
