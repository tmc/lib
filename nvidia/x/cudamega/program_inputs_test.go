package cudamega_test

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

// inputsProgram builds a two-source fan-in and binds it to a program. The
// consumer c depends on a then b, so the worker's single data input is a.
func inputsProgram(inputs []string, extents [3]uint64) (*cudamega.Program, error) {
	g, err := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "a", Rank: 0, Scope: cudataskgraph.ScopeDevice, Extent: extents[0]},
		{ID: "b", Rank: 0, Scope: cudataskgraph.ScopeDevice, Extent: extents[1]},
		{ID: "c", Deps: []string{"a", "b"}, Inputs: inputs, Rank: 1, Scope: cudataskgraph.ScopeDevice, Extent: extents[2]},
	})
	if err != nil {
		return nil, err
	}
	ops := map[string]cudamega.Operation{
		"a": {Kind: cudamega.Add, Scalar: 1},
		"b": {Kind: cudamega.Add, Scalar: 2},
		"c": {Kind: cudamega.Multiply, Scalar: 3},
	}
	return cudamega.NewProgram(g, ops, cudamega.Grid{CTAs: 2, ThreadsPerCTA: 64}, cudamega.SM121a)
}

var inputsTests = []struct {
	name    string
	inputs  []string
	extents [3]uint64
	refuse  string // substring of the refusal; empty means accepted
}{
	{name: "undeclared", inputs: nil},
	{name: "first-dependency", inputs: []string{"a"}},
	{name: "uniform-extent", extents: [3]uint64{64, 64, 64}},
	{name: "two-inputs", inputs: []string{"a", "b"}, refuse: `task "c" reads 2 inputs`},
	{name: "second-dependency", inputs: []string{"b"}, refuse: `task "c" reads input "b"`},
	{name: "mixed-extent", extents: [3]uint64{64, 64, 128}, refuse: "extent"},
	{name: "partly-declared-extent", extents: [3]uint64{64, 0, 64}, refuse: "extent"},
	{name: "undeclared-first-extent", extents: [3]uint64{0, 64, 64}, refuse: "extent"},
}

// TestNewProgramRefusesUnexecutableInputs is the partial repair of the
// worker's one-input, one-extent binding: what it cannot execute is refused at
// construction instead of being run against the first dependency and the
// launch extent.
func TestNewProgramRefusesUnexecutableInputs(t *testing.T) {
	for _, tt := range inputsTests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := inputsProgram(tt.inputs, tt.extents)
			switch {
			case tt.refuse == "" && err != nil:
				t.Fatalf("NewProgram: %v", err)
			case tt.refuse != "" && err == nil:
				t.Fatalf("NewProgram accepted inputs %q extents %v as a %d-step program", tt.inputs, tt.extents, len(p.Steps()))
			case tt.refuse != "" && !strings.Contains(err.Error(), tt.refuse):
				t.Fatalf("NewProgram: %v, want refusal containing %q", err, tt.refuse)
			}
		})
	}
}

// TestUnsupportedInputsIsDistinguishable checks that the refusal is the one
// sentinel, and that refusals for other reasons are not it.
func TestUnsupportedInputsIsDistinguishable(t *testing.T) {
	for _, tt := range inputsTests {
		_, err := inputsProgram(tt.inputs, tt.extents)
		if got, want := errors.Is(err, cudamega.ErrUnsupportedInputs), tt.refuse != ""; got != want {
			t.Errorf("%s: errors.Is(%v, ErrUnsupportedInputs) = %v, want %v", tt.name, err, got, want)
		}
	}
	g, err := cudataskgraph.New([]cudataskgraph.Task{{ID: "a", Rank: 0, Scope: cudataskgraph.ScopeCluster}})
	if err != nil {
		t.Fatal(err)
	}
	_, err = cudamega.NewProgram(g, map[string]cudamega.Operation{"a": {Kind: cudamega.Add}}, cudamega.Grid{CTAs: 1, ThreadsPerCTA: 32}, cudamega.SM121a)
	if err == nil || errors.Is(err, cudamega.ErrUnsupportedInputs) {
		t.Fatalf("cluster-scope refusal = %v, want a refusal that is not ErrUnsupportedInputs", err)
	}
}

func ExampleNewProgram_unsupportedInputs() {
	g, err := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "x", Rank: 0, Scope: cudataskgraph.ScopeDevice},
		{ID: "y", Rank: 0, Scope: cudataskgraph.ScopeDevice},
		{ID: "sum", Deps: []string{"x", "y"}, Inputs: []string{"x", "y"}, Rank: 1, Scope: cudataskgraph.ScopeDevice},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	ops := map[string]cudamega.Operation{
		"x":   {Kind: cudamega.Add, Scalar: 1},
		"y":   {Kind: cudamega.Add, Scalar: 2},
		"sum": {Kind: cudamega.Add},
	}
	_, err = cudamega.NewProgram(g, ops, cudamega.Grid{CTAs: 1, ThreadsPerCTA: 32}, cudamega.SM121a)
	fmt.Println(errors.Is(err, cudamega.ErrUnsupportedInputs))
	fmt.Println(err)
	// Output:
	// true
	// task "sum" reads 2 inputs and the worker reads one: unsupported task inputs
}
