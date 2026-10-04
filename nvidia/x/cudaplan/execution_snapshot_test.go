package cudaplan

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
)

func TestCompilePrivatelyFreezesExecutionMetadata(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{2, 3}, cudair.Float32)
	y := g.AddReduce("sum", x, 1)
	y.SourceLoc.Callers = []cudair.Frame{{Function: "original"}}
	if err := g.SetOutputs(y); err != nil {
		t.Fatal(err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	p, err := Compile(snap, WithFiniteMega(true))
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	raw := string(p.megaArtifact.Raw)
	x.Shape[1] = 100
	x.Strides[0] = 100
	x.Type = cudair.Int32
	y.Name = "changed"
	y.Axes[0] = 0
	y.SourceLoc.Callers[0].Function = "changed"
	y.Inputs[0] = y
	frozen := p.snap.NodeByID(y.ID)
	if frozen.Name == "changed" || frozen.Axes[0] != 1 || frozen.Inputs[0].Shape[1] != 3 || frozen.Inputs[0].Type != cudair.Float32 || frozen.SourceLoc.Callers[0].Function != "original" {
		t.Fatal("compiled metadata follows public node mutation")
	}
	b := NewBindings()
	b.inputs["x"] = BufferBinding{typed: true, dtype: cudair.Float32}
	if err := b.validateTypes(p.snap); err != nil {
		t.Fatal(err)
	}
	if string(p.megaArtifact.Raw) != raw {
		t.Fatal("artifact changed")
	}
}

func TestCompileExcludesUnreachableUnsupportedWork(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{4}, cudair.Float32)
	y := g.AddUnary("neg", x)
	unused := g.AddInput("unused", []int{4}, cudair.Float32)
	dead := g.AddUnary("exp", unused)
	if err := g.SetOutputs(y); err != nil {
		t.Fatal(err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	p, err := Compile(snap, WithFiniteMega(true))
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	if len(p.Steps()) != 1 || len(p.requiredInputs) != 1 || p.snap.NodeByID(dead.ID) != nil || strings.Contains(string(p.megaArtifact.Raw), "ex2") {
		t.Fatal("unreachable work retained in execution plan")
	}
}

func TestWorkspaceReductionDoesNotAliasItsLiveInput(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{1024, 4}, cudair.Float32)
	a := g.AddUnary("neg", x)
	b := g.AddReduce("sum", a, 1)
	c := g.AddUnary("neg", b)
	if err := g.SetOutputs(c); err != nil {
		t.Fatal(err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	p, err := Compile(snap, WithFiniteMega(true))
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	aOff, bOff := p.intermediateLoc[a.Name], p.intermediateLoc[b.Name]
	if aOff < bOff+b.SizeBytes() && bOff < aOff+a.SizeBytes() {
		t.Fatal("parallel reduction output overlaps its still-live input")
	}
	if p.intermediateLoc[c.Name] != aOff {
		t.Fatal("dead buffer no longer reused after its final consumer")
	}
}
