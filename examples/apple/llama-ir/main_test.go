package main

import "testing"

func TestLlamaIR(t *testing.T) {
	if err := run(); err != nil {
		t.Fatalf("run failed: %v", err)
	}
}

func Example() {
	if err := run(); err != nil {
		panic(err)
	}
	// Output:
	// validated symbolic decoder graph: nodes=23 output=[8 64] scratch=16384 bytes
	// lowering unsupported: compile: node 2: operation "rmsnorm" unsupported
}

func TestDecoderGraphValid(t *testing.T) {
	g, out := decoderGraph()
	if err := g.Validate(); err != nil {
		t.Fatal(err)
	}
	got := out.Shape()
	if len(got) != 2 || got[0] != sequence || got[1] != dimension {
		t.Fatalf("output shape=%v", got)
	}
	if _, err := out.Compile("DecoderLayer"); err == nil {
		t.Fatal("attention graph unexpectedly lowered")
	}
}
