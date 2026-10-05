package main

import (
	"strings"
	"testing"
)

func TestParameterWitnessRetainsReadback(t *testing.T) {
	g := new(gen)
	m := &module{kernels: []*kernel{{name: "sentinel", params: []param{{cName: "value", typ: "uint32", cType: "unsigned int", width: 4, abiAlignment: 4}}}}}
	if err := g.parameterWitness(&config{typeName: "fixture"}, m, "compileFixture", "newFixture"); err != nil {
		t.Fatal(err)
	}
	s := g.String()
	read := strings.Index(s, "out.CopyToHost(got)")
	receipt := strings.Index(s, `"parameter-abi", "sentinel", got, want`)
	check := strings.Index(s, "if got[0] != want[0]")
	if read < 0 || receipt <= read || check <= receipt {
		t.Fatalf("readback, receipt and comparison order: %d, %d, %d", read, receipt, check)
	}
	if !strings.Contains(s, "Observed, Expected []uint64") || !strings.Contains(s, "if err != nil { t.Fatal(err) }; t.Logf") {
		t.Fatal("receipt loses raw arrays or ignores serialization error")
	}
}
