package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// This is a recorded CPU compiler/disassembler fixture, not sampled GPU PCs.
func TestSidecarRecordedLineAttribution(t *testing.T) {
	dir := filepath.Join("testdata", "sidecar", "recorded-bar")
	cuda, err := os.ReadFile(filepath.Join(dir, "bar.cu"))
	if err != nil {
		t.Fatal(err)
	}
	metadata, err := os.ReadFile(filepath.Join(dir, "bar.egraph.json"))
	if err != nil {
		t.Fatal(err)
	}
	source, err := egraph.ReadMetadata(strings.NewReader(string(metadata)), string(cuda))
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(dir, "bar.sass"))
	if err != nil {
		t.Fatal(err)
	}
	dis, err := cudasass.ParseString(string(raw))
	if err != nil {
		t.Fatal(err)
	}
	type location struct {
		line       int
		class      egraph.Id
		attributed bool
	}
	want := map[uint64]location{0x0: {64, 0, false}, 0xe0: {28, 2, true}, 0x570: {59, 4, true}, 0x5a0: {77, 0, false}}
	seen := make(map[uint64]bool)
	for _, f := range dis.Functions {
		if f.Name != "chain" {
			continue
		}
		for _, inst := range f.Instructions {
			w, ok := want[inst.Address]
			if !ok {
				continue
			}
			if filepath.Base(inst.SourceFile) != "bar.cu" || inst.SourceLine != w.line {
				t.Fatalf("PC %#x location = %s:%d, want bar.cu:%d", inst.Address, inst.SourceFile, inst.SourceLine, w.line)
			}
			class, ok := source.ClassAt(inst.SourceLine)
			if ok != w.attributed || ok && class != w.class {
				t.Fatalf("PC %#x class = %d/%v, want %d/%v", inst.Address, class, ok, w.class, w.attributed)
			}
			seen[inst.Address] = true
		}
	}
	if len(seen) != len(want) {
		t.Fatalf("recorded PCs found %d, want %d", len(seen), len(want))
	}
	edited, err := egraph.ReadMetadata(strings.NewReader(string(metadata)), string(cuda)+"\n// edited\n")
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := edited.ClassAt(28); ok {
		t.Fatal("edited source retained recorded line attribution")
	}
}
