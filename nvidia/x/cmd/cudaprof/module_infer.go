package main

import (
	"bytes"
	"debug/elf"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// moduleInference counts what inferModules did with the kernel samples that
// named no module.
type moduleInference struct {
	Inferred  int // labelled with the one module whose CUBIN defines the symbol
	Absent    int // symbol in no captured CUBIN a sample can name
	Ambiguous int // symbol in more than one captured CUBIN
}

// inferModules labels each kernel sample that carries no "module_id" with
// the module whose CUBIN defines the sample's kernel symbol, when exactly
// one captured module does, and marks it "module_inferred"="symbol" so a
// reader can tell the join was made by name rather than by the producer.
//
// A producer labels only the kernels it followed back to a module. Launches
// through the runtime API go through cuLibrary and cuKernel handles the
// agent does not track, and without this every such kernel drops out of the
// per-module views of a capture with more than one module. A symbol defined
// by no module, or by several, is left unlabelled and counted: the name is
// then no evidence of which image ran.
//
// A module the driver did not number is still a candidate, so it can make a
// symbol ambiguous, but a sample is never labelled with it: it has no label
// to carry.
func inferModules(cap *cudacapture.Capture) moduleInference {
	var n moduleInference
	if cap.Profile == nil || len(cap.Modules) < 2 {
		return n
	}
	defined := make(map[string][]int)
	for i, mod := range cap.Modules {
		for _, sym := range cubinFunctions(cap.Files[mod.Code.CUBIN]) {
			defined[sym] = append(defined[sym], i)
		}
	}
	for _, s := range cap.Profile.Sample {
		if subtype := s.Label["subtype"]; len(subtype) == 1 && subtype[0] != "kernel" {
			continue
		}
		if len(s.Label["module_id"]) != 0 {
			continue
		}
		name := s.Label["name"]
		if len(name) != 1 {
			n.Absent++
			continue
		}
		mods := defined[name[0]]
		switch {
		case len(mods) == 0:
			n.Absent++
		case len(mods) > 1:
			n.Ambiguous++
		case moduleLabel(cap.Modules[mods[0]]) == "":
			n.Absent++
		default:
			s.Label["module_id"] = []string{moduleLabel(cap.Modules[mods[0]])}
			s.Label["module_inferred"] = []string{"symbol"}
			n.Inferred++
		}
	}
	return n
}

// cubinFunctions returns the names of the functions a CUBIN holds code for,
// each once: the compiler gives every function its own .text.<name> section,
// and the section survives where the symbol table does not. Anything that is
// not a CUDA ELF image holds nothing.
func cubinFunctions(cubin []byte) []string {
	if len(cubin) == 0 {
		return nil
	}
	f, err := elf.NewFile(bytes.NewReader(cubin))
	if err != nil {
		return nil
	}
	defer f.Close()
	if f.Machine != elf.EM_CUDA {
		return nil
	}
	seen := make(map[string]bool)
	var names []string
	for _, sec := range f.Sections {
		name, ok := strings.CutPrefix(sec.Name, ".text.")
		if !ok || name == "" || seen[name] {
			continue
		}
		seen[name] = true
		names = append(names, name)
	}
	return names
}
