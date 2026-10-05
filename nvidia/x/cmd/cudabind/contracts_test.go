package main

import (
	"github.com/tmc/lib/nvidia/x/internal/libclang"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestLaunchContractSyntax(t *testing.T) {
	params := []param{{cName: "out", name: "out", typ: "cudamem.Slice[float32]", slice: true}, {cName: "n", name: "n", typ: "int32"}, {cName: "factor", name: "factor", typ: "float32"}}
	for _, tt := range []struct {
		name, text string
		valid      bool
	}{
		{"extent", "len(out) >= n", true},
		{"coverage", "grid.x * block.x >= n", true},
		{"shared", "sharedBytes >= (n + 1) * 4", true},
		{"block", "block.x == 128", true},
		{"missing_slice", "len(absent) >= n", false},
		{"missing_count", "len(out) >= absent", false},
		{"float_count", "len(out) >= factor", false},
		{"slice_count", "len(out) >= out", false},
		{"scalar_length", "len(n) >= 1", false},
		{"call", "len(out) >= evil(n)", false},
		{"division", "len(out) >= n / 2", false},
		{"negative", "len(out) >= -1", false},
		{"bad_axis", "grid.w >= n", false},
		{"overflow", "len(out) >= 18446744073709551616", false},
		{"assignment", "out = n", false},
		{"broken", "len(out) >=", false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseLaunchContracts("// kernel documentation\n// cudabind: "+tt.text, params)
			if (err == nil) != tt.valid {
				t.Fatalf("parse = %v, want valid %v", err, tt.valid)
			}
			if tt.valid && (len(got) != 1 || got[0].text != tt.text) {
				t.Fatalf("contracts = %#v", got)
			}
		})
	}
	if _, err := parseLaunchContracts("// cudabind: "+strings.Repeat("(", 20)+"n"+strings.Repeat(")", 20)+" >= 1", params); err == nil {
		t.Fatal("deep expression accepted")
	}
}

func TestWitnessMeaningfulBytes(t *testing.T) {
	records := []*record{{goName: "inner", size: 8, fields: []field{{cName: "x", goName: "X", typ: "uint8", off: 0, size: 1}}}, {goName: "outer", size: 24, fields: []field{{cName: "items", goName: "Items", typ: "[2]inner", off: 8, size: 16}}}}
	got, err := meaningfulBytes("outer", 24, records)
	if err != nil || len(got) != 2 || got[0] != 8 || got[1] != 16 {
		t.Fatalf("mask = %v, %v", got, err)
	}
	if _, err := meaningfulBytes("opaque", 8, []*record{{goName: "opaque", size: 8, opaque: true}}); err == nil {
		t.Fatal("opaque parameter accepted")
	}
}

func TestAlignmentDelegated(t *testing.T) {
	b, err := os.ReadFile("gen_contract.go")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), "cudakernel.CheckPointerAlignment(") {
		t.Fatal("pointer alignment bypasses cudakernel")
	}
	if regexp.MustCompile(`(?:pointer|ptr|uint64\(pointer\)|uintptr\(pointer\))\s*%`).Match(b) || strings.Contains(string(b), "unsafe.Alignof") {
		t.Fatal("second pointer alignment definition")
	}
}

func TestGenerate_BadAnnotationFailsGeneration(t *testing.T) {
	lib, err := libclang.Open(os.Getenv("CUDABIND_LIBCLANG"))
	if err != nil {
		t.Skip(err)
	}
	if _, err := os.Stat(filepath.Join(lib.ResourceDir(), "include", "__clang_cuda_runtime_wrapper.h")); err != nil {
		t.Skip("compatible CUDA libclang resource required")
	}
	for _, tt := range []struct{ name, contract string }{{"missing", "len(missing) >= n"}, {"malformed", "len(out) >="}, {"unsupported", "len(out) >= n / 2"}} {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			t.Chdir(dir)
			t.Setenv("GOPACKAGE", "testbad")
			src := "// bounded writes output.\n// cudabind: " + tt.contract + "\nextern \"C\" __global__ void bounded(float* out, int n) {}\n"
			// The annotation must survive a preceding unannotated prototype.
			src = "extern \"C\" __global__ void bounded(float* out, int n);\n" + src
			if err := os.WriteFile("bad.cu", []byte(src), 0600); err != nil {
				t.Fatal(err)
			}
			cfg, err := newConfig([]string{"-type", "badModule", "-std", "c++17", "bad.cu"})
			if err != nil {
				t.Fatal(err)
			}
			err = run(cfg)
			if err == nil || !strings.Contains(err.Error(), "launch contract") {
				t.Fatalf("bad declaration error = %v", err)
			}
			if _, err := os.Stat("bad.cu.gen.go"); !os.IsNotExist(err) {
				t.Fatalf("failed generation wrote output: %v", err)
			}
		})
	}
}

func FuzzLaunchContracts(f *testing.F) {
	for _, s := range []string{"len(out) >= n", "len(missing) >= n", "grid.x * block.x >= n", "n >= -1"} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, s string) {
		params := []param{{cName: "out", slice: true}, {cName: "n", typ: "int32"}}
		got, err := parseLaunchContracts("// cudabind: "+s, params)
		if err == nil {
			var g gen
			next := 0
			for _, c := range got {
				g.contractExpr("testModule", c.left, params, &next, "sharedBytes")
				g.contractExpr("testModule", c.right, params, &next, "sharedBytes")
			}
		}
	})
}
